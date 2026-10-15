"""Pull real per-pod KPI time series from the cluster's Thanos, aligned to the topology.

No synthetic data anywhere in this file. Four counters per pod per step, the same shape the
solver expects: [steps][nodes][kpis].

    python experiments/oberon_live/fetch_metrics.py --minutes 90 --step 30 --out counters.npz
"""
from __future__ import annotations
import argparse, json, ssl, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

QUERIES = {   # kpi name -> promql template, {sel} is the pod/namespace selector
    "cpu_rate":           'sum by (pod) (rate(container_cpu_usage_seconds_total{{{sel},container!=""}}[2m]))',
    "memory_working_set": 'sum by (pod) (container_memory_working_set_bytes{{{sel},container!=""}})',
    "net_rx_rate":        'sum by (pod) (rate(container_network_receive_bytes_total{{{sel}}}[2m]))',
    "net_tx_rate":        'sum by (pod) (rate(container_network_transmit_bytes_total{{{sel}}}[2m]))',
}


def host_and_token() -> tuple[str, str]:
    h = subprocess.run(["kubectl", "-n", "openshift-monitoring", "get", "route",
                        "thanos-querier", "-o", "jsonpath={.spec.host}"],
                       capture_output=True, text=True, check=True).stdout.strip()
    t = subprocess.run(["oc", "whoami", "-t"], capture_output=True, text=True, check=True).stdout.strip()
    return h, t


def query_range(host: str, token: str, q: str, start: float, end: float, step: int) -> dict:
    url = f"https://{host}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": q, "start": f"{start:.0f}", "end": f"{end:.0f}", "step": str(step)})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
        return json.loads(r.read())


def fetch(topo: dict, minutes: int, step: int, end: float | None = None):
    host, token = host_and_token()
    end = end or time.time()
    start = end - minutes * 60
    namespaces = sorted({n["namespace"] for n in topo["nodes"]})
    sel = 'namespace=~"%s"' % "|".join(namespaces)
    # Map by network FUNCTION, not pod instance. A restart replaces the pod and changes its
    # name; the function is what an operator tracks and what the topology models. This also
    # means a pod that is replaced mid-window contributes both halves of its own series.
    fn_of_node = {n["function"]: n["id"] for n in topo["nodes"]}
    index: dict[str, int] = {}
    for ns in namespaces:
        j = subprocess.run(["kubectl", "-n", ns, "get", "pods", "-o",
                            "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}"],
                           capture_output=True, text=True).stdout.split()
        for pod in j:
            fn = "-".join(pod.split("-")[:-2]) if pod.count("-") >= 2 else pod
            if fn in fn_of_node:
                index[pod] = fn_of_node[fn]
    N, K = len(topo["nodes"]), len(topo["kpi_names"])
    steps = int((end - start) // step) + 1
    X = np.full((steps, N, K), np.nan)
    grid = np.array([start + i * step for i in range(steps)])

    for k, name in enumerate(topo["kpi_names"]):
        res = query_range(host, token, QUERIES[name].format(sel=sel), start, end, step)
        if res.get("status") != "success":
            raise RuntimeError(f"{name}: {res}")
        hit = 0
        for series in res["data"]["result"]:
            pod = series["metric"].get("pod")
            if pod not in index:
                continue
            hit += 1
            j = index[pod]
            ts = np.array([float(a) for a, _ in series["values"]])
            vs = np.array([float(b) for _, b in series["values"]])
            col = np.interp(grid, ts, vs, left=np.nan, right=np.nan)
            cur = X[:, j, k]
            X[:, j, k] = np.where(np.isfinite(cur), np.where(np.isfinite(col), cur + col, cur), col)
        print(f"  {name:<20} {hit:3d}/{N} pods matched", flush=True)
    return X, grid


def standardise(X: np.ndarray) -> np.ndarray:
    """Per node per KPI: z-score against this window, so counters in different units compare.
    Missing series become zero, which reads as 'nominal' rather than as a fault."""
    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            v = X[:, j, k]
            m = np.isfinite(v)
            if m.sum() < 5 or np.nanstd(v[m]) < 1e-12:
                continue
            Z[:, j, k] = np.nan_to_num((v - np.nanmean(v[m])) / np.nanstd(v[m]), nan=0.0)
    return Z


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="experiments/oberon_live/topology.json")
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--out", default="experiments/oberon_live/counters.npz")
    a = ap.parse_args()
    topo = json.loads(Path(a.topology).read_text())
    print(f"fetching {a.minutes} min at {a.step}s from Thanos for {len(topo['nodes'])} pods")
    X, grid = fetch(topo, a.minutes, a.step)
    cov = float(np.isfinite(X).mean())
    Z = standardise(X)
    np.savez_compressed(a.out, raw=X, z=Z, t=grid)
    print(f"\nshape {X.shape}  coverage {cov:.1%}  -> {a.out}")
    print(f"standardised range [{Z.min():.1f}, {Z.max():.1f}]")
