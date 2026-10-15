"""Build a propagation topology from the 5G stack actually running on the cluster.

Nodes are the running pods. Edges are the 3GPP service-based and O-RAN dependencies between
the network functions they implement, oriented the way a fault propagates, cause to effect.
Nothing here is invented for the experiment: NRF is service discovery so every function depends
on it, AUSF cannot authenticate without UDM, SMF controls UPF over N4, O-DU feeds O-CU, and so on.

    python experiments/oberon_live/topology_from_cluster.py --out experiments/oberon_live/topology.json
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

NAMESPACES = ["telco-ran", "telco-core", "telco-smo", "telco-bss-oss"]

# (cause function, effect function, relation, coupling)
# Couplings follow the same logic as the synthetic ontology: a function that carries the user
# plane or the transport couples strongly; one that only governs signalling couples weakly.
DEPS = [
    # --- O-RAN user plane, strongest: each stage physically carries the one above it
    ("oru",    "odu",    "fronthaul_feeds",      0.14),
    ("odu",    "ocuup",  "f1u_feeds",            0.13),
    ("odu",    "ocucp",  "f1c_feeds",            0.11),
    ("ocuup",  "upf",    "n3_feeds",             0.12),
    ("ocucp",  "amf",    "n2_signals",           0.08),
    ("nearrt-ric", "odu",   "e2_controls",       0.04),
    ("nearrt-ric", "ocucp", "e2_controls",       0.04),
    # --- 5G core user plane
    ("smf",    "upf",    "n4_controls",          0.12),
    ("upf",    "telco-vessel", "n6_carries",     0.10),
    # --- 5G core control plane, weaker: degrades sessions without removing capacity
    ("nrf",    "amf",    "discovery_serves",     0.05),
    ("nrf",    "smf",    "discovery_serves",     0.05),
    ("nrf",    "ausf",   "discovery_serves",     0.05),
    ("nrf",    "udm",    "discovery_serves",     0.05),
    ("nrf",    "pcf",    "discovery_serves",     0.05),
    ("nrf",    "nssf",   "discovery_serves",     0.05),
    ("scp",    "amf",    "proxies",              0.05),
    ("scp",    "smf",    "proxies",              0.05),
    ("udr",    "udm",    "repository_serves",    0.07),
    ("udr",    "pcf",    "repository_serves",    0.07),
    ("udm",    "ausf",   "subscriber_serves",    0.07),
    ("udm",    "amf",    "subscriber_serves",    0.06),
    ("ausf",   "amf",    "authenticates",        0.06),
    ("amf",    "smf",    "session_triggers",     0.07),
    ("pcf",    "smf",    "policy_governs",       0.05),
    ("chf",    "smf",    "charging_governs",     0.04),
    ("nssf",   "amf",    "slice_selects",        0.04),
    ("udsf",   "udr",    "state_serves",         0.05),
    ("eir",    "amf",    "equipment_checks",     0.03),
    ("bsf",    "pcf",    "binding_serves",       0.04),
    # --- IMS
    ("ims-hss","icscf",  "subscriber_serves",    0.06),
    ("icscf",  "scscf",  "routes",               0.08),
    ("scscf",  "pcscf",  "routes",               0.08),
    ("mrf",    "scscf",  "media_serves",         0.05),
    # --- exposure and analytics
    ("nef",    "capif",  "exposes",              0.04),
    ("nwdaf",  "pcf",    "analytics_informs",    0.03),
    ("gmlc",   "lmf",    "location_requests",    0.04),
    ("sepp",   "amf",    "roaming_serves",       0.03),
    ("n3iwf",  "amf",    "untrusted_access",     0.04),
    ("tsctsf", "pcf",    "tsc_governs",          0.03),
    ("nssaaf", "nssf",   "slice_auth",           0.03),
    # --- SMO / O-Cloud over the RAN
    ("ocloud", "odu",    "hosts",                0.06),
    ("ocloud", "ocucp",  "hosts",                0.06),
    ("ptp-bridge", "odu","timing_feeds",         0.09),
    ("smo",    "nearrt-ric", "manages",          0.03),
    ("o2ims-shim", "ocloud", "manages",          0.03),
    ("r1",     "nearrt-ric", "r1_serves",        0.03),
    # --- BSS/OSS on top of the service
    ("oss-inventory",   "frontend-gateway", "inventory_serves", 0.05),
    ("event-store",     "oss-inventory",    "events_feed",      0.05),
    ("frontend-gateway","bss-storefront",   "gateway_serves",   0.07),
    ("bss-billing",     "bss-storefront",   "billing_serves",   0.05),
    ("evidence",        "oss-console-frontend", "evidence_feeds", 0.04),
    ("bss-storefront",  "bss-storefront-frontend", "serves",    0.08),
]

# KPIs pulled from Thanos, in this order. Effect vectors say how a degradation shows up.
KPI_NAMES = ["cpu_rate", "memory_working_set", "net_rx_rate", "net_tx_rate"]
# user-plane carriers show it in throughput and cpu; control-plane in cpu and memory
EFFECT_USER_PLANE   = [0.9, 0.5, -0.8, -0.8]
EFFECT_CONTROL      = [0.9, 0.7, -0.3, -0.3]
EFFECT_TRANSPORT    = [0.7, 0.4, -0.9, -0.9]
USER_PLANE = {"upf", "ocuup", "odu", "oru", "telco-vessel"}
TRANSPORT  = {"ptp-bridge", "ocloud", "n3iwf"}


def pods() -> list[dict]:
    out = []
    for ns in NAMESPACES:
        j = subprocess.run(["kubectl", "-n", ns, "get", "pods", "-o", "json"],
                           capture_output=True, text=True, check=True).stdout
        for p in json.loads(j)["items"]:
            if p.get("status", {}).get("phase") != "Running":
                continue
            name = p["metadata"]["name"]
            # strip the replicaset and pod hashes to get the function name
            fn = "-".join(name.split("-")[:-2]) if name.count("-") >= 2 else name
            out.append({"namespace": ns, "pod": name, "function": fn})
    return out


def build(ps: list[dict]) -> dict:
    by_fn: dict[str, list[int]] = {}
    nodes = []
    for p in ps:
        i = len(nodes)
        fn = p["function"]
        kind = "user_plane" if fn in USER_PLANE else "transport" if fn in TRANSPORT else "control"
        nodes.append({"id": i, "name": f"{p['namespace']}/{p['pod']}", "type": kind,
                      "domain": p["namespace"], "function": fn, "pod": p["pod"],
                      "namespace": p["namespace"]})
        by_fn.setdefault(fn, []).append(i)

    edges, coupling = [], {}
    for src_fn, dst_fn, rel, beta in DEPS:
        for s in by_fn.get(src_fn, []):
            for d in by_fn.get(dst_fn, []):
                edges.append({"src": s, "dst": d, "relation": rel})
                coupling[rel] = beta
    return {
        "rho": 0.85, "kpi_names": KPI_NAMES, "coupling": coupling,
        "effect": {"user_plane": EFFECT_USER_PLANE, "control": EFFECT_CONTROL,
                   "transport": EFFECT_TRANSPORT},
        "nodes": nodes, "edges": edges,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/oberon_live/topology.json")
    a = ap.parse_args()
    ps = pods()
    t = build(ps)
    Path(a.out).write_text(json.dumps(t, indent=1))
    orphans = [n["function"] for n in t["nodes"]
               if not any(e["src"] == n["id"] or e["dst"] == n["id"] for e in t["edges"])]
    print(f"{len(t['nodes'])} running pods, {len(t['edges'])} dependency edges, "
          f"{len(t['coupling'])} relation types -> {a.out}")
    print(f"  max(beta)/(1-rho) = {max(t['coupling'].values())/(1-t['rho']):.3f}")
    if orphans:
        print(f"  {len(orphans)} pods with no modelled dependency: {sorted(set(orphans))}")
