"""Intel Xeon benchmark for the healing stack. Runs anywhere; designed for Oberon.

Answers four questions that the laptop cannot:

  1. ENVIRONMENT   what silicon, how many cores, is AMX present, which BLAS backs the solver
  2. SCALING       how do the flatness solver and the GNN scale with thread count
  3. PRECISION     does bfloat16 autocast help the GNN (AMX accelerates bf16 matmul)
  4. SIZE          how far does the twin scale before the dense covariance becomes the wall
  5. REPRODUCIBLE  same seed on different silicon: do the published numbers come back

Writes results/oberon_bench.json and results/oberon_bench.md.
Usage:  python scripts/oberon_bench.py [--quick]
"""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

from gnn_healing.twin import generate_twin
from gnn_healing.twin.schema import K
from gnn_healing.sim import make_episodes
from gnn_healing.baselines.flatness import FlatnessSolver
from gnn_healing.gnn import HealingGNN

def usable_cores() -> int:
    """Cores this process may actually run on. os.cpu_count() reports the host inside a
    container and would send the thread ladder far past the cgroup limit."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
QUICK = "--quick" in sys.argv
log = lambda *a: print(*a, flush=True)


# ---------------------------------------------------------------- 1. environment
def environment() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "logical_cores": usable_cores(),
        "host_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "env": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_AFFINITY", "KMP_BLOCKTIME")},
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": bool(getattr(torch, "xpu", None) and torch.xpu.is_available()),
    }
    # CPU model and accelerator flags (Linux only)
    flags, model = set(), None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
            if model and flags:
                break
    except Exception:
        pass
    info["cpu_model"] = model or platform.processor() or "unknown"
    watch = ["amx_tile", "amx_bf16", "amx_int8", "amx_fp16",
             "avx512f", "avx512_bf16", "avx512_vnni", "avx2", "sgx", "tdx_guest"]
    info["cpu_flags"] = {f: (f in flags) for f in watch} if flags else "unavailable"
    info["amx"] = any(f.startswith("amx") and flags and f in flags for f in watch)

    # which BLAS is behind torch and numpy
    cfg = torch.__config__.show()
    info["torch_blas"] = next((l.strip() for l in cfg.split("\n") if "BLAS_INFO" in l), "n/a")[:160]
    info["torch_has_mkl"] = bool(getattr(torch.backends, "mkl", None) and torch.backends.mkl.is_available())
    info["torch_has_mkldnn"] = bool(getattr(torch.backends, "mkldnn", None) and torch.backends.mkldnn.is_available())
    try:
        import importlib
        info["intel_extension_for_pytorch"] = importlib.import_module("intel_extension_for_pytorch").__version__
    except Exception:
        info["intel_extension_for_pytorch"] = None
    return info


# ---------------------------------------------------------------- timing helpers
def time_solver(twin, ep, repeats: int = 3) -> float:
    """microseconds per step of the flatness filter"""
    s = FlatnessSolver(twin)
    s.field(ep)                                      # warm
    best = min(_t(lambda: s.field(ep)) for _ in range(repeats))
    return best * 1e6 / ep.T


def time_gnn(model, x, ei, et, nt, iters: int, bf16: bool = False) -> float:
    """microseconds per forward pass"""
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else _null()
    with torch.no_grad(), ctx:
        for _ in range(3):
            model(x, ei, et, nt)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, ei, et, nt)
        return (time.perf_counter() - t0) * 1e6 / iters


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _t(fn) -> float:
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0


# ---------------------------------------------------------------- 2. thread scaling
def thread_scaling(twin, ep, model, base) -> list[dict]:
    cores = usable_cores()
    ladder = [n for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= cores]
    if cores not in ladder:
        ladder.append(cores)
    if QUICK:
        ladder = ladder[:3] + ladder[-1:]
    x = torch.randn(twin.n, 8 * K)
    original = torch.get_num_threads()
    rows = []
    for n in ladder:
        torch.set_num_threads(n)
        iters = 30 if QUICK else 100
        r = {"threads": n,
             "solver_us_per_step": round(time_solver(twin, ep, repeats=2), 1),
             "gnn_fp32_us": round(time_gnn(model, x, base.edge_index, base.edge_type, base.node_type, iters), 1)}
        try:
            r["gnn_bf16_us"] = round(time_gnn(model, x, base.edge_index, base.edge_type, base.node_type, iters, bf16=True), 1)
        except Exception as e:
            r["gnn_bf16_us"] = None; r["bf16_error"] = str(e)[:90]
        rows.append(r)
        log(f"    threads={n:4d}  solver {r['solver_us_per_step']:9.1f} us/step   "
            f"gnn fp32 {r['gnn_fp32_us']:9.1f} us   bf16 {r['gnn_bf16_us']}")
    torch.set_num_threads(original)
    return rows


# ---------------------------------------------------------------- 4. size scaling
def size_scaling() -> list[dict]:
    """Grow the twin until the dense N x N covariance becomes the wall.
    This is the gap recorded in docs/state-and-coordination.md, measured."""
    configs = [(2, 6, 2, 3), (3, 12, 3, 4), (4, 24, 4, 5), (6, 40, 4, 6),
               (8, 80, 4, 8), (10, 160, 4, 10)]
    if QUICK:
        configs = configs[:2]
    SOLVER_NODE_CAP = 4000        # the filter is O(N^3) per step; past this a job would hang
    rows = []
    for n_core, n_agg, gpa, cpg in configs:
        tw = generate_twin(n_core=n_core, n_agg=n_agg, gnb_per_agg=gpa, cells_per_gnb=cpg, seed=0)
        ep = make_episodes(tw, 1, T=40, nominal_fraction=0.0, seed=1)[0]
        cov_gb = tw.n * tw.n * 4 / 1e9
        row = {"nodes": tw.n, "edges": int(len(tw.edges)), "dense_covariance_gb": round(cov_gb, 4)}
        if tw.n > SOLVER_NODE_CAP:
            row["solver_us_per_step"] = None
            row["solver_skipped"] = f"dense O(N^3) filter skipped above {SOLVER_NODE_CAP} nodes"
        else:
            try:
                t0 = time.perf_counter(); FlatnessSolver(tw).field(ep); dt = time.perf_counter() - t0
                row["solver_us_per_step"] = round(dt * 1e6 / ep.T, 1)
            except Exception as e:
                row["solver_us_per_step"] = None; row["solver_error"] = str(e)[:90]
        m = HealingGNN(8).eval()
        d = tw.to_pyg(); x = torch.randn(tw.n, 8 * K)
        row["gnn_us"] = round(time_gnn(m, x, d.edge_index, d.edge_type, d.node_type, 20), 1)
        rows.append(row)
        log(f"    {row['nodes']:5d} nodes  cov {row['dense_covariance_gb']:8.4f} GB  "
            f"solver {row['solver_us_per_step']} us/step  gnn {row['gnn_us']} us")
    return rows


# ---------------------------------------------------------------- 5. reproducibility
def reproducibility() -> dict:
    """Same seed, different silicon. Compares against the committed laptop run."""
    ref_path = ROOT / "results" / "report.json"
    out = {"reference_present": ref_path.exists()}
    if not ref_path.exists():
        return out
    ref = json.loads(ref_path.read_text())
    out["reference"] = {"hit@1": ref["mean"].get("hit@1"), "hit@1_flatness": ref["mean"].get("hit@1_flatness"),
                        "delay_gnn": ref["mean"].get("delay_gnn"), "theta": ref.get("theta"),
                        "twin_nodes": ref["twin"]["nodes"]}
    # deterministic, cheap, silicon-sensitive: regenerate the twin and the operator spectrum
    tw = generate_twin(seed=0)
    s = FlatnessSolver(tw)
    ep = make_episodes(tw, 1, T=160, nominal_fraction=0.0, seed=1)[0]
    d = s.field(ep)
    out["recomputed"] = {
        "twin_nodes": tw.n, "twin_edges": int(len(tw.edges)),
        "operator_spectral_radius": round(float(np.max(np.abs(np.linalg.eigvals(s.M)))), 10),
        "field_max": round(float(d.max()), 8),
        "field_sum": round(float(d.sum()), 6),
        "argmax_node": int(np.unravel_index(np.argmax(d), d.shape)[1]),
        "true_origin": int(ep.origin),
    }
    out["twin_matches_reference"] = (tw.n == out["reference"]["twin_nodes"])
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    R = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "quick": QUICK}

    log("[1/5] environment")
    R["environment"] = environment()
    e = R["environment"]
    log(f"    {e['cpu_model']}")
    log(f"    {e['logical_cores']} logical cores, torch threads {e['torch_num_threads']}, AMX={e['amx']}")

    log("[2/5] building reference twin")
    twin = generate_twin(seed=0)
    base = twin.to_pyg()
    ep = make_episodes(twin, 1, T=160, nominal_fraction=0.0, seed=1)[0]
    model = HealingGNN(8).eval()
    R["model_parameters"] = sum(p.numel() for p in model.parameters())
    log(f"    twin {twin.n} nodes / {len(twin.edges)} edges, model {R['model_parameters']:,} params")

    log("[3/5] thread scaling")
    R["thread_scaling"] = thread_scaling(twin, ep, model, base)

    log("[4/5] size scaling")
    R["size_scaling"] = size_scaling()

    log("[5/5] reproducibility vs committed laptop run")
    R["reproducibility"] = reproducibility()

    R["runtime_s"] = round(time.time() - t_start, 1)
    (OUT / "oberon_bench.json").write_text(json.dumps(R, indent=2, default=str))
    write_markdown(R, OUT / "oberon_bench.md")
    log("\n" + (OUT / "oberon_bench.md").read_text())
    return 0


def write_markdown(R: dict, path: Path) -> None:
    e = R["environment"]
    ts = R["thread_scaling"]
    best = min(ts, key=lambda r: r["gnn_fp32_us"])
    one = next((r for r in ts if r["threads"] == 1), ts[0])
    L = [f"# Hardware benchmark: {e['cpu_model']}", "",
         f"Run {R['started']}, {R['runtime_s']} s total. "
         f"{'AMX present.' if e['amx'] else 'No AMX on this machine, so the bfloat16 row is a baseline, not a result.'}", "",
         "## Environment", "", "| | |", "|---|---|",
         f"| CPU | {e['cpu_model']} |",
         f"| logical cores | {e['logical_cores']} |",
         f"| AMX present | {e['amx']} |"]
    if isinstance(e["cpu_flags"], dict):
        on = [f for f, v in e["cpu_flags"].items() if v]
        L.append(f"| accelerator flags | {', '.join(on) if on else 'none detected'} |")
    L += [f"| torch | {e['torch']}, MKL={e['torch_has_mkl']}, oneDNN={e['torch_has_mkldnn']} |",
          f"| Intel Extension for PyTorch | {e['intel_extension_for_pytorch'] or 'not installed'} |",
          f"| CUDA / XPU | {e['cuda_available']} / {e['xpu_available']} |",
          f"| model parameters | {R['model_parameters']:,} |", "",
          "## Thread scaling", "",
          "| threads | solver us/step | GNN fp32 us | GNN bf16 us |", "|---|---|---|---|"]
    for r in ts:
        L.append(f"| {r['threads']} | {r['solver_us_per_step']} | {r['gnn_fp32_us']} | {r['gnn_bf16_us'] or 'n/a'} |")
    sp = one["gnn_fp32_us"] / best["gnn_fp32_us"] if best["gnn_fp32_us"] else 1
    if best["threads"] == 1:
        L += ["", f"**More threads did not help.** Best GNN latency is {best['gnn_fp32_us']} us at a "
                  f"single thread; the widest setting measured was {ts[-1]['threads']} threads at "
                  f"{ts[-1]['gnn_fp32_us']} us, which is {ts[-1]['gnn_fp32_us']/best['gnn_fp32_us']:.1f}x slower. "
                  f"A {R['environment']['logical_cores']}-core box running this graph with default thread "
                  f"settings is paying synchronisation cost on a workload too small to split. Pin "
                  f"`OMP_NUM_THREADS=1` per replica and scale by running more replicas instead.", ""]
    else:
        L += ["", f"Best GNN latency {best['gnn_fp32_us']} us at {best['threads']} threads, "
                  f"a {sp:.1f}x speedup over single-threaded. Beyond that, synchronisation cost "
                  f"outweighs the parallelism available in a graph this size.", ""]
    bf = [r for r in ts if r.get("gnn_bf16_us")]
    if bf:
        b = min(bf, key=lambda r: r["gnn_bf16_us"])
        ratio = b["gnn_bf16_us"] / b["gnn_fp32_us"]
        verdict = ("bfloat16 is faster, consistent with AMX matrix units doing the work"
                   if ratio < 0.95 else
                   "bfloat16 is NOT faster here, so autocast is paying conversion cost with no "
                   "matrix-unit payoff; expected on silicon without AMX")
        L.append(f"Best bfloat16 latency {b['gnn_bf16_us']} us at {b['threads']} threads, "
                 f"ratio to fp32 {ratio:.2f}. {verdict}.")
    L += ["", "## Size scaling", "",
          "At a five-minute telemetry cadence one step has 300,000,000 us of budget. The dense "
          "covariance column is the gap recorded in `docs/state-and-coordination.md`: the filter is "
          "O(N^3) per step and the covariance is O(N^2) in memory, so this table is where that "
          "stops being theoretical.", "",
          "| nodes | edges | dense covariance | solver us/step | GNN us/step |", "|---|---|---|---|---|"]
    for r in R["size_scaling"]:
        sv = r.get("solver_us_per_step")
        sv = str(sv) if sv is not None else ("skipped" if r.get("solver_skipped") else "FAILED")
        L.append(f"| {r['nodes']} | {r['edges']} | {r['dense_covariance_gb']} GB | {sv} | {r['gnn_us']} |")
    rep = R["reproducibility"]
    L += ["", "## Reproducibility across silicon", ""]
    if rep.get("reference_present"):
        rc = rep["recomputed"]
        L += ["| quantity | value |", "|---|---|",
              f"| twin nodes / edges | {rc['twin_nodes']} / {rc['twin_edges']} |",
              f"| matches committed run | {rep['twin_matches_reference']} |",
              f"| operator spectral radius | {rc['operator_spectral_radius']} |",
              f"| field max | {rc['field_max']} |",
              f"| field sum | {rc['field_sum']} |",
              f"| read-off argmax node / true origin | {rc['argmax_node']} / {rc['true_origin']} |"]
    else:
        L.append("No committed `results/report.json` to compare against; run `make pipeline` first.")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
