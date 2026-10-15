"""Run the read-off against real cluster telemetry.

    python experiments/oberon_live/run.py                       # analyse a window
    python experiments/oberon_live/run.py --expect <pod-name>   # score it against a known fault
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "onprem"))
from rca_service.core import Topology, RootCauseEngine   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--topology", default="experiments/oberon_live/topology.json")
ap.add_argument("--counters", default="experiments/oberon_live/counters.npz")
ap.add_argument("--expect", default=None, help="pod name that was actually faulted")
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()

topo = Topology.load(a.topology)
d = np.load(a.counters)
Z = d["z"]
eng = RootCauseEngine(topo)
print(f"topology  {topo.n} pods, {len(topo.edges)} dependency edges, "
      f"stability {topo.stability_margin():.3f}")
print(f"telemetry {Z.shape[0]} steps x {Z.shape[1]} pods x {Z.shape[2]} KPIs, real, from Thanos")
print(f"operator  built in {eng.gain_iters} Riccati iterations, 0 learned parameters\n")

f = eng.locate(Z)
acc = f.residual
order = np.argsort(-acc)

print(f"{'rank':>4}  {'residual':>9}  pod")
for r, i in enumerate(order[:a.top], 1):
    mark = ""
    if a.expect and a.expect in topo.nodes[i]["name"]:
        mark = "   <-- the pod that was actually faulted"
    print(f"{r:>4}  {acc[i]:>9.3f}  {topo.nodes[i]['name']}{mark}")

print(f"\nseparation    robust z over noise floor {f.separation_z:6.1f}")
print(f"              margin over runner-up      {f.margin_over_runner_up:6.3f}")
print(f"              confidence                 {f.confidence:6.3f}")

if a.expect:
    hit1 = a.expect in topo.nodes[order[0]]["name"]
    rank = next((r for r, i in enumerate(order, 1) if a.expect in topo.nodes[i]["name"]), None)
    print(f"\nEXPECTED {a.expect}")
    print(f"  hit@1 {'YES' if hit1 else 'no'}   rank {rank} of {topo.n}")
else:
    print("\nNo fault was injected in this window. A high separation here would be a false "
          "positive, or a real anomaly nobody told us about.")
