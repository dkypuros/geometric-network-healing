"""Thirty-second demo: inject a transport fault, find it, explain it, emit the intent.

    python onprem/demo.py                      # against the local engine
    python onprem/demo.py --url http://host:8080   # against a running service
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rca_service.core import Topology, RootCauseEngine, healing_intent   # noqa: E402
from rca_service.simulate import inject                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--topology", default=str(Path(__file__).parent / "topology" / "ran_core_transport.json"))
ap.add_argument("--kind", default="agg", help="node type to fault: agg, csr, enodeb, upf, mme ...")
ap.add_argument("--url", default=None)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

topo = Topology.load(a.topology)
print(f"topology  {topo.n} nodes, {len(topo.edges)} edges, "
      f"{len({e['relation'] for e in topo.edges})} relation types")
print(f"stability max(beta_r)/(1-rho) = {topo.stability_margin():.3f}  (must be < 1)\n")

t0 = time.perf_counter()
eng = RootCauseEngine(topo)
print(f"operator built in {time.perf_counter()-t0:.2f}s, "
      f"{eng.gain_iters} Riccati iterations, 0 learned parameters\n")

v = [i for i, n in enumerate(topo.nodes) if n["type"] == a.kind][2]
X, injn = inject(topo, eng.M, steps=140, origin=v, severity=0.5, ramp=40, onset=45, seed=a.seed)
print(f"injected  {topo.name(v)} ({a.kind}) at step {injn.onset}, ramp {injn.ramp}")
print(f"          the operator never sees this; it only sees {X.shape[0]} steps "
      f"of {X.shape[1]} x {X.shape[2]} counters\n")

if a.url:
    import urllib.request
    req = urllib.request.Request(f"{a.url}/v1/root-cause", method="POST",
                                 data=json.dumps({"counters": X.tolist()}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); out = json.loads(urllib.request.urlopen(req, timeout=300).read())
    print(f"service responded in {time.perf_counter()-t0:.2f}s (includes JSON transport)")
    found, conf = out["origin"]["name"], out["confidence"]
    top = [(r["node"], r["value"]) for r in out["explanation"]["top_residuals"]]
    blast = out["blast_radius"]["count"]
else:
    t0 = time.perf_counter(); f = eng.locate(X); el = time.perf_counter() - t0
    print(f"located in {el*1e3:.1f} ms total, {el*1e6/X.shape[0]:.0f} us per step")
    found, conf = f.origin_name, f.confidence
    top = [(topo.name(i), x) for i, x in f.ranked[:5]]
    blast = len(f.blast_radius)

print(f"\nFOUND     {found}   {'CORRECT' if found == topo.name(v) else 'WRONG, injected ' + topo.name(v)}")
print(f"          confidence {conf:.3f}, blast radius {blast} nodes\n")
print("the explanation is the residual, one number per node:")
for name, val in top:
    print(f"   {val:8.4f}  {name}" + ("   <- the only node not obeying the dynamics" if name == found else ""))
print(f"\n{len(topo.nodes)-1} other nodes are flat under the operator. That is the whole argument.")
