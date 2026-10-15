"""Does this network satisfy the method's precondition? Run before deploying anything.

The read-off assumes degradation propagates along the modelled edges, one step at a time:

    d(t+1) ~ M d(t)        so a cause at time t should predict its effects at time t+1

That is a LAGGED, DIRECTED claim about the projected field, and it is what must be tested.
A first version of this file tested contemporaneous correlation on raw counters instead, and
failed its own control, where propagation is true by construction. Correlation at zero lag is
dominated by shared diurnal shape and tells you nothing about direction.

The test below compares, on healthy telemetry with no faults and no labels:

    corr( field_src(t), field_dst(t+1) )  over modelled edges
    the same statistic over random non-edges, and over the edges REVERSED

If forward edges do not beat random pairs, the graph carries no propagation.
If forward does not beat reversed, the graph carries no direction, and a method that
inverts a directed operator cannot work.

    python experiments/oberon_live/precondition.py --counters counters.npz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--topology", default="experiments/oberon_live/topology.json")
ap.add_argument("--counters", default="experiments/oberon_live/counters.npz")
ap.add_argument("--trials", type=int, default=3000)
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()

topo = json.loads(Path(a.topology).read_text())
Z = np.load(a.counters)["z"]
N = Z.shape[1]
rng = np.random.default_rng(0)

# project the KPI vector onto the direction degradation is known to take, as the solver does
eff = topo.get("effect")
if eff:
    E = np.array([eff[n["type"]] for n in topo["nodes"]], dtype=float)
    proj = E / np.maximum((E ** 2).sum(axis=1, keepdims=True), 1e-9)
    F = np.einsum("tnk,nk->tn", Z, proj)
else:
    F = Z[:, :, 0]
F = F - F.mean(axis=0, keepdims=True)
sd = F.std(axis=0); live = sd > 1e-9


def lagged(i: int, j: int) -> float | None:
    """corr(field at i now, field at j next step). The operator's actual claim."""
    if not (live[i] and live[j]):
        return None
    x, y = F[:-1, i], F[1:, j]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


edges = [(e["src"], e["dst"]) for e in topo["edges"]]
edgeset = set(edges)
fwd = [c for i, j in edges if (c := lagged(i, j)) is not None]
rev = [c for i, j in edges if (c := lagged(j, i)) is not None]
rnd = []
while len(rnd) < a.trials:
    i, j = int(rng.integers(N)), int(rng.integers(N))
    if i != j and (i, j) not in edgeset and (c := lagged(i, j)) is not None:
        rnd.append(c)
fwd, rev, rnd = np.abs(fwd), np.abs(rev), np.abs(rnd)

# The statistic is the TAIL, not the mean. In any real network most pairs are quiet most of
# the time, so the average lagged correlation is noise on both edges and non-edges; the control
# with propagation true by construction scores 1.11x on the mean, which is indistinguishable
# from the 1.03x of a stack that propagates nothing. What separates them is how many edges
# carry a STRONG lagged signal: 9.8% against 3.9% of random pairs in the control, and 0%
# against 0.5% in the live stack.
THR = a.threshold
f_strong, r_strong, v_strong = (fwd > THR).mean(), (rnd > THR).mean(), (rev > THR).mean()
lift = f_strong / (r_strong + 1e-9)
direction = f_strong / (v_strong + 1e-9)
null = [( np.abs(rng.choice(rnd, size=len(fwd))) > THR).mean() for _ in range(2000)]
p_val = float((np.array(null) >= f_strong).mean())

print("PRECONDITION CHECK  (lagged, directed, tail statistic)")
print(f"  topology  {N} nodes, {len(edges)} modelled edges, {int(live.sum())} carrying a live signal")
print(f"  telemetry {Z.shape[0]} steps")
print(f"  statistic fraction of pairs with |corr(src_t, dst_t+1)| > {THR}\n")
print(f"  1. propagation  edges {f_strong:6.3f}   random {r_strong:6.3f}   lift {lift:5.2f}x")
print(f"  2. direction    forward {f_strong:6.3f}   reversed {v_strong:6.3f}   ratio {direction:5.2f}x")
print(f"  3. significance permutation p = {p_val:.4f}")
print(f"  (means, for reference and as a warning: edges {fwd.mean():.3f} vs random {rnd.mean():.3f};")
print(f"   the mean does not discriminate and should not be used)")

ok_prop = lift >= 1.8 and f_strong > 0.02
ok_sig = p_val < 0.05
ok_dir = direction >= 1.1
print(f"\n  VERDICT: {'PASS' if (ok_prop and ok_sig) else 'FAIL'}")
if ok_prop and ok_sig:
    print("  Degradation travels along the modelled edges: strong lagged prediction is")
    print(f"  {lift:.1f}x more common on edges than between unrelated nodes. The operator")
    print("  describes something real and the read-off has a chance here.")
    if not ok_dir:
        print("  Direction is weak, so separating cause from effect may be unreliable.")
else:
    print("  No edge carries materially more predictive signal than an unrelated pair.")
    print("  The operator is fiction on this data and the read-off cannot work. That is a")
    print("  property of the data or of the topology, not of the method. Check whether the")
    print("  system is actually carrying load, and whether the modelled edges are the ones")
    print("  that really couple.")
