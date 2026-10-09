"""Experiments for docs/flatness: what the flatness read-off can do that a trained GNN cannot.

E2 learning curve     GNN hit@1 vs number of labelled training episodes; read-off needs none
E3 two faults         two simultaneous origins; support has two nodes, softmax assumes one
E4 new topology       different graph, no retraining; GNN zero-shot vs read-off with new M
E5 identification     (rho, beta) identified from unlabelled episodes; read-off with identified M
E6 misspecification   test physics differ from training/assumed physics; hit@1 for both, doubt score,
                      and self-calibration restoring the read-off
E1 worked incident    figure: observed KPIs, estimated field, residual source

Writes results/flatness_experiments.json, docs/flatness/exp_tables.tex, docs/flatness/fig_*.pdf
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gnn_healing.twin import generate_twin
from gnn_healing.sim import make_episodes, simulate, sample_fault, PropagationParams, NominalParams
from gnn_healing.sim.faults import FaultType
from gnn_healing.gnn import train, TrainConfig, calibrate_threshold, detect, rank_root_causes
from gnn_healing.gnn.detect import _scores
from gnn_healing.baselines.flatness import FlatnessSolver, identify_operator
from gnn_healing.evaluation import hit_at_k

ONLY = set(a for a in sys.argv[1:])            # e.g. e3 e5 e6 : rerun only those, reuse cached E2 + model
root = Path(__file__).resolve().parents[1]
out_dir = root / "results"; fig_dir = root / "docs" / "flatness"
T, W, EPOCHS, RCA_DELAY, SEED = 160, 8, 20, 5, 0
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
quiet = lambda *a, **k: None

twinA = generate_twin(seed=SEED); baseA = twinA.to_pyg()
eps = make_episodes(twinA, 160, T=T, seed=SEED)
train_eps, test_eps = eps[:120], eps[120:]
test_f = [e for e in test_eps if e.fault is not None]
calA = make_episodes(twinA, 8, T=T, nominal_fraction=1.0, seed=SEED + 10_000)
log(f"twin A: {twinA.n} nodes; train {len(train_eps)}, test faults {len(test_f)}")


def eval_gnn(model, twin, base, tests, cal):
    theta = calibrate_threshold(model, cal, base, W)
    h1, h3, dl = [], [], []
    for ep in tests:
        tg = detect(model, ep, base, W, theta, start=ep.onset)
        r = rank_root_causes(model, ep, base, W, (tg if tg is not None else ep.onset) + RCA_DELAY)
        h1.append(hit_at_k(r, ep.origin, 1)); h3.append(hit_at_k(r, ep.origin, 3))
        dl.append(None if tg is None else tg - ep.onset)
    d = [x for x in dl if x is not None]
    return {"hit@1": float(np.mean(h1)), "hit@3": float(np.mean(h3)),
            "detection_rate": len(d) / len(tests), "delay": float(np.mean(d)) if d else None}


def eval_flat(solver, tests, cal):
    theta = solver.calibrate(cal)
    h1, h3, dl, db = [], [], [], []
    for ep in tests:
        tf = solver.detect(ep, theta, start=ep.onset)
        t = (tf if tf is not None else ep.onset) + RCA_DELAY
        r = solver.rank(ep, t)
        h1.append(hit_at_k(r, ep.origin, 1)); h3.append(hit_at_k(r, ep.origin, 3))
        dl.append(None if tf is None else tf - ep.onset); db.append(solver.doubt(ep, t + 15, theta))
    d = [x for x in dl if x is not None]
    return {"hit@1": float(np.mean(h1)), "hit@3": float(np.mean(h3)),
            "detection_rate": len(d) / len(tests), "delay": float(np.mean(d)) if d else None,
            "doubt": float(np.mean(db))}


json_path = out_dir / "flatness_experiments.json"
R = json.loads(json_path.read_text()) if (ONLY and json_path.exists()) else {}
R["config"] = {"T": T, "window": W, "epochs": EPOCHS, "rca_delay": RCA_DELAY, "seed": SEED, "n_test_faults": len(test_f)}
flatA = FlatnessSolver(twinA)
model_path = out_dir / "flatness_gnn_full.pt"
want = lambda e: (not ONLY) or (e in ONLY)

# ---------------- E2: learning curve
if want("e2"):
    log("E2 learning curve")
    curve, models = [], {}
    for n in [4, 8, 16, 32, 64, 120]:
        torch.manual_seed(SEED)
        m = train(train_eps[:n], baseA, TrainConfig(epochs=EPOCHS, window=W, seed=SEED), log=quiet)
        models[n] = m
        g = eval_gnn(m, twinA, baseA, test_f, calA)
        n_labels = sum(1 for e in train_eps[:n] if e.fault is not None)
        curve.append({"train_episodes": n, "labelled_fault_episodes": n_labels, **g})
        log(f"  n={n:3d} labelled={n_labels:3d} hit@1={g['hit@1']:.2f} delay={g['delay']}")
    gnn_full = models[120]
    torch.save(gnn_full.state_dict(), model_path)
else:
    from gnn_healing.gnn import HealingGNN
    if model_path.exists():
        gnn_full = HealingGNN(W); gnn_full.load_state_dict(torch.load(model_path)); gnn_full.eval()
    else:
        log("training the full GNN once (E2 cached)")
        torch.manual_seed(SEED)
        gnn_full = train(train_eps, baseA, TrainConfig(epochs=EPOCHS, window=W, seed=SEED), log=quiet)
        torch.save(gnn_full.state_dict(), model_path)
    curve = R["E2"]["gnn"]
flat_full = eval_flat(flatA, test_f, calA)
R["E2"] = {"gnn": curve, "flatness": flat_full}
log(f"  flatness hit@1={flat_full['hit@1']:.2f} delay={flat_full['delay']} doubt={flat_full['doubt']:.2f}")

# ---------------- E3: two simultaneous faults
if want("e3"):
  log("E3 two faults")
  rng = np.random.default_rng(123)
  both_g, any_g, both_f, any_f = [], [], [], []
  for i in range(30):
      f1 = sample_fault(twinA.nodes, T, rng)
      f2 = sample_fault(twinA.nodes, T, rng)
      while f2.origin == f1.origin or f2.kind == f1.kind:
          f2 = sample_fault(twinA.nodes, T, rng)
      ep = simulate(twinA, T, [f1, f2], np.random.default_rng(500 + i))
      t = min(max(f1.onset + f1.ramp, f2.onset + f2.ramp) + 5, T - 1)      # after both ramps complete
      _, rca = _scores(gnn_full, ep, baseA, W)
      rg = np.argsort(-rca[t])[:2].tolist()
      rf = flatA.rank(ep, t)[:2]
      origins = {f1.origin, f2.origin}
      both_g.append(float(set(rg) == origins)); any_g.append(float(len(set(rg) & origins) > 0))
      both_f.append(float(set(rf) == origins)); any_f.append(float(len(set(rf) & origins) > 0))
  R["E3"] = {"n": 30, "gnn": {"both_in_top2": float(np.mean(both_g)), "any_in_top2": float(np.mean(any_g))},
             "flatness": {"both_in_top2": float(np.mean(both_f)), "any_in_top2": float(np.mean(any_f))}}
  log(f"  both-in-top2  GNN {np.mean(both_g):.2f}  flatness {np.mean(both_f):.2f}")

# ---------------- E4: new topology, no retraining
if want("e4"):
  log("E4 new topology")
  twinB = generate_twin(n_core=3, n_agg=8, gnb_per_agg=2, cells_per_gnb=3, n_upf=3, n_services=8, seed=7)
  baseB = twinB.to_pyg()
  testB = [e for e in make_episodes(twinB, 50, T=T, seed=7) if e.fault is not None]
  calB = make_episodes(twinB, 8, T=T, nominal_fraction=1.0, seed=7 + 10_000)
  gB = eval_gnn(gnn_full, twinB, baseB, testB, calB)
  fB = eval_flat(FlatnessSolver(twinB), testB, calB)
  R["E4"] = {"twin_B": {"nodes": twinB.n, "edges": int(len(twinB.edges))}, "n": len(testB), "gnn_zero_shot": gB, "flatness": fB}
  log(f"  twin B {twinB.n} nodes: GNN zero-shot hit@1={gB['hit@1']:.2f}  flatness hit@1={fB['hit@1']:.2f}")

# ---------------- E5: identify the operator from unlabelled episodes
if want("e5"):
  log("E5 identification")
  rho_hat, beta_hat = identify_operator(twinA, train_eps)
  fid = eval_flat(FlatnessSolver(twinA, rho=rho_hat, beta=beta_hat), test_f, calA)
  R["E5"] = {"true": {"rho": PropagationParams().rho, "beta": PropagationParams().beta},
             "identified": {"rho": rho_hat, "beta": beta_hat}, "episodes_used": len(train_eps), "labels_used": 0,
             "flatness_with_identified": fid}
  log(f"  rho {rho_hat:.3f} (true {PropagationParams().rho})  beta {beta_hat:.3f} (true {PropagationParams().beta})  hit@1={fid['hit@1']:.2f}")

# ---------------- E6: misspecified physics at test time
if want("e6"):
  log("E6 misspecification")
  rows = []
  for i, (rho_s, beta_s) in enumerate([(0.85, 0.13), (0.80, 0.13), (0.72, 0.11), (0.62, 0.09)]):
      prop = PropagationParams(rho=rho_s, beta=beta_s)
      e_mis = make_episodes(twinA, 50, T=T, seed=900 + i, prop=prop)
      tf_mis = [e for e in e_mis if e.fault is not None]
      cal_mis = make_episodes(twinA, 8, T=T, nominal_fraction=1.0, seed=900 + i + 10_000, prop=prop)
      g = eval_gnn(gnn_full, twinA, baseA, tf_mis, cal_mis)                  # trained at (0.85, 0.13)
      f_assumed = eval_flat(flatA, tf_mis, cal_mis)                           # assumes (0.85, 0.13)
      rh, bh = identify_operator(twinA, e_mis)                                # self-calibrate, no labels
      f_ident = eval_flat(FlatnessSolver(twinA, rho=rh, beta=bh), tf_mis, cal_mis)
      rows.append({"rho_sim": rho_s, "beta_sim": beta_s, "attenuation_per_hop": beta_s / (1 - rho_s), "n": len(tf_mis),
                   "gnn": g, "flatness_assumed": f_assumed, "identified": {"rho": rh, "beta": bh}, "flatness_identified": f_ident})
      log(f"  sim ({rho_s},{beta_s}): GNN hit@1={g['hit@1']:.2f}  flat(assumed) hit@1={f_assumed['hit@1']:.2f} doubt={f_assumed['doubt']:.2f}"
          f"  flat(identified rho={rh:.2f},beta={bh:.3f}) hit@1={f_ident['hit@1']:.2f} doubt={f_ident['doubt']:.2f}")
  R["E6"] = rows

# ---------------- E1: worked incident figure
log("E1 figure")
ep = next(e for e in test_f if e.fault.kind == FaultType.FIBER_DEGRADE) if any(e.fault.kind == FaultType.FIBER_DEGRADE for e in test_f) else test_f[0]
d_hat = flatA.field(ep); s_hat = flatA.source(d_hat)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
im0 = ax[0].imshow(ep.X[:, :, 1].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3); ax[0].set_title("observed KPI (latency), all nodes")
im1 = ax[1].imshow(d_hat.T, aspect="auto", cmap="magma"); ax[1].set_title(r"estimated field $\hat d$ (Kalman, known $M$)")
im2 = ax[2].imshow(s_hat.T, aspect="auto", cmap="magma"); ax[2].set_title(r"residual source $\hat s=\hat d(t)-M\hat d(t-1)$")
for a in ax:
    a.set_xlabel("time step"); a.axvline(ep.onset, color="w", lw=0.8, ls="--")
    a.axhline(ep.origin, color="cyan", lw=0.8, ls=":")
ax[0].set_ylabel("node"); ax[2].text(ep.T * 0.55, ep.origin - 2, f"origin node {ep.origin} ({ep.fault.kind.value})", color="cyan", fontsize=8)
for a, im in zip(ax, (im0, im1, im2)): fig.colorbar(im, ax=a, shrink=0.8)
fig.savefig(fig_dir / "fig_incident.pdf"); plt.close(fig)
R["E1"] = {"episode_fault": ep.fault.kind.value, "origin": int(ep.origin), "onset": int(ep.onset),
           "moduli_read_off": flatA.moduli(ep, ep.onset + 30), "true": {"severity_per_step": ep.fault.severity, "ramp": ep.fault.ramp}}

fig, ax = plt.subplots(figsize=(5.2, 3.3), constrained_layout=True)
xs = [c["labelled_fault_episodes"] for c in curve]
ax.plot(xs, [c["hit@1"] for c in curve], "o-", label="trained GNN (hit@1)")
ax.axhline(flat_full["hit@1"], color="C3", ls="--", label="flatness read-off (0 labels)")
ax.set_xscale("log"); ax.set_xlabel("labelled fault episodes available for training"); ax.set_ylabel("root cause hit@1 on held-out faults")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.savefig(fig_dir / "fig_learning_curve.pdf"); plt.close(fig)

R["runtime_s"] = round(time.time() - t0, 1)
json_path.write_text(json.dumps(R, indent=2, default=float))
curve = R["E2"]["gnn"]; flat_full = R["E2"]["flatness"]; rows = R["E6"]

# ---------------- LaTeX tables
def fmt(x, d=2): return "--" if x is None else f"{x:.{d}f}"
L = []
L += [r"\begin{center}\small\begin{tabular}{@{}rrrrr@{}}\toprule training episodes & labelled faults & GNN hit@1 & GNN hit@3 & GNN delay \\ \midrule"]
for c in curve: L.append(f"{c['train_episodes']} & {c['labelled_fault_episodes']} & {fmt(c['hit@1'])} & {fmt(c['hit@3'])} & {fmt(c['delay'],1)} \\\\")
L += [r"\midrule", f"flatness read-off & 0 & {fmt(flat_full['hit@1'])} & {fmt(flat_full['hit@3'])} & {fmt(flat_full['delay'],1)} \\\\", r"\bottomrule\end{tabular}\end{center}"]
(fig_dir / "tab_e2.tex").write_text("\n".join(L) + "\n")
e3 = R["E3"]
(fig_dir / "tab_e3.tex").write_text(
 r"\begin{center}\small\begin{tabular}{@{}lrr@{}}\toprule two simultaneous faults, 30 episodes & trained GNN & flatness read-off \\ \midrule" "\n"
 f"both origins in the top-2 & {fmt(e3['gnn']['both_in_top2'])} & {fmt(e3['flatness']['both_in_top2'])} \\\\\n"
 f"at least one origin in the top-2 & {fmt(e3['gnn']['any_in_top2'])} & {fmt(e3['flatness']['any_in_top2'])} \\\\\n"
 r"\bottomrule\end{tabular}\end{center}" "\n")
e4 = R["E4"]
(fig_dir / "tab_e4.tex").write_text(
 r"\begin{center}\small\begin{tabular}{@{}lrr@{}}\toprule " f"new topology ({e4['twin_B']['nodes']} nodes, {e4['twin_B']['edges']} edges), {e4['n']} faults & GNN zero-shot & flatness, new $M$ \\\\ \\midrule" "\n"
 f"detection rate & {fmt(e4['gnn_zero_shot']['detection_rate'])} & {fmt(e4['flatness']['detection_rate'])} \\\\\n"
 f"detection delay (steps) & {fmt(e4['gnn_zero_shot']['delay'],1)} & {fmt(e4['flatness']['delay'],1)} \\\\\n"
 f"root cause hit@1 / hit@3 & {fmt(e4['gnn_zero_shot']['hit@1'])} / {fmt(e4['gnn_zero_shot']['hit@3'])} & {fmt(e4['flatness']['hit@1'])} / {fmt(e4['flatness']['hit@3'])} \\\\\n"
 r"\bottomrule\end{tabular}\end{center}" "\n")
e5 = R["E5"]
(fig_dir / "tab_e5.tex").write_text(
 r"\begin{center}\small\begin{tabular}{@{}lrrr@{}}\toprule & true & identified (0 labels) & relative error \\ \midrule" "\n"
 f"$\\rho$ & {e5['true']['rho']:.3f} & {e5['identified']['rho']:.3f} & {abs(e5['identified']['rho']-e5['true']['rho'])/e5['true']['rho']*100:.1f}\\% \\\\\n"
 f"$\\beta$ & {e5['true']['beta']:.3f} & {e5['identified']['beta']:.3f} & {abs(e5['identified']['beta']-e5['true']['beta'])/e5['true']['beta']*100:.1f}\\% \\\\\n"
 r"\midrule" "\n"
 f"read-off with identified operator: hit@1 / hit@3 / delay & \\multicolumn{{3}}{{r}}{{{fmt(e5['flatness_with_identified']['hit@1'])} / {fmt(e5['flatness_with_identified']['hit@3'])} / {fmt(e5['flatness_with_identified']['delay'],1)}}} \\\\\n"
 r"\bottomrule\end{tabular}\end{center}" "\n")
L = [r"\begin{center}\footnotesize\begin{tabular}{@{}lrrrrrr@{}}\toprule test $(\rho,\beta)$ & per-hop & GNN & read-off, assumed & doubt & identified $(\hat\rho,\hat\beta)$ & read-off, identified \\ \midrule"]
for r_ in rows:
    L.append(f"({r_['rho_sim']:.2f}, {r_['beta_sim']:.2f}) & {r_['attenuation_per_hop']:.2f} & {fmt(r_['gnn']['hit@1'])} & {fmt(r_['flatness_assumed']['hit@1'])} & {fmt(r_['flatness_assumed']['doubt'])} & ({r_['identified']['rho']:.2f}, {r_['identified']['beta']:.3f}) & {fmt(r_['flatness_identified']['hit@1'])} \\\\")
L += [r"\bottomrule\end{tabular}\end{center}", r"\noindent{\footnotesize GNN and read-off columns are hit@1. Doubt is \eqref{eq:doubt} for the assumed operator.}\par\medskip"]
(fig_dir / "tab_e6.tex").write_text("\n".join(L) + "\n")
mo = R["E1"]["moduli_read_off"]
(fig_dir / "tab_e1.tex").write_text(
 r"\begin{center}\small\begin{tabular}{@{}lrr@{}}\toprule " f"incident ({R['E1']['episode_fault'].replace('_',' ')}) & true & read off the residual \\\\ \\midrule" "\n"
 f"origin node & {R['E1']['origin']} & {mo['origin']} \\\\\n"
 f"onset step & {R['E1']['onset']} & {mo['onset']} \\\\\n"
 f"severity (source per step at plateau) & {R['E1']['true']['severity_per_step']:.2f} & {mo['severity']:.2f} \\\\\n"
 r"\bottomrule\end{tabular}\end{center}" "\n")
log("done")
