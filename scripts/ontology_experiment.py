"""E7: does per-relation coupling fix the identification defect?

Experiment five of the paper identified the operator from unlabelled telemetry and recovered
rho to within 1% but underestimated beta by about 35%. The hypothesis recorded at the time was
that one scalar was being asked to average several genuinely different couplings. The ontology
makes that testable, because it declares a beta_r per relation.

Setup. The twin now propagates with the ontology's heterogeneous couplings, which span 0.04 on
the control plane to 0.14 on transport, a 3.5x spread. Then four operators are compared, all
scored on the same held-out faults:

    assumed       the published single beta = 0.13, no identification at all
    identified    one scalar identified from unlabelled episodes
    per-relation  one beta_r per relation identified from the SAME unlabelled episodes
    oracle        the true ontology values, an upper bound

Writes results/ontology_experiment.json and docs/flatness/tab_e7.tex.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np

from gnn_healing import ontology
from gnn_healing.twin import generate_twin
from gnn_healing.twin.schema import RELATIONS
from gnn_healing.sim import make_episodes, PropagationParams
from gnn_healing.baselines.flatness import (FlatnessSolver, identify_operator,
                                            identify_operator_per_relation, identify_operator_grouped)
from gnn_healing.evaluation import hit_at_k

ROOT = Path(__file__).resolve().parents[1]
T, RCA_DELAY, SEED = 160, 5, 0
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def evaluate(solver: FlatnessSolver, tests, cal) -> dict:
    theta = solver.calibrate(cal)
    h1, h3, delay = [], [], []
    for ep in tests:
        tf = solver.detect(ep, theta, start=ep.onset)
        t = (tf if tf is not None else ep.onset) + RCA_DELAY
        r = solver.rank(ep, t)
        h1.append(hit_at_k(r, ep.origin, 1)); h3.append(hit_at_k(r, ep.origin, 3))
        delay.append(None if tf is None else tf - ep.onset)
    d = [x for x in delay if x is not None]
    return {"hit@1": float(np.mean(h1)), "hit@3": float(np.mean(h3)),
            "detection_rate": len(d) / len(tests),
            "delay": float(np.mean(d)) if d else None}


def main() -> int:
    truth = ontology.coupling_by_relation()
    rho_true = PropagationParams().rho
    log(f"ontology couplings: {min(truth.values()):.3f} to {max(truth.values()):.3f}, "
        f"{max(truth.values())/min(truth.values()):.1f}x spread; "
        f"stability {ontology.stability_margin(rho_true):.3f}")

    # the twin now propagates heterogeneously, which is the whole point
    prop = PropagationParams(from_ontology=True)
    twin = generate_twin(seed=SEED)
    log("simulating with per-relation propagation")
    eps = make_episodes(twin, 160, T=T, seed=SEED, prop=prop)
    train, test = eps[:120], eps[120:]
    test_f = [e for e in test if e.fault is not None]
    cal = make_episodes(twin, 8, T=T, nominal_fraction=1.0, seed=SEED + 10_000, prop=prop)
    log(f"{len(train)} train episodes (labels discarded), {len(test_f)} held-out faults")

    log("identifying, single scalar")
    rho_s, beta_s = identify_operator(twin, train)
    log(f"    rho {rho_s:.3f}  beta {beta_s:.4f}")

    log("identifying, per relation")
    rho_r, beta_r = identify_operator_per_relation(twin, train)
    log(f"    rho {rho_r:.3f}")
    for r in RELATIONS:
        log(f"      {r.value:<28} true {truth[r.value]:.3f}  identified {beta_r[r.value]:.4f}")

    # Upper bound: the SAME per-relation regression on the true latent field with the origin
    # excluded. This separates "is the model right" from "is the field estimate good enough".
    log("identifying, per relation, on the true latent field (upper bound, uses labels)")
    import numpy as _np
    from gnn_healing.operator import design_matrix as _dm
    Xs, ys = [], []
    for e in train:
        Xf, yf = _dm(twin, e.D, per_relation=True), e.D[1:].ravel()
        if e.fault is not None:
            m = _np.ones((e.D.shape[0] - 1, twin.n), bool); m[:, e.origin] = False
            Xf, yf = Xf[m.ravel()], yf[m.ravel()]
        Xs.append(Xf); ys.append(yf)
    th = _np.linalg.lstsq(_np.concatenate(Xs), _np.concatenate(ys), rcond=None)[0]
    rho_c = float(th[0]); beta_c = {r.value: float(max(th[1 + i], 0.0)) for i, r in enumerate(RELATIONS)}
    log(f"    rho {rho_c:.4f} (true {rho_true})")

    log("identifying, tied by the ontology's property hierarchy")
    rho_g, beta_g, diag_g = identify_operator_grouped(twin, train)
    log(f"    rho {rho_g:.3f}  groups {{" + ", ".join(f"{k}:{v:.4f}" for k, v in diag_g['by_group'].items()) + "}")
    cond_r = getattr(identify_operator_per_relation, "last_condition_number", float("nan"))
    log(f"    condition number: 9 free coefficients {cond_r:.3g} vs 3 grouped {diag_g['condition_number']:.3g}")

    # how well was the coupling vector recovered, in each case
    tv = np.array([truth[r.value] for r in RELATIONS])
    pv = np.array([beta_r[r.value] for r in RELATIONS])
    gv = np.array([beta_g[r.value] for r in RELATIONS])
    sv = np.full_like(tv, beta_s)

    def rec(v, extra=None):
        d = {"mae": float(np.mean(np.abs(v - tv))), "max_abs_err": float(np.max(np.abs(v - tv))),
             "spread_recovered": float(v.max() - v.min())}
        if v.std() > 0:
            d["corr_with_truth"] = float(np.corrcoef(v, tv)[0, 1])
        if extra:
            d.update(extra)
        return d

    cv = np.array([beta_c[r.value] for r in RELATIONS])
    recovery = {"single": rec(sv), "per_relation": rec(pv, {"condition_number": cond_r}),
                "per_relation_clean_field": rec(cv),
                "grouped": rec(gv, {"condition_number": diag_g["condition_number"],
                                    "by_group": diag_g["by_group"]}),
                "true_spread": float(tv.max() - tv.min())}
    log(f"coupling recovery MAE: single {recovery['single']['mae']:.4f}  "
        f"per-relation {recovery['per_relation']['mae']:.4f}  "
        f"grouped {recovery['grouped']['mae']:.4f}  "
        f"per-relation on clean field {recovery['per_relation_clean_field']['mae']:.4f}")

    log("scoring four operators on the same held-out faults")
    variants = {
        "assumed_single_0.13": FlatnessSolver(twin, rho=rho_true, beta=0.13),
        "identified_single":   FlatnessSolver(twin, rho=rho_s, beta=beta_s),
        "identified_per_rel":  FlatnessSolver(twin, rho=rho_r, beta_by_relation=beta_r),
        "identified_grouped":  FlatnessSolver(twin, rho=rho_g, beta_by_relation=beta_g),
        "per_rel_clean_field": FlatnessSolver(twin, rho=rho_c, beta_by_relation=beta_c),
        "oracle_ontology":     FlatnessSolver(twin, rho=rho_true, beta_by_relation=truth),
    }
    scores = {}
    for name, solver in variants.items():
        scores[name] = evaluate(solver, test_f, cal)
        s = scores[name]
        log(f"    {name:<22} hit@1 {s['hit@1']:.2f}  hit@3 {s['hit@3']:.2f}  "
            f"detect {s['detection_rate']:.2f}  delay {s['delay']}")

    R = {"ontology_truth": truth, "rho_true": rho_true,
         "identified": {"single": {"rho": rho_s, "beta": beta_s},
                        "per_relation": {"rho": rho_r, "beta_r": beta_r},
                        "grouped": {"rho": rho_g, "beta_r": beta_g, **diag_g},
                        "per_relation_clean_field": {"rho": rho_c, "beta_r": beta_c}},
         "coupling_recovery": recovery, "scores": scores,
         "n_test_faults": len(test_f), "n_train_episodes": len(train),
         "labels_used": 0, "runtime_s": round(time.time() - t0, 1)}
    (ROOT / "results" / "ontology_experiment.json").write_text(json.dumps(R, indent=2, default=float))
    write_tex(R)
    log("wrote results/ontology_experiment.json and docs/flatness/tab_e7.tex")
    return 0


def write_tex(R: dict) -> None:
    truth, beta_r = R["ontology_truth"], R["identified"]["per_relation"]["beta_r"]
    beta_s, rec, sc = R["identified"]["single"]["beta"], R["coupling_recovery"], R["scores"]
    L = [r"\begin{center}\small\begin{tabular}{@{}lrrrrr@{}}\toprule",
         r"relation & true $\beta_r$ & single & per-relation & grouped & clean-field \\ \midrule"]
    beta_g = R["identified"]["grouped"]["beta_r"]
    beta_c = R["identified"]["per_relation_clean_field"]["beta_r"]
    for r in RELATIONS:
        L.append(f"\\texttt{{{r.value.replace('_', ' ').replace('>', '$>$')}}} & "
                 f"{truth[r.value]:.3f} & {beta_s:.3f} & {beta_r[r.value]:.3f} & {beta_g[r.value]:.3f} & {beta_c[r.value]:.3f} \\\\")
    L += [r"\midrule",
          f"mean absolute error & & {rec['single']['mae']:.4f} & {rec['per_relation']['mae']:.4f} & {rec['grouped']['mae']:.4f} & {rec['per_relation_clean_field']['mae']:.4f} \\\\",
          f"spread recovered (true {rec['true_spread']:.3f}) & & 0.000 & {rec['per_relation']['spread_recovered']:.3f} & {rec['grouped']['spread_recovered']:.3f} & {rec['per_relation_clean_field']['spread_recovered']:.3f} \\\\",
          f"free coefficients & & 1 & {len(RELATIONS)} & {len(rec['grouped']['by_group'])} & {len(RELATIONS)} \\\\",
          f"condition number & & -- & {rec['per_relation']['condition_number']:.3g} & {rec['grouped']['condition_number']:.3g} & -- \\\\",
          r"\bottomrule\end{tabular}\end{center}", "",
          r"\begin{center}\small\begin{tabular}{@{}lrrr@{}}\toprule",
          r"operator & hit@1 & hit@3 & detection delay \\ \midrule"]
    label = {"assumed_single_0.13": r"assumed single $\beta=0.13$, no identification",
             "identified_single": r"identified single $\beta$, 0 labels",
             "identified_per_rel": r"identified $\beta_r$ per relation, 0 labels",
             "identified_grouped": r"identified $\beta_r$ tied by the ontology hierarchy, 0 labels",
             "per_rel_clean_field": r"\emph{upper bound}: $\beta_r$ fitted on the true field",
             "oracle_ontology": r"oracle: the ontology's true $\beta_r$"}
    for k, v in sc.items():
        dly = "--" if v["delay"] is None else f"{v['delay']:.1f}"
        L.append(f"{label[k]} & {v['hit@1']:.2f} & {v['hit@3']:.2f} & {dly} \\\\")
    L += [r"\bottomrule\end{tabular}\end{center}",
          f"\\noindent\\textit{{Twin propagating with the ontology's heterogeneous couplings; "
          f"{R['n_test_faults']} held-out faults; identification used {R['n_train_episodes']} "
          f"episodes with their labels discarded.}}"]
    (ROOT / "docs" / "flatness" / "tab_e7.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
