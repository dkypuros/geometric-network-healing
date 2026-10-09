"""Render results/report.json into docs/math/chapters/09_experiments_table.tex (run by `make pdf`)."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
r = json.loads((root / "results" / "report.json").read_text())
z = {"gnn": 0, "threshold": 0, "service": 0, "zscore": 0, "service3": 0, "zscore3": 0}
m, fa, pa, dr = r["mean"], {**z, **r.get("false_alarms_on_nominal", {})}, {**z, **r.get("pre_onset_alarms_on_fault_episodes", {})}, {**z, **r.get("detection_rate", {})}
r.setdefault("healed_rate_within_3_cycles", 0.0)
def f(k, d=1): 
    v = m.get(k); return "--" if v is None else f"{v:.{d}f}"
rows = [
  ("Fault episodes / nominal episodes", f"{r['n_fault_episodes']} / {r['n_nominal_episodes']}"),
  ("Detection rate: GNN / static threshold", f"{dr['gnn']:.2f} / {dr['threshold']:.2f}"),
  (f"Detection rate: service $z$ (calibrated $z={r.get('z_service', 0):.1f}$ / textbook $z=3$)", f"{dr['service']:.2f} / {dr['service3']:.2f}"),
  (f"Detection rate: per-node $z$ (calibrated $z={r.get('z_node', 0):.1f}$ / textbook $z=3$)", f"{dr['zscore']:.2f} / {dr['zscore3']:.2f}"),
  ("False alarms on nominal: GNN / thr. / service $z$ cal. / service $z{=}3$ / node $z$ cal. / node $z{=}3$", f"{fa['gnn']} / {fa['threshold']} / {fa['service']} / {fa['service3']} / {fa['zscore']} / {fa['zscore3']}"),
  ("Pre-onset false alarms on fault episodes, same order", f"{pa['gnn']} / {pa['threshold']} / {pa['service']} / {pa['service3']} / {pa['zscore']} / {pa['zscore3']}"),
  ("Detection delay after onset (steps): GNN / static threshold", f"{f('delay_gnn')} / {f('delay_threshold')}"),
  ("Detection delay (steps): service $z$ cal. / $z{=}3$; node $z$ cal. / $z{=}3$", f"{f('delay_service')} / {f('delay_service3')}; {f('delay_zscore')} / {f('delay_zscore3')}"),
  ("Lead time of GNN vs static threshold (steps, where it fired)", f('lead_vs_threshold')),
  ("Lead time of GNN vs service $z$ (calibrated / $z{=}3$)", f"{f('lead_vs_service')} / {f('lead_vs_service3')}"),
  ("Lead time of GNN vs per-node $z$ (calibrated / $z{=}3$)", f"{f('lead_vs_zscore')} / {f('lead_vs_zscore3')}"),
  ("Root cause hit@1 / hit@3", f"{f('hit@1',2)} / {f('hit@3',2)}"),
  ("Healed within 3 verify-and-re-trigger cycles / mean cycles", f"{r['healed_rate_within_3_cycles']:.2f} / {f('healing_cycles',2)}"),
  ("War-room set at alarm time (nodes) / triage reduction@3", f"{f('n_degraded_at_alarm')} / {f('triage_reduction@3',2)}"),
  ("Revenue-weighted exposure: GNN-timed / threshold-timed healing", f"{f('exposure_gnn_loop')} / {f('exposure_threshold_loop')}"),
]
out = ["\\begin{center}\\small\\begin{tabular}{@{}p{9.2cm}r@{}}\\toprule Metric & Value \\\\ \\midrule"]
out += [f"{a} & {b} \\\\" for a, b in rows]
out += ["\\bottomrule\\end{tabular}\\end{center}", "",
        "\\begin{center}\\small\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
        "Fault type & $n$ & delay GNN & delay thr. & delay service $z{=}3$ & hit@1 & hit@3 \\\\ \\midrule"]
for k, v in r["by_fault"].items():
    g = lambda kk, d=1: "--" if v.get(kk) is None else f"{v[kk]:.{d}f}"
    out.append(f"{k.replace('_', ' ')} & {v.get('n', '-')} & {g('delay_gnn')} & {g('delay_threshold')} & {g('delay_service3')} & {g('hit@1',2)} & {g('hit@3',2)} \\\\")
out += ["\\bottomrule\\end{tabular}\\end{center}",
        f"\\noindent\\textit{{Twin: {r['twin']['nodes']} nodes, {r['twin']['edges']} edges. Detection threshold $\\theta={r['theta']:.3f}$. "
        f"Config: episodes={r['config']['episodes']}, epochs={r['config']['epochs']}, T={r['config']['T']}, window={r['config']['window']}, "
        f"seed={r['config']['seed']}. Runtime {r['runtime_s']}\\,s on CPU.}}"]
(root / "docs" / "math" / "chapters" / "09_experiments_table.tex").write_text("\n".join(out) + "\n")
print("wrote 09_experiments_table.tex")

# ---- second output: the flatness paper's table (docs/flatness/results_table.tex)
if "flatness" in r:
    fl = r["flatness"]
    rows2 = [
      ("Fault episodes / nominal episodes", f"{r['n_fault_episodes']} / {r['n_nominal_episodes']}", ""),
      ("Parameters learned", f"{r.get('model_params', 0):,}", "0"),
      ("Detection rate", f"{dr['gnn']:.2f}", f"{fl['detection_rate']:.2f}"),
      ("False alarms on nominal / pre-onset on fault episodes", f"{fa['gnn']} / {pa['gnn']}", f"{fa.get('flatness',0)} / {pa.get('flatness',0)}"),
      ("Detection delay after onset (steps)", f('delay_gnn'), f('delay_flatness')),
      ("Root cause hit@1 / hit@3", f"{f('hit@1',2)} / {f('hit@3',2)}", f"{f('hit@1_flatness',2)} / {f('hit@3_flatness',2)}"),
      ("Top-1 agreement between the two", "", f('flatness_agrees_gnn_top1', 2)),
      ("Static-threshold alarm delay, for scale (steps)", f('delay_threshold'), f('delay_threshold')),
    ]
    out2 = ["\\begin{center}\\small\\begin{tabular}{@{}p{7.2cm}rr@{}}\\toprule Metric & trained GNN & flatness solve \\\\ \\midrule"]
    out2 += [f"{a} & {b} & {c} \\\\" for a, b, c in rows2]
    out2 += ["\\bottomrule\\end{tabular}\\end{center}", "",
             "\\begin{center}\\small\\begin{tabular}{@{}lrrrrr@{}}\\toprule Fault type & $n$ & delay GNN & delay flatness & hit@1 GNN & hit@1 flatness \\\\ \\midrule"]
    for k, v in r["by_fault"].items():
        g = lambda kk, d=1: "--" if v.get(kk) is None else f"{v[kk]:.{d}f}"
        out2.append(f"{k.replace('_', ' ')} & {v.get('n','-')} & {g('delay_gnn')} & {g('delay_flatness')} & {g('hit@1',2)} & {g('hit@1_flatness',2)} \\\\")
    out2 += ["\\bottomrule\\end{tabular}\\end{center}",
             f"\\noindent\\textit{{Twin: {r['twin']['nodes']} nodes, {r['twin']['edges']} edges; {r['config']['episodes']} episodes, seed {r['config']['seed']}; "
             f"GNN trained {r['config']['epochs']} epochs; both detectors calibrated on the same nominal episodes. Runtime {r['runtime_s']}\\,s on CPU.}}"]
    (root / "docs" / "flatness" / "results_table.tex").write_text("\n".join(out2) + "\n")
    print("wrote docs/flatness/results_table.tex")
