"""End-to-end pipeline:  twin -> simulate -> baselines -> GNN -> detect -> RCA -> intent -> closed loop.

    gnn-healing pipeline --episodes 120 --epochs 30 --out results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .twin import generate_twin
from .sim import make_episodes, PropagationParams, NominalParams
from .baselines import (threshold_alarm_detection, zscore_detection, calibrate_zscore, service_baseline,
                        calibrate_service_z, service_zscore_trigger, degraded_set)
from .gnn import train, TrainConfig, calibrate_threshold, detect, rank_root_causes, describe_backend
from .gnn.detect import degradation_estimate
from .intent import propose_branches, select_order, to_tmf921_intent
from .orchestration import decompose_to_service_orders, MockOrchestrator, closed_loop, healing_cycles
from .evaluation import lead_time, hit_at_k, triage_reduction, summarize


def run_pipeline(a) -> dict:
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("backend:", json.dumps(describe_backend()))
    twin = generate_twin(seed=a.seed)
    base = twin.to_pyg()
    print(f"twin: {twin.n} nodes, {len(twin.edges)} directed edges, "
          f"{int(base.edge_index.shape[1])} message-passing edges incl. transposes")
    print(twin.nodes["type"].value_counts().to_dict())

    eps = make_episodes(twin, a.episodes, T=a.T, seed=a.seed)
    n_train = int(0.6 * len(eps)); n_cal = int(0.15 * len(eps))
    train_eps, cal_eps, test_eps = eps[:n_train], eps[n_train:n_train + n_cal], eps[n_train + n_cal:]
    cal_nominal = [e for e in cal_eps if e.fault is None] or [e for e in train_eps if e.fault is None][:4]
    print(f"episodes: train {len(train_eps)}  calibration {len(cal_eps)} (nominal {len(cal_nominal)})  test {len(test_eps)}")

    cfg = TrainConfig(epochs=a.epochs, window=a.window, seed=a.seed)
    model = train(train_eps, base, cfg)
    theta = calibrate_threshold(model, cal_nominal, base, cfg.window)
    service_idx = twin.nodes.index[twin.nodes["type"] == "service"].to_numpy()
    mu_s, sd_s = service_baseline(cal_nominal, service_idx)
    z_service = calibrate_service_z(cal_nominal, service_idx, mu_s, sd_s)
    z_node = calibrate_zscore(cal_nominal)
    print(f"calibrated on {len(cal_nominal)} nominal episodes: GNN theta = {theta:.3f}, "
          f"service z = {z_service:.2f}, per-node z = {z_node:.2f} (same margin, zero false alarms on calibration)")

    rows, false_alarms, nominal_count = [], {"threshold": 0, "zscore": 0, "service": 0, "gnn": 0}, 0
    pre_onset_alarms = {"threshold": 0, "zscore": 0, "service": 0, "gnn": 0}
    orch = MockOrchestrator(exec_delay=a.exec_delay)
    sample_intent = None
    for i, ep in enumerate(test_eps):
        t_thr = threshold_alarm_detection(ep)
        t_z = zscore_detection(ep, z=z_node)
        t_s = service_zscore_trigger(ep, service_idx, mu_s, sd_s, z_service)
        t_g = detect(model, ep, base, cfg.window, theta)
        if ep.fault is None:
            nominal_count += 1
            for k, t in (("threshold", t_thr), ("zscore", t_z), ("service", t_s), ("gnn", t_g)):
                false_alarms[k] += int(t is not None)
            continue
        # a firing before onset is a false alarm; detection delay is measured from the first firing at/after onset
        for k, t in (("threshold", t_thr), ("zscore", t_z), ("service", t_s), ("gnn", t_g)):
            pre_onset_alarms[k] += int(t is not None and t < ep.onset)
        t_thr = threshold_alarm_detection(ep, start=ep.onset)
        t_z = zscore_detection(ep, z=z_node, start=ep.onset)
        t_s = service_zscore_trigger(ep, service_idx, mu_s, sd_s, z_service, start=ep.onset)
        t_g = detect(model, ep, base, cfg.window, theta, start=ep.onset)
        t_rca = (t_g if t_g is not None else ep.onset) + a.rca_delay
        ranking = rank_root_causes(model, ep, base, cfg.window, t_rca)
        d_hat = degradation_estimate(model, ep, base, cfg.window, t_rca)
        t_warroom = min((t_thr if t_thr is not None else ep.T - 1) + a.rca_delay, ep.T - 1)
        n_deg = len(degraded_set(ep, t_warroom))        # nodes a war room would face at alarm time
        # business layer + closed loop
        branches = propose_branches(twin, ranking, d_hat, k=3)
        chosen = select_order(branches, budget=a.budget)
        intent = to_tmf921_intent(chosen, intent_id=f"healing-{i}")
        orders = decompose_to_service_orders(intent)
        eff = orch.execute(orders, t_rca)
        gnn_loop = healing_cycles(twin, ep.fault, ep.T, ranking[:3], eff, cycle_delay=a.rca_delay + a.exec_delay, seed=1000 + i)
        thr_loop = closed_loop(twin, ep.fault, ep.T, None if t_thr is None else t_thr + a.rca_delay + a.exec_delay, seed=1000 + i)
        if sample_intent is None:
            sample_intent = {"intent": intent, "service_orders": orders, "branches": [b.__dict__ for b in branches]}
        rows.append({
            "fault": ep.fault.kind.value, "origin_type": twin.nodes.loc[ep.origin, "type"],
            "onset": ep.onset, "t_threshold": t_thr, "t_zscore": t_z, "t_service": t_s, "t_gnn": t_g,
            "detected_gnn": t_g is not None, "detected_threshold": t_thr is not None,
            "delay_gnn": (t_g - ep.onset) if t_g is not None else None,
            "delay_threshold": (t_thr - ep.onset) if t_thr is not None else None,
            "delay_service": (t_s - ep.onset) if t_s is not None else None,
            "lead_vs_service": lead_time(t_s, t_g, ep.T),
            "lead_vs_threshold": lead_time(t_thr, t_g, ep.T),
            "lead_vs_zscore": lead_time(t_z, t_g, ep.T),
            "hit@1": hit_at_k(ranking, ep.origin, 1), "hit@3": hit_at_k(ranking, ep.origin, 3),
            "n_degraded_at_alarm": n_deg, "triage_reduction@3": triage_reduction(3, n_deg),
            "healing_cycles": gnn_loop["cycles"], "healed": gnn_loop["healed"],
            "exposure_gnn_loop": gnn_loop["revenue_weighted_exposure"],
            "exposure_threshold_loop": thr_loop["revenue_weighted_exposure"],
            "top_branch_domain": chosen[0].domain if chosen else None,
        })

    summ = summarize(rows)
    n_f = len(rows)
    report = {
        "config": {k: v for k, v in vars(a).items() if k != "fn"}, "twin": {"nodes": twin.n, "edges": int(len(twin.edges))}, "theta": theta,
        "n_fault_episodes": n_f, "n_nominal_episodes": nominal_count,
        "false_alarms_on_nominal": false_alarms, "pre_onset_alarms_on_fault_episodes": pre_onset_alarms,
        "detection_rate": {"gnn": float(np.mean([r["detected_gnn"] for r in rows])),
                           "threshold": float(np.mean([r["detected_threshold"] for r in rows])),
                           "service": float(np.mean([r["t_service"] is not None for r in rows]))},
        "healed_rate_within_3_cycles": float(np.mean([r["healed"] for r in rows])),
        "mean": summ, "by_fault": {}, "sample_intent": sample_intent, "orchestrator_log_head": orch.log[:3],
        "runtime_s": round(time.time() - t_start, 1),
    }
    for kind in sorted({r["fault"] for r in rows}):
        report["by_fault"][kind] = summarize([r for r in rows if r["fault"] == kind])
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "episodes.json").write_text(json.dumps(rows, indent=1, default=float))
    torch.save(model.state_dict(), out / "model.pt")
    _write_markdown(report, out / "results.md")
    print(open(out / "results.md").read())
    print("sample intent:", json.dumps(sample_intent["intent"], indent=1) if sample_intent else None)
    return report


def _write_markdown(r: dict, path: Path) -> None:
    m = r["mean"]
    L = ["# Results", "",
         f"Twin: {r['twin']['nodes']} nodes, {r['twin']['edges']} edges. "
         f"Fault episodes: {r['n_fault_episodes']}, nominal: {r['n_nominal_episodes']}. Runtime {r['runtime_s']} s.", "",
         "| metric | value |", "|---|---|",
         f"| detection rate, GNN | {r['detection_rate']['gnn']:.2f} |",
         f"| detection rate, static threshold | {r['detection_rate']['threshold']:.2f} |",
         f"| detection rate, service-layer z-score trigger | {r['detection_rate']['service']:.2f} |",
         f"| false alarms on nominal episodes (GNN / threshold / service z / per-node z) | {r['false_alarms_on_nominal']['gnn']} / {r['false_alarms_on_nominal']['threshold']} / {r['false_alarms_on_nominal']['service']} / {r['false_alarms_on_nominal']['zscore']} of {r['n_nominal_episodes']} |",
         f"| pre-onset false alarms on fault episodes (GNN / threshold / service z / per-node z) | {r['pre_onset_alarms_on_fault_episodes']['gnn']} / {r['pre_onset_alarms_on_fault_episodes']['threshold']} / {r['pre_onset_alarms_on_fault_episodes']['service']} / {r['pre_onset_alarms_on_fault_episodes']['zscore']} of {r['n_fault_episodes']} |",
         f"| mean detection delay after onset, GNN (steps) | {m.get('delay_gnn', float('nan')):.1f} |",
         f"| mean detection delay after onset, static threshold (steps) | {m.get('delay_threshold', float('nan')):.1f} |",
         f"| mean detection delay after onset, service-layer z-score (steps) | {m.get('delay_service', float('nan')):.1f} |",
         f"| mean lead time, GNN vs static threshold (steps, episodes where threshold fired) | {m.get('lead_vs_threshold', float('nan')):.1f} |",
         f"| mean lead time, GNN vs service-layer z-score (steps, episodes where it fired) | {m.get('lead_vs_service', float('nan')):.1f} |",
         f"| mean lead time, GNN vs per-node z-score (steps, episodes where it fired) | {m.get('lead_vs_zscore', float('nan')):.1f} |",
         f"| healed within 3 verify-and-re-trigger cycles | {r['healed_rate_within_3_cycles']:.2f} |",
         f"| mean healing cycles | {m.get('healing_cycles', 0):.2f} |",
         f"| root cause hit@1 | {m.get('hit@1', 0):.2f} |",
         f"| root cause hit@3 | {m.get('hit@3', 0):.2f} |",
         f"| degraded nodes at static-alarm time (war-room set, mean) | {m.get('n_degraded_at_alarm', 0):.1f} |",
         f"| triage reduction @3 (share of the war-room set a top-3 RCA list skips) | {m.get('triage_reduction@3', 0):.2f} |",
         f"| revenue-weighted exposure, GNN-timed healing | {m.get('exposure_gnn_loop', 0):.1f} |",
         f"| revenue-weighted exposure, threshold-timed healing | {m.get('exposure_threshold_loop', 0):.1f} |",
         "", "## By fault type", "", "| fault | n | delay GNN | delay thr | hit@1 | hit@3 |", "|---|---|---|---|---|---|"]
    for k, v in r["by_fault"].items():
        L.append(f"| {k} | - | {v.get('delay_gnn', float('nan')):.1f} | {v.get('delay_threshold', float('nan')):.1f} | {v.get('hit@1', 0):.2f} | {v.get('hit@3', 0):.2f} |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gnn-healing")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pipeline", help="run the full closed-loop experiment")
    q.add_argument("--episodes", type=int, default=120)
    q.add_argument("--epochs", type=int, default=30)
    q.add_argument("--T", type=int, default=160)
    q.add_argument("--window", type=int, default=8)
    q.add_argument("--rca-delay", type=int, default=5)
    q.add_argument("--exec-delay", type=int, default=3)
    q.add_argument("--budget", type=int, default=6)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", default="results")
    q.set_defaults(fn=run_pipeline)
    t = sub.add_parser("twin", help="print the synthetic twin summary")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=lambda a: print(generate_twin(seed=a.seed).nodes.groupby(["domain", "type"]).size()))
    b = sub.add_parser("backend", help="show detected hardware / libraries")
    b.set_defaults(fn=lambda a: print(json.dumps(describe_backend(), indent=2)))
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    main()
