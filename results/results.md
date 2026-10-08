# Results

Twin: 93 nodes, 198 edges. Fault episodes: 33, nominal: 7. Runtime 1605.0 s.

| metric | value |
|---|---|
| detection rate, GNN | 1.00 |
| detection rate, static threshold | 0.97 |
| detection rate, service-layer z-score (calibrated z=4.9 / textbook z=3) | 0.03 / 0.58 |
| detection rate, per-node z-score (calibrated z=7.9 / textbook z=3) | 0.00 / 1.00 |
| false alarms on 7 nominal episodes: GNN / threshold / service z cal. / service z=3 / node z cal. / node z=3 | 0 / 0 / 0 / 1 / 0 / 7 |
| pre-onset false alarms on 33 fault episodes: same order | 1 / 0 / 0 / 4 / 0 / 33 |
| mean detection delay after onset (steps): GNN / static threshold | 21.6 / 35.8 |
| mean detection delay after onset (steps): service z cal. / service z=3 / node z cal. / node z=3 | 38.0 / 58.6 / nan / 6.9 |
| mean lead time of GNN vs static threshold (steps, where it fired) | 14.2 |
| mean lead time of GNN vs service z (calibrated / z=3) | 18.0 / 37.2 |
| mean lead time of GNN vs per-node z (calibrated / z=3) | nan / -14.6 |
| healed within 3 verify-and-re-trigger cycles | 1.00 |
| mean healing cycles | 1.00 |
| root cause hit@1 | 1.00 |
| root cause hit@3 | 1.00 |
| degraded nodes at static-alarm time (war-room set, mean) | 15.9 |
| triage reduction @3 (share of the war-room set a top-3 RCA list skips) | 0.33 |
| revenue-weighted exposure, GNN-timed healing | 334.2 |
| revenue-weighted exposure, threshold-timed healing | 815.8 |

## Edge-partitioned inference (AI grid)

| metric | full graph | per-site + regional merge |
|---|---|---|
| detection rate | 1.00 | 0.79 |
| mean detection delay after onset (steps) | 21.6 | 25.3 |
| root cause hit@1 / hit@3 | 1.00 / 1.00 | 0.76 / 0.76 |
| hit@1 on site-local faults / on regional faults (19 regional) | - | 1.00 / 0.58 |
| edge top-1 agrees with full-graph top-1 | - | 0.76 |
| scalars leaving a site per step (KPIs vs uplinked summary) | 52 | 7 |

## By fault type

| fault | n | delay GNN | delay thr | delay service z=3 | hit@1 | hit@3 |
|---|---|---|---|---|---|---|
| amf_signaling_storm | 9 | 25.6 | 50.2 | 58.0 | 1.00 | 1.00 |
| fiber_degrade | 6 | 22.2 | 35.5 | 42.0 | 1.00 | 1.00 |
| ladn_dn_degrade | 1 | 20.0 | 25.0 | 31.0 | 1.00 | 1.00 |
| psa_overload | 3 | 20.3 | 23.3 | 15.0 | 1.00 | 1.00 |
| router_congestion | 5 | 20.4 | 35.0 | 72.0 | 1.00 | 1.00 |
| sleeping_cell | 2 | 15.0 | 18.0 | 92.0 | 1.00 | 1.00 |
| upf_overload | 7 | 19.4 | 32.1 | 66.8 | 1.00 | 1.00 |
