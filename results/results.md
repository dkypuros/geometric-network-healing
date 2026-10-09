# Results

Twin: 74 nodes, 130 edges. Fault episodes: 32, nominal: 8. Runtime 647.1 s.

| metric | value |
|---|---|
| detection rate, GNN | 1.00 |
| detection rate, static threshold | 1.00 |
| detection rate, service-layer z-score (calibrated z=4.1 / textbook z=3) | 0.00 / 0.31 |
| detection rate, per-node z-score (calibrated z=8.0 / textbook z=3) | 0.00 / 1.00 |
| false alarms on 8 nominal episodes: GNN / threshold / service z cal. / service z=3 / node z cal. / node z=3 | 0 / 0 / 0 / 4 / 0 / 8 |
| pre-onset false alarms on 32 fault episodes: same order | 0 / 0 / 0 / 3 / 0 / 30 |
| mean detection delay after onset (steps): GNN / static threshold | 22.9 / 32.6 |
| mean detection delay after onset (steps): service z cal. / service z=3 / node z cal. / node z=3 | nan / 56.2 / nan / 9.6 |
| mean lead time of GNN vs static threshold (steps, where it fired) | 9.8 |
| mean lead time of GNN vs service z (calibrated / z=3) | nan / 32.0 |
| mean lead time of GNN vs per-node z (calibrated / z=3) | nan / -13.3 |
| healed within 3 verify-and-re-trigger cycles | 1.00 |
| mean healing cycles | 1.00 |
| root cause hit@1 | 1.00 |
| root cause hit@3 | 1.00 |
| degraded nodes at static-alarm time (war-room set, mean) | 13.9 |
| triage reduction @3 (share of the war-room set a top-3 RCA list skips) | 0.35 |
| revenue-weighted exposure, GNN-timed healing | 328.3 |
| revenue-weighted exposure, threshold-timed healing | 573.1 |

## Flatness solve vs trained GNN (same episodes, same calibration, zero training)

| metric | trained GNN | flatness solve |
|---|---|---|
| detection rate | 1.00 | 1.00 |
| false alarms on nominal / pre-onset on fault episodes | 0 / 0 | 0 / 0 |
| mean detection delay after onset (steps) | 22.9 | 23.8 |
| mean lead time of flatness over GNN (steps, where GNN fired) | - | -1.0 |
| root cause hit@1 / hit@3 | 1.00 / 1.00 | 1.00 / 1.00 |
| top-1 agreement between the two | - | 1.00 |
| parameters learned | 241,222 | 0 |

## By fault type

| fault | n | delay GNN | delay flatness | delay thr | hit@1 GNN | hit@1 flatness |
|---|---|---|---|---|---|---|
| amf_signaling_storm | 7 | 27.9 | 31.7 | 45.0 | 1.00 | 1.00 |
| fiber_degrade | 7 | 22.7 | 23.0 | 28.6 | 1.00 | 1.00 |
| router_congestion | 7 | 22.3 | 23.6 | 32.3 | 1.00 | 1.00 |
| sleeping_cell | 8 | 18.5 | 17.1 | 25.4 | 1.00 | 1.00 |
| upf_overload | 3 | 24.7 | 26.0 | 33.3 | 1.00 | 1.00 |
