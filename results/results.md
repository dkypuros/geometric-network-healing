# Results

Twin: 74 nodes, 120 edges. Fault episodes: 23, nominal: 7. Runtime 384.4 s.

| metric | value |
|---|---|
| detection rate, GNN | 1.00 |
| detection rate, static threshold | 1.00 |
| false alarms on nominal episodes (GNN / threshold / z-score) | 0 / 0 / 7 of 7 |
| pre-onset false alarms on fault episodes (GNN / threshold / z-score) | 0 / 0 / 22 of 23 |
| mean detection delay after onset, GNN (steps) | 21.6 |
| mean detection delay after onset, threshold (steps) | 32.2 |
| mean lead time vs static threshold (steps) | 10.6 |
| mean lead time vs per-node z-score (steps) | -11.0 |
| root cause hit@1 | 1.00 |
| root cause hit@3 | 1.00 |
| triage reduction @3 (share of degraded nodes not inspected) | 0.00 |
| revenue-weighted exposure, GNN-timed healing | 85.5 |
| revenue-weighted exposure, threshold-timed healing | 173.2 |

## By fault type

| fault | n | delay GNN | delay thr | hit@1 | hit@3 |
|---|---|---|---|---|---|
| amf_signaling_storm | - | 33.0 | 66.5 | 1.00 | 1.00 |
| fiber_degrade | - | 23.4 | 29.1 | 1.00 | 1.00 |
| router_congestion | - | 18.7 | 31.3 | 1.00 | 1.00 |
| sleeping_cell | - | 17.4 | 26.3 | 1.00 | 1.00 |
| upf_overload | - | 22.0 | 31.2 | 1.00 | 1.00 |
