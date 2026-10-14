# Hardware benchmark: arm

Run 2026-10-14T18:26:37, 3.6 s total. No AMX on this machine, so the bfloat16 row is a baseline, not a result.

## Environment

| | |
|---|---|
| CPU | arm |
| logical cores | 8 |
| AMX present | False |
| torch | 2.14.1, MKL=False, oneDNN=False |
| Intel Extension for PyTorch | not installed |
| CUDA / XPU | False / False |
| model parameters | 241,222 |

## Thread scaling

| threads | solver us/step | GNN fp32 us | GNN bf16 us |
|---|---|---|---|
| 1 | 75.3 | 3153.6 | 6798.1 |
| 2 | 74.8 | 6182.1 | 10313.3 |
| 4 | 75.8 | 8744.7 | 13595.7 |
| 8 | 88.9 | 15457.7 | 19630.8 |

**More threads did not help.** Best GNN latency is 3153.6 us at a single thread; the widest setting measured was 8 threads at 15457.7 us, which is 4.9x slower. A 8-core box running this graph with default thread settings is paying synchronisation cost on a workload too small to split. Pin `OMP_NUM_THREADS=1` per replica and scale by running more replicas instead.

Best bfloat16 latency 6798.1 us at 1 threads, ratio to fp32 2.16. bfloat16 is NOT faster here, so autocast is paying conversion cost with no matrix-unit payoff; expected on silicon without AMX.

## Size scaling

At a five-minute telemetry cadence one step has 300,000,000 us of budget. The dense covariance column is the gap recorded in `docs/state-and-coordination.md`: the filter is O(N^3) per step and the covariance is O(N^2) in memory, so this table is where that stops being theoretical.

| nodes | edges | dense covariance | solver us/step | GNN us/step |
|---|---|---|---|---|
| 74 | 130 | 0.0 GB | 81.7 | 10197.2 |
| 220 | 456 | 0.0002 GB | 1111.7 | 9508.8 |

## Reproducibility across silicon

| quantity | value |
|---|---|
| twin nodes / edges | 74 / 130 |
| matches committed run | True |
| operator spectral radius | 0.8500000238 |
| field max | 4.23620272 |
| field sum | 1853.262085 |
| read-off argmax node / true origin | 66 / 66 |
