# Hardware benchmark: Intel(R) Xeon(R) 6767P

Run 2026-10-14T23:30:48, 3703.2 s total. AMX present.

## Environment

| | |
|---|---|
| CPU | Intel(R) Xeon(R) 6767P |
| logical cores | 256 |
| AMX present | True |
| accelerator flags | amx_tile, amx_bf16, amx_int8, avx512f, avx512_bf16, avx512_vnni, avx2, sgx |
| torch | 2.14.1+cpu, MKL=True, oneDNN=True |
| Intel Extension for PyTorch | not installed |
| CUDA / XPU | False / False |
| model parameters | 241,222 |

## Thread scaling

| threads | solver us/step | GNN fp32 us | GNN bf16 us |
|---|---|---|---|
| 1 | 172.5 | 4170.0 | 4756.4 |
| 2 | 126.5 | 4661.5 | 8427.9 |
| 4 | 126.1 | 6581.3 | 35163.9 |
| 8 | 174.8 | 7296.5 | 80910.3 |
| 16 | 130.3 | 8593.2 | 212805.0 |
| 32 | 129.7 | 12367.3 | 615092.5 |
| 64 | 127.7 | 22510.0 | 1336767.2 |
| 128 | 128.2 | 187243.9 | 2831296.2 |
| 256 | 127.1 | 14880070.7 | 14431912.3 |

**More threads did not help.** Best GNN latency is 4170.0 us at a single thread; the widest setting measured was 256 threads at 14880070.7 us, which is 3568.4x slower. A 256-core box running this graph with default thread settings is paying synchronisation cost on a workload too small to split. Pin `OMP_NUM_THREADS=1` per replica and scale by running more replicas instead.

Best bfloat16 latency 4756.4 us at 1 threads, ratio to fp32 1.14. bfloat16 is NOT faster here, so autocast is paying conversion cost with no matrix-unit payoff; expected on silicon without AMX.

## Size scaling

At a five-minute telemetry cadence one step has 300,000,000 us of budget. The dense covariance column is the gap recorded in `docs/state-and-coordination.md`: the filter is O(N^3) per step and the covariance is O(N^2) in memory, so this table is where that stops being theoretical.

| nodes | edges | dense covariance | solver us/step | GNN us/step |
|---|---|---|---|---|
| 74 | 130 | 0.0 GB | 156.0 | 2972.5 |
| 220 | 456 | 0.0002 GB | 2446.1 | 4018.4 |
| 642 | 1419 | 0.0016 GB | 40622.2 | 7922.1 |
| 1222 | 2730 | 0.006 GB | 218318.6 | 11894.1 |
| 3066 | 7009 | 0.0376 GB | 2821226.9 | 31371.4 |
| 7390 | 17089 | 0.2184 GB | skipped | 96270.9 |

## Reproducibility across silicon
## Reproducibility across silicon

