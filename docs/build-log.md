# Build log

A full record of what was built, what was measured, what was wrong, and what remains open.
Written so that someone who was not here can audit the work rather than take it on trust.

Dates are the session dates. Every number is reproducible by the command named beside it.

---

## 1. What exists now

| artifact | what it is |
|---|---|
| `src/gnn_healing/` | the research package: twin, simulator, baselines, graph network, solver, intent, orchestration |
| `onprem/` | a deterministic root-cause service: container, HTTP API, OpenShift manifests |
| `docs/math/geometric-network-healing.pdf` | 29-page monograph, the full derivation |
| `docs/flatness/four_lines_one_node.pdf` | 8-page paper, seven experiments |
| `docs/the-method.md` | the method in one page |
| `docs/what-the-demo-shows.md` | the deployed industry system, read off its own UI |
| `docs/industry-comparison.md` | written from published material, partly superseded |
| `docs/state-and-coordination.md` | deployment state, where Redis fits, known gaps |
| `docs/nvidia.md` | the accelerated path |
| `docs/ontology/network-healing.ttl` | the relation vocabulary and its couplings |
| `onprem/ontology/ran-core-transport.ttl` | the same for the operator-shaped topology |
| `deploy/oberon/` | one Kubernetes Job that benchmarks all of it on Intel Xeon 6 |
| `tests/` | 23 tests |

Branch `edge-ai-grid` additionally carries PDU session anchors, Local Area Data Networks,
SSC-mode relocation cost, and per-site inference.

---

## 2. The method

### 2.1 The object

A fault is four numbers: origin node, onset step, severity, ramp length. They are embedded in
`N x K x T` observations. The method is recovering the four from the many.

### 2.2 The operator

```
M = rho*I + sum over relations r of ( beta_r * P_r )      P_r = D_in^-1 A_r^T,  sum_r P_r = P
```

Every per-relation matrix uses the same global in-degree normalisation, so the parts sum to the
whole and setting all couplings equal recovers the single-coupling operator to floating-point
equality. Asserted in `tests/test_ontology.py::test_equal_couplings_collapse_to_the_published_operator`.

Faults attenuate per hop when `max_r beta_r / (1 - rho) < 1`. The on-prem service refuses a
topology that violates this.

### 2.3 The claim

Dynamics `d(t+1) = M d(t) + s(t)` with the source at one node. Rearranged:

```
d(t+1) - M d(t) = s(t),  zero everywhere except the origin
```

Proof is the rearrangement. Two corollaries come free: several simultaneous faults give a support
with several nodes, and the residual at the origin *is* the source, so severity is read rather
than estimated.

### 2.4 Why it is not trivial

The field is never observed. Observations are `x_v = n_v + d_v e_tau(v)`, the field projected
through a per-type effect direction and buried in diurnal and correlated noise. An estimator is
needed before the residual can be taken. Every remaining difficulty traces to that substitution.

### 2.5 The asymmetry with learning

A graph network solving this must learn the **adjoint**: degradation flows cause to effect, but
inference must flow effect back to cause, which is why its architecture carries transposed
relations and why it needs labelled incidents. The read-off needs no adjoint. It applies the
forward operator once and subtracts.

---

## 3. Experiments and results

All on a 74-node synthetic twin with 130 typed directed edges unless stated. Opponent is a
relational graph network, three message-passing layers over nine relations and their transposes,
241,222 parameters, trained 30 epochs on 120 simulated episodes. Both detectors calibrated on the
same nominal episodes with the same margin.

### E1. Head to head

`make pipeline`

| metric | trained network | read-off |
|---|---|---|
| parameters learned | 241,222 | 0 |
| detection rate | 1.00 | 1.00 |
| false alarms, nominal / pre-onset | 0 / 0 | 0 / 0 |
| detection delay after onset | 22.9 steps | 23.8 steps |
| root cause hit@1 / hit@3 | 1.00 / 1.00 | 1.00 / 1.00 |
| top-1 agreement | — | 1.00 |

A tie on every headline metric, at zero parameters.

### E2. Label efficiency — no separation

Retrained at 4, 8, 16, 32, 64 and 120 episodes. hit@1 was 1.00 at every size. **The twin is too
easy for this to discriminate.** Recorded as a null result rather than drawn as a curve.

### E3. Two simultaneous faults — tie

30 episodes, two origins of different types, evaluated after both ramps complete. Both methods
1.00 on both-in-top-2. The read-off handles it by corollary; the network handles it because its
per-node logits are local read-outs. A guarantee against an empirical property.

An earlier version of this experiment evaluated 12 steps after the later onset, before its ramp
completed, and scored both methods near 0.30. That was a statement about the second fault being
too small to see, not about either method.

### E4. New topology, zero-shot — tie

A second twin, 99 nodes, 179 edges, more core and aggregation routers. The network trained on
twin A scored 1.00 zero-shot; so did the read-off with a recomputed operator. A relational
network is inductive by construction, so it transfers across topologies sharing a vocabulary.
Not an advantage for either.

### E5. Identification from unlabelled telemetry

`rho` recovered to within 1%; `beta` underestimated by about 35%. Hypothesis recorded at the time:
one scalar averaging several different couplings.

### E6. Misspecified physics — **the learned model wins**

Test episodes generated with propagation parameters differing from those assumed.

| test physics | network hit@1 | read-off hit@1 | doubt |
|---|---|---|---|
| (0.85, 0.13) matched | 1.00 | 1.00 | 0.12 |
| (0.80, 0.13) | 1.00 | 1.00 | 0.02 |
| (0.72, 0.11) | 0.95 | 0.90 | 0.00 |
| (0.62, 0.09) | **0.93** | **0.67** | 0.00 |

Three findings, all against the method. The learned model degrades more gracefully. **The doubt
statistic failed**: it falls toward zero as drift increases, because faster decay shrinks the
degraded set until it holds little but the origin. The argument that a wrong operator makes the
field visibly non-flat is right; the statistic built to measure it is not. And self-calibration
restores the read-off at moderate drift but makes it worse at severe drift.

### E7. Per-relation coupling — the E5 hypothesis was wrong

`make e7`. Twin propagating with heterogeneous couplings from the ontology, 0.04 to 0.14.

| fit | coupling MAE | notes |
|---|---|---|
| single scalar, noisy projection | 0.0716 | recovers no spread at all |
| per-relation, noisy projection | 0.0523 | correlation 0.797 with truth |
| grouped by property hierarchy | 0.0751 | conditioning 10x better, estimates worse |
| **per-relation, true field, origin excluded** | **0.0044** | `rho` = 0.8500 exactly |

The last row overturns the E5 explanation. On perfectly clean data a single coupling returns
0.0055, not an average of the true values. With `rho` free, least squares prefers driving `rho`
toward 1 and abandoning the coupling over averaging genuinely different ones. The earlier bias
looked mild only because that twin's couplings were uniform.

Two further negative results, recorded because they cost time: tying relations by the ontology's
property hierarchy improves conditioning from 1.5e3 to 1.6e2 and makes estimates **worse**;
temporal smoothing also hurts, because the dynamics live in the step-to-step change smoothing
removes. The gap between 0.0044 and 0.0523 is observation noise at 0.88 of the field's own
standard deviation. The principled repair is expectation-maximisation between the filter and the
regression. Not done.

### E8. On real Intel silicon

`make pipeline` inside `deploy/oberon/run.sh`, on an Intel Xeon 6767P, 256 vCPU, 503 GiB, AMX
present, no GPU, under OpenShift.

**Thread scaling.** The solver is flat; the network collapses.

| threads | solver | network fp32 | network bf16 |
|---|---|---|---|
| 1 | 172.5 µs | 4,170 µs | 4,756 µs |
| 64 | 127.7 µs | 22,510 µs | 1,336,767 µs |
| 256 | 127.1 µs | **14,880,071 µs** | 14,431,912 µs |

At 256 threads the network is **3,568 times slower** than at one. Kubernetes enforces CPU limits
with CFS quota while the container still sees every core, so a default deployment spawns 256
threads for a workload too small to split. The on-prem manifest pins thread counts to 1 and
scales by replica.

**bfloat16 did not help, with AMX physically present.** Ratio 1.14 at one thread, far worse
threaded. AMX accelerates dense matrix multiply; this model is dominated by scatter and gather in
message passing. Autocast pays conversion cost on tensors that never reach a matrix unit. My
prediction before the run was the opposite, and it is recorded as having failed.

**Reproducibility.** The pipeline returned identical published numbers on different silicon.

---

## 4. Defects found and what was done

Listed because the corrections are part of the evidence.

### 4.1 The dense filter is too slow at the scale that matters

The per-step Kalman filter inverts an `N x N` matrix every step, which is cubic.

| nodes | solver | network | crossover |
|---|---|---|---|
| 74 | 156 µs | 2,973 µs | solver 19x faster |
| 642 | 40,622 µs | 7,922 µs | network 5x faster |
| 1,222 | 218,319 µs | 11,894 µs | network 18x faster |
| 3,066 | 2,821,227 µs | 31,371 µs | network 90x faster |

**The crossover sits between 220 and 642 nodes, below the 1,367 of the deployed industry twin.**

Earlier the same day I had written that the scaling objection did not bite at that size, having
checked memory at 7.5 MB and ignored time. That was wrong and is corrected in place.

**Repair, mathematical not operational.** `M`, `Q` and `R` are constant, so the covariance
recursion converges and the gain does not need recomputing. Solving the Riccati recursion once at
startup, 14 iterations, reduces each step to two matrix-vector products:

| nodes | per-step | steady-state | speedup | same origin |
|---|---|---|---|---|
| 642 | 14,533 µs | 48 µs | 304x | 1.00 agreement |
| 1,222 | 62,852 µs | 262 µs | 240x | 1.00 agreement |

Field correlation 0.998. At 1,222 nodes the solver is now 45x faster than the network rather than
18x slower. `FlatnessSolver(..., steady_state=True)`.

### 4.2 The onset estimator was reading the filter's startup transient

The steady-state filter begins from a zero field, so the first steps carry a transient whose
residual reaches 0.325 against a real signal of 0.461. The onset scan walked into it, giving
errors from −47 to +12 steps.

Repaired with a burn-in of three time constants, `3/(1-rho)`, plus a sustained-level requirement.
Residual bias is now **consistently late by a median of 16 steps and never early**, which is a
detection lag of roughly half the ramp. It is therefore labelled a detection time rather than a
start time, in the API payload itself.

### 4.3 The confidence metric was meaningless on a large graph

Originally the share of residual energy at the top node, which divides by every node in the
graph. A correct answer on 781 nodes scored 0.02. Replaced with separation against the noise
floor, a robust z using the median absolute deviation, plus the margin over the runner-up. The
same correct answers now score a median of 0.958 with robust z between 15 and 19.

### 4.4 The README contradicted its own results

The front matter described "a learned inverse of the propagation operator" while the results
table sixty lines below reported 241,222 parameters against 0 with every metric identical.
Rewritten. The network remains, as the opponent it is, including the row where it wins.

### 4.5 The doubt statistic still fails

Unrepaired and reported as such in E6 and in the paper.

---

## 5. Reading the deployed industry system

Two recordings of a booth demo were available, so the comparison was redone from the running
system rather than from slides. Full detail in `docs/what-the-demo-shows.md`.

**Corrected.** The graph model is labelled "GNN Inference, Anomaly Detection" and reports
`Primary origin: unknown` at that stage. Root cause comes from a separate agent executing
**13 tools**: performance counters, alarm correlation, config check, z-scores over an H3
resolution-7 materialised view, a topology walker, and a discovery pipeline against a knowledge
base. The evidence naming the node is a counter, `dl_prb_utilization +7327%`. That is an agentic
query loop, not a learned inverse.

**Corrected.** The deployed twin is 1,367 nodes and 1,915 edges, not operator scale.

**Confirmed.** Detection is a service-layer z-score, stated verbatim on their own Business Value
page, which was the protocol assumption in E1.

**Disclosed by the system itself.** The ranking formula carried "Placeholder values, live GNN
scoring pending," and the live feed read "SIMULATED Automation Engine run, ESOA was not
contacted." Normal for a conference demo. It means 139.9 seconds measured the agent pipeline and
seven API calls rather than a physical remediation.

**The ontology comparison.** Their business vocabulary carries region profiles, revenue density,
a utility function and an activation condition. It has **no coupling term**. Three of four region
profile names match ours, independently chosen, and the mass-event weight of 3 matches exactly.
Theirs indexes the business layer; ours indexes the propagation operator. They are complementary,
and the second does not appear to exist elsewhere.

---

## 6. The on-premises service

`onprem/`. One UBI container, numpy, a topology JSON file. No model weights, no training, no
labels, no GPU, no egress.

Topology in operator shape: HSS, MME, AMF, SMF, UPF, AGG, CSR, eNodeB, gNodeB, sector, H3 hexbin.
781 nodes, 996 edges, 11 relation types.

| | |
|---|---|
| root cause hit@1 | **1.000** over 27 faults across 8 node types |
| inference | 172 µs per step, 52 ms for a 140-step window |
| operator build | 0.63 s once, 14 Riccati iterations |
| separation | robust z 15 to 19 |

Endpoints: `/healthz`, `/readyz`, `GET /v1/topology`, `POST /v1/root-cause`,
`POST /v1/healing-intent`, `POST /v1/topology`. The healing intent carries its own evidence so a
consumer can audit the claim.

Runs under `restricted-v2`: non-root arbitrary UID, no privilege escalation, all capabilities
dropped, read-only root filesystem.

---

## 7. What is still open

| gap | state |
|---|---|
| **no real telemetry** | every number here comes from a simulator written for this repository |
| hit@1 saturates at 1.00 | the twins are too easy to discriminate between methods |
| the linear assumption | real propagation has knees; a nonlinear ground truth has never been tested |
| the doubt statistic | fails, unrepaired |
| identification under noise | limited by projection noise; needs EM between filter and regression |
| dense covariance at six figures | the steady-state gain fixes time; memory still needs a banded form |
| concurrency policy | per-process; needs a distributed lock with an authoritative gate |
| onset | a detection time lagging true onset by about half the ramp |

The first row is the one that matters. Everything else is an engineering note.

---

## 8. How to reproduce any of it

```bash
make setup
make test             # 23 tests, including a numerical equivariance check
make pipeline         # E1
make experiments      # E2 to E6, including the one the method loses
make e7               # per-relation coupling from unlabelled telemetry
make ontology         # the vocabulary and its couplings
make onprem-demo      # the on-premises service, inject and locate
make pdf && make paper
./deploy/oberon/run.sh    # the whole thing on Intel Xeon 6, needs a cluster login
```
