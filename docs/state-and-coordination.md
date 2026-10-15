# State and coordination

**Where Redis belongs in a deployment of this system, and where it does not.**

This repository is a research artifact: one process, one command, everything in memory. That is
deliberate and it should stay that way. This note is about what changes when the same method runs
as a real service across many sites, which is the shape described in
[`docs/edge-ai-grid.md`](https://github.com/dkypuros/geometric-network-healing/blob/edge-ai-grid/docs/edge-ai-grid.md)
on the `edge-ai-grid` branch.

The whole strategy in one line: **Redis holds the state that must survive a process restart, and
the state that must be shared between concurrent healers. Nothing else.**

---

## First, sort the state

Most arguments about "should we use Redis" are really arguments about which kind of state someone
is thinking of. There are four kinds here and only one of them is a Redis problem.

| kind | example in this repo | belongs in |
|---|---|---|
| cold, structural | the twin topology, the relation vocabulary, model weights | a graph store, git, a model registry |
| cold, historical | months of KPI history for training or identification | object store or warehouse |
| **hot, local, recursive** | the Kalman filter's current field estimate | process memory, **checkpointed to Redis** |
| **hot, shared** | which domain is currently mid-remediation | **Redis** |

The top two rows are the twin and the archive. See [`docs/nvidia.md`](nvidia.md) for the storage
and sampling path at scale, and
[`docs/industry-comparison.md`](industry-comparison.md) for how that compares with a managed
temporal graph database. Neither is Redis. The Redis graph module was end-of-lifed in January 2025.

---

## Job 1: keep the filter alive across restarts

**Why.** The flatness read-off is *recursive*. Look at
[`FlatnessSolver.field()`](../src/gnn_healing/baselines/flatness.py#L65): the estimate at step `t`
is computed from the estimate at `t-1`. In the repo that loop runs inside one function over a
complete episode. In production it is a long-lived streaming process, and a restart throws the
state away.

That matters more here than in most systems, because the metric this project exists to improve is
*detection lead time*. A restarted site is blind until the filter re-warms. At a five-minute
telemetry cadence, losing ten steps of warm-up is nearly an hour with no detector, which is the
same order as the silent-degradation window we are trying to compress.

**What to store.**

| | |
|---|---|
| key | `site:{s}:filter` |
| value | the field estimate `d` and its covariance, serialised |
| cadence | every step |
| TTL | longer than any plausible restart, e.g. one hour |

At five minutes per step the write is free, so checkpoint every step and recovery is exact. Do not
build a clever every-N-steps scheme; it buys nothing and costs you warm-up on recovery.

**Sizing caveat.** The covariance in
[`FlatnessSolver.field()`](../src/gnn_healing/baselines/flatness.py#L65) is currently dense and
`N × N`, which does not survive contact with a real topology:

| nodes | field vector | dense covariance |
|---|---|---|
| 74 (this repo's twin) | 300 B | 22 KB |
| 100,000 (regional) | 400 KB | 40 GB |

**Partly resolved.** The time cost, which was always the sharper half of this problem, is fixed
by `steady_state=True`: the system is time-invariant, so the gain is solved once and each step
becomes two matrix-vector products, 240x faster at 1,222 nodes. What a checkpoint must carry then
is the field and the precomputed gain, and the gain is static, so only the field is hot state.

A regional deployment at six figures of nodes still needs a banded, low-rank, or information-form
covariance regardless of whether Redis is involved. Once it has one, the per-site checkpoint is kilobytes. **The
checkpointing question and the scaling question are the same question, and neither is answered in
this repo today.**

The detector's calibrated threshold, from
[`FlatnessSolver.calibrate()`](../src/gnn_healing/baselines/flatness.py#L88) and
[`calibrate_threshold()`](../src/gnn_healing/gnn/detect.py#L33), is also per-site state worth
keeping alongside it. It is recomputed from healthy history and should not be recomputed on every
pod start.

---

## Job 2: serialise risky changes across sites

**This is the strongest case for Redis in the whole system, and it is a real hole in what is
currently built.**

[`select_order()`](../src/gnn_healing/intent/biem.py#L53) enforces the change-risk concurrency
policy, at most one high-risk remediation per domain, like this:

```python
chosen, used, domains = [], 0, set()
for b in sorted(branches, key=lambda b: -b.priority_score):
    if one_change_per_domain and b.domain in domains:
        continue
```

That `domains` set lives for the duration of one function call in one process. It is correct for
the single-loop experiment in [`cli.py`](../src/gnn_healing/cli.py) and it fails the moment two
sites heal at once, which is the normal case on the
[`edge-ai-grid`](https://github.com/dkypuros/geometric-network-healing/tree/edge-ai-grid) branch
where every site runs its own loop.

**What to store.**

| | |
|---|---|
| key | `lock:change:{domain}` |
| primitive | set-if-not-exists with expiry |
| TTL | maximum expected remediation duration plus margin |

The expiry matters more than the lock. A healer that crashes mid-remediation must not hold RAN
hostage forever.

---

## Job 3: make the loop idempotent

Two separate concerns, both one-liners in Redis, both serious if skipped.

**Deduplicate incidents.** One fault seen by three sites is one incident, not three. Key on a
fingerprint of origin node plus onset bucket, set-if-not-exists with a TTL covering the incident
lifetime. Without this, the ranked candidates from
[`healing_cycles()`](../src/gnn_healing/orchestration/tmf641.py#L58) get pursued several times in
parallel.

**Guard service orders.** [`decompose_to_service_orders()`](../src/gnn_healing/orchestration/tmf641.py#L15)
produces orders with stable ids; [`MockOrchestrator`](../src/gnn_healing/orchestration/tmf641.py#L27)
executes them. In production, a duplicated order is not a harmless retry. Issuing "reroute traffic"
twice is an outage. Record the id before dispatch, set-if-not-exists, refuse on collision.

---

## The topology is two tiers, not one

A site must keep detecting when it cannot reach the hub. If the filter checkpoint lives at the hub,
you have coupled a site's ability to see a fault to the very link the fault may have broken.

```
edge site s                                  metro hub
-----------                                  ---------
solver  --checkpoint every step-->  LOCAL    lock:change:{domain}     <-- shared
  |      (local Redis or local disk)         incident:{fingerprint}
  | 7 scalars/step                           intent:{id} cycle state
  v                                          order:{id} idempotency
site summary stream  ------------------->  regional merge, BIEM selection
                                                   |
                                                   v
                                             orchestrator  <-- authoritative gate
```

Local state is local. Shared state is shared. The site summary stream is a transport question, not
a state question; Redis Streams works and is lighter, but if the operator already runs Kafka, that
is the telco convention and there is no reason to fight it.

---

## The safety caveat, stated plainly

**Distributed locks in Redis are advisory, not authoritative.** The well-known failure is an expiry
firing while a slow process still believes it holds the lock, which hands you exactly the two
concurrent changes the policy forbids. For network remediation that is the dangerous case, not a
theoretical one.

The correct posture:

- Redis rejects the obvious conflicts fast and cheaply, covering almost all of them.
- The **orchestrator remains the final gate**, with its own concurrency check and a monotonic
  fencing token it validates before acting.

Redis makes the policy efficient. It must never be the only thing enforcing it.

Note the asymmetry in what a Redis outage costs you:

| lost state | consequence |
|---|---|
| filter checkpoints | sites re-warm, detection gap, recoverable |
| incident dedup | duplicate incidents, noisy, not dangerous |
| **concurrency locks** | **two simultaneous changes in one domain, dangerous** |

That last row is the argument for the orchestrator gate, and also for persistence and replication
if you do rely on Redis here.

---

## What is actually on the cluster, checked rather than assumed

Oberon runs three Redis instances today. Checking them changed this note's argument, so the
findings are recorded here rather than left as theory.

| where | what it is | usable for the filter checkpoint? |
|---|---|---|
| `argocd/argocd-redis` | ArgoCD's internal cache | no, platform internal, never touch |
| `quay-enterprise/quay-quay-redis` | Quay's internal cache | no, platform internal, never touch |
| `confidential-agents/redis` | shared cache for an ephemeral agent sandbox | **no, and the reason matters** |

The third one looked promising: a general-purpose Redis 7, sitting beside MLflow and a
confidential-computing agent pool. It is started with `--save '' --appendonly no` and mounts no
volume. That is persistence explicitly disabled, both snapshotting and append-only file, and the
pod has already restarted three times in sixty-one days.

So the one non-platform Redis on this cluster is a deliberately ephemeral cache. It cannot serve
the single use case this note argues Redis is for, which is keeping a recursive filter's state
alive across a restart so that detection lead time is not lost. **A cache with persistence turned
off cannot checkpoint anything across its own restart.** It is correctly configured for what it
does, which is coordinating short-lived agents that can afford to lose state; it is simply not
the same job.

Its network policy also scopes egress to pods labelled `app=ephemeral-agent` inside that
namespace, so reaching it from elsewhere would mean editing someone else's policy.

Two conclusions. If the filter checkpoint is ever built here, it needs its own Redis with
append-only persistence enabled, in its own namespace, and sized from a banded covariance rather
than the dense one. And for benchmark results there is already an answer that needs nothing new:
the MLflow service in that same namespace, which the lab's other repositories already use for
tracing.

## What stays out

- **The twin.** It needs history and versioning. See [`docs/nvidia.md`](nvidia.md).
- **The ontology.** A few hundred triples defining the vocabulary belong in a Turtle file under
  version control, authored with `rdflib`, not in a server. Property graph and RDF are different
  data models and the Redis graph module is gone either way.
- **The math.** A network hop costs more than the matrix-vector product it would be wrapping.
  Nothing in [`flatness.py`](../src/gnn_healing/baselines/flatness.py) or
  [`gnn/model.py`](../src/gnn_healing/gnn/model.py) should know Redis exists.
- **This repository.** Every number in
  [the monograph](math/geometric-network-healing.pdf) and
  [the paper](flatness/four_lines_one_node.pdf) runs from one command on a laptop. That
  reproducibility is the asset. A broker and a cache would buy nothing and cost a reader the
  ability to check the work.

---

## Known gaps this note records

Stated here so a reviewer does not have to find them.

| gap | where | severity |
|---|---|---|
| concurrency policy is per-process | [`select_order()`](../src/gnn_healing/intent/biem.py#L53) | real; breaks with concurrent sites |
| dense `N × N` covariance | [`FlatnessSolver.field()`](../src/gnn_healing/baselines/flatness.py#L65) | blocks regional scale and checkpointing alike |
| no order idempotency | [`tmf641.py`](../src/gnn_healing/orchestration/tmf641.py) | fine in simulation, unsafe in production |
| no filter persistence | [`flatness.py`](../src/gnn_healing/baselines/flatness.py) | costs detection lead time on restart |

None of these affect the results in the monograph or the paper, because those run a single process
over complete episodes. All of them affect a deployment.

---

## Where this sits in the repo

| document | question it answers |
|---|---|
| [the paper](flatness/four_lines_one_node.pdf) | is the root cause learnable or readable? |
| [the monograph](math/geometric-network-healing.pdf) | what is the full method, derived? |
| [`docs/nvidia.md`](nvidia.md) | where does the compute and the graph live at scale? |
| [`docs/industry-comparison.md`](industry-comparison.md) | how does this relate to the deployed industry design? |
| **this note** | **what state exists, where does it live, and what breaks if it is lost?** |
| [`docs/edge-ai-grid.md`](https://github.com/dkypuros/geometric-network-healing/blob/edge-ai-grid/docs/edge-ai-grid.md) | how does it split across edge sites? (branch) |
