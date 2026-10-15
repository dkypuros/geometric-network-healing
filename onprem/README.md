# Deterministic root cause, on premises

A root-cause service that finds the origin of a cross-domain degradation by solving the
propagation equations rather than learning them. No model weights, no training, no labelled
incidents, no GPU, and no network egress. One container, numpy, and a topology file.

```
d(t+1) = M d(t) + s(t)        M = rho*I + sum over relations r of ( beta_r * P_r )
s(t)   = d(t) - M d(t-1)      zero at every node except where the fault started
```

Apply the operator to the degradation field and read where the result is not zero. That is the
whole method. The residual is also the explanation: one number per node, and on the topology
shipped here 780 of 781 of them are flat.

## Measured

On the included RAN, Core and Transport topology of 781 nodes and 996 edges:

| | |
|---|---|
| root cause hit@1 | **1.000** over 27 injected faults spanning 8 node types |
| inference | **172 µs per step**, 52 ms for a 140-step window |
| operator build | 0.63 s once at load, 14 Riccati iterations |
| learned parameters | **0** |
| labelled incidents required | **0** |
| separation of the origin from the noise floor | robust z of 15 to 19 |
| deterministic | yes, identical input gives identical output |

Faults were injected at aggregation routers, cell-site routers, eNodeBs, gNodeBs, UPFs, MMEs,
sectors and the HSS. Every one was located.

## Demoing it

[`DEMO.md`](DEMO.md) is the use case and a five-minute talk track: what to click, what to say at
each of the three panels, the numbers that land, the limits to state before anyone asks, and the
questions that come back.

## Running now

Deployed on the lab cluster, two replicas, built in-cluster from this directory:

```
https://rca-service-<namespace>.apps.<your-cluster>
```

| | |
|---|---|
| operator | 781 nodes, 996 edges, stability 0.933, built in 1.4 s at pod start |
| learned parameters | 0 |
| live demo | `GET /v1/demo?kind=agg` injects a fault and locates it |

The page itself is one screen. It exists to show a single thing: the residual, one bar per node,
where one bar stands over hundreds of flat ones. That picture is the argument. Alongside it sit
the finding and the TMF921-shaped healing intent with its evidence block, so the claim can be
audited rather than trusted.

```bash
oc new-build --binary --name=rca-service --strategy=docker -n gnn-healing
oc start-build rca-service --from-dir=onprem --follow -n gnn-healing
kubectl -n gnn-healing apply -f onprem/deploy/rca-service.yaml
```

## Try it locally

```bash
python onprem/demo.py                    # inject, locate, explain, 30 seconds
python onprem/demo.py --kind csr         # fault a cell-site router instead
python -m rca_service.server --topology topology/ran_core_transport.json --port 8080
```

```
FOUND     AGG-0012   CORRECT
          confidence 0.968, blast radius 141 nodes

the explanation is the residual, one number per node:
     7.3700  AGG-0012   <- the only node not obeying the dynamics
     1.5238  SECTOR-0396
     1.4680  SECTOR-0394
```

## The API

| | |
|---|---|
| `GET /healthz`, `GET /readyz` | liveness and readiness |
| `GET /v1/topology` | what operator is loaded, its couplings, its stability margin |
| `POST /v1/root-cause` | counters `[steps][nodes][kpis]` → origin, onset, severity, blast radius, residual |
| `POST /v1/healing-intent` | the same input → a TMF921-shaped healing intent |
| `POST /v1/topology` | replace the loaded topology at runtime |

The healing intent carries its own evidence, so whatever consumes it can audit the claim:

```json
"evidence": {
  "method": "propagation-residual",
  "statement": "the only node where the observed field does not obey d(t) = M d(t-1)",
  "residual_at_origin": 0.368612,
  "confidence": 0.928
}
```

## Why this shape, for an on-premises deployment

- **Nothing to phone home to.** No model registry, no inference endpoint, no cloud dependency.
  The container holds numpy and a JSON file.
- **Deterministic and therefore certifiable.** The same counters always produce the same node.
  There is no sampling, no temperature, no retry that returns something else.
- **Auditable.** The explanation is a number per node and an equation, not a traversal or an
  attention weight. An engineer can recompute it by hand for any node.
- **Cheap.** Half a CPU and a gigabyte. Single-threaded on purpose, see below.
- **It composes.** It replaces a discovery step and emits the intent shape an existing executor
  already consumes. It does not want to own orchestration.

## Bring your own topology

The service is topology-agnostic. It takes a typed directed graph and the couplings, and builds
the operator from them:

```json
{
  "rho": 0.85,
  "kpi_names": ["throughput", "latency", "prb_utilization", "drop_rate"],
  "coupling":  { "agg_feeds_csr": 0.14, "mme_manages_enodeb": 0.04 },
  "effect":    { "agg": [-0.3, 1.0, 0.8, 0.4] },
  "nodes": [ {"id": 0, "name": "AGG-0000", "type": "agg", "domain": "Transport"} ],
  "edges": [ {"src": 0, "dst": 1, "relation": "agg_feeds_csr"} ]
}
```

Edges point the way a fault propagates, cause to effect. A published graph often names the
inverse relation, for example "eNodeB managed_by MME"; that is the same edge read backwards.

The couplings are the only thing that is not pure topology, and they do not have to be guessed.
They can be identified from unlabelled healthy telemetry; see
[`scripts/ontology_experiment.py`](../scripts/ontology_experiment.py) in the parent repository
and the ontology at [`onprem/ontology/ran-core-transport.ttl`](ontology/ran-core-transport.ttl).

The service refuses a topology whose couplings would make faults amplify rather than attenuate,
which is `max(beta_r) / (1 - rho) >= 1`.

## Deploying

```bash
podman build -t rca-service:latest onprem/
kubectl apply -f onprem/deploy/rca-service.yaml
```

Runs under the OpenShift `restricted-v2` security context: non-root with an arbitrary UID, no
privilege escalation, all capabilities dropped, read-only root filesystem.

**Scale by replica, not by thread.** The workload is dense linear algebra on a small matrix, and
threading it costs more than it saves. Measured on an Intel Xeon 6767P with 256 vCPU, the same
computation at 256 threads was 3,568 times slower than at one thread, because Kubernetes enforces
CPU limits with CFS quota while the container still sees every core. The manifest pins the thread
environment variables to 1 for this reason.

## What this does not do, stated plainly

- **It does not detect.** It localises. Pair it with whatever already opens your loop.
- **The onset is a detection time, not a start time.** It lags the true onset by roughly half the
  ramp length, a median of 16 steps in testing. It is reported with that caveat in the payload.
- **The couplings must be right.** A model-based method with the wrong model is worse than a
  learned one with no model. That is measured, not hypothesised; see experiment six in
  [the paper](../docs/flatness/four_lines_one_node.pdf), where a trained graph network is the
  more robust of the two under misspecified physics.
- **It has not met production telemetry.** Everything above is on synthetic counters from a
  topology of the right shape. Real propagation is not exactly linear and a signalling storm does
  not spread like a fibre cut. The next experiment is one month of real topology and telemetry,
  and the result goes in this repository either way.

## Where the evidence lives

| | |
|---|---|
| [the method in one page](../docs/the-method.md) | what is proven, measured, and open |
| [the paper](../docs/flatness/four_lines_one_node.pdf) | seven experiments, including the two the method loses |
| [the monograph](../docs/math/geometric-network-healing.pdf) | the full derivation |
| [the ontology](../docs/ontology/network-healing.ttl) | couplings as declared terms, not constants in code |
