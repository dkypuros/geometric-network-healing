# What the deployed demo actually does

[`docs/industry-comparison.md`](industry-comparison.md) was written from published material:
slides, an article, and a training appendix. This page is written from the running system, read
off two recordings of the booth demo at DTW on 7 October 2026. It corrects three things I had
wrong and confirms several I had guessed.

Everything below is visible on screen. Where the demo contradicts the slides, I record the demo.

---

## The architecture, as the UI labels it

The app is "Business-Aware GNN-Healing Networks", served from a Google Cloud address, with five
sections: Intent Translator, Digital Twin, Agents Execution, Business Value, Ask Me.

The Agents Execution page shows a ten-stage pipeline and, in its Agent Flow tab, names each agent
and the substrate it runs on:

| stage | role as labelled | runs on |
|---|---|---|
| GNN Inference | **Anomaly Detection** | GCP |
| Reflex Agent | Domain Triage & Priority | GCP |
| Detective Agent | **Root Cause Discovery** | Ericsson on-prem |
| Engineer Agent | Smart Orchestrator, BIEM Utility | GCP |
| Executor Agent | TMF641 Dispatch | Ericsson on-prem |
| Automation Engine | ESOA, Ericsson Service Orchestration & Assurance | Ericsson on-prem |
| Reflection | compliance check and loop closure | GCP |

The handoffs between GCP and Ericsson on-prem are drawn explicitly. That split is the multi-vendor
claim made concrete.

---

## Correction 1: the GNN detects, it does not localise

I had written that the model performs root cause by link prediction over an anomaly node. The UI
says otherwise. The box is labelled **GNN Inference, Anomaly Detection**, and at that stage the
panel reads:

> Anomaly score: 0.85. **Primary origin: unknown.** Blast radius: N/A downstream nodes.

Root cause arrives later, from the **Detective Agent**, and that agent is not a graph model. Its
Agent Flow panel shows **Tool Execution (13 tools)**, including:

- PM Counters (performance)
- FM Alarms, correlating against an alarm cluster view
- Config Check against change requests
- **Anomaly Z-Scores** against `mv_base_hex07_anom`, a materialised view of H3 resolution-7 bins
- **Topology Walker** over `EDGE_EntityToEntity`
- RCD Pipeline against a knowledge base

The evidence that actually names the node appears as a tool result:
`dl_prb_utilization +7327% on gNB-SYN-932`, alongside `8 alarms correlated — RAN domain confirmed`.

So root cause discovery here is **an agentic tool-use loop over queries, z-scores and a graph
walk**, not a learned inverse. That is a materially different design from the one I compared
against, and it is a fairer target.

## Correction 2: detection is a z-score, and they say so

The Business Value page states the K1 claim in its own words:

> Today, silent degradations surface through complaint aggregation with median lag measured in
> hours. The Catalyst detects them through **service-layer Z-score analysis within sub-minute
> windows**. [...] Compression exceeds 95 percent on the silent slice and is observable on
> **Champion-calibrated synthetic data in the working PoC**.

This confirms the protocol assumption in experiment one of [the paper](flatness/four_lines_one_node.pdf):
the industry trigger is a service-layer z-score, and the headline compression figure is measured
on synthetic data. Both are stated plainly by the system itself.

## Correction 3: the scale is smaller than I assumed

| view | size |
|---|---|
| Digital Twin | 1,367 nodes, 1,915 edges |
| scenario during execution | 194 nodes, 251 links |
| full network entity count | 6,271 entities |

I had been reasoning about regional twins of 100,000 nodes when discussing the dense-covariance
wall. The deployed demo is three orders of magnitude smaller than that, and only about eighteen
times this repository's 74-node twin.

**Correction, added after measuring on Oberon.** My first version of this paragraph said the
scaling objection did not bite at this size, because a dense covariance at 1,367 nodes is only
7.5 MB. That checked memory and ignored time. The per-step filter inverts an `N x N` matrix every
step, which is cubic, and on an Intel Xeon 6767P it costs 14.5 ms at 642 nodes and 219 ms at
1,222. The crossover where the learned model becomes the faster of the two sits between 220 and
642 nodes, well below the demo's scale.

The repair is mathematical rather than operational. The system is linear and time-invariant, so
the Kalman gain converges and has no business being recomputed every step. Solving the Riccati
recursion once at startup, which takes fourteen iterations, reduces each step to two
matrix-vector products:

| nodes | per-step filter | steady-state | speedup | same origin found |
|---|---|---|---|---|
| 642 | 14,533 µs | 48 µs | 304x | yes, 1.00 agreement |
| 1,222 | 62,852 µs | 262 µs | 240x | yes, 1.00 agreement |

At 1,222 nodes the learned model takes 11,894 µs, so the corrected solver is about **45 times
faster** rather than 18 times slower. Field estimates agree to a correlation of 0.998 and the
ranked origin is identical in every episode tested. `FlatnessSolver(..., steady_state=True)`.

---

## What the demo says about its own limits

Two disclosures appear in the system's own output, and they deserve to be quoted rather than
paraphrased.

On the scoring:

> Ranking Formula: `anomalyImpactScore = gnnConfidence × cellProfileWeight × revenueDensity ×
> timeOfDayFactor × appCriticalityFactor`
> ⚠ **Placeholder values — live GNN scoring pending**

On the execution:

> **SIMULATED** Automation Engine run (`eic_enm.mock: true`) — **ESOA was not contacted**;
> order TMF641-RAN-ACT-…

So in this run the business ranking used placeholder inputs and the orchestration engine was
mocked. That is entirely normal for a conference demo and I note it without criticism. It does
mean the end-to-end figure, 139.9 seconds against a 130 second target, measures the agent
pipeline and the API calls rather than a physical remediation.

---

## BIEM, read from the JSON-LD

The Intent Translator renders the Business Intent Extension Model directly. The vocabulary uses a
`biz:` prefix and reuses the TM Forum intent ontology under `imo:`:

```jsonld
"@type": "Intent",
"category": "BusinessOutcomeIntent",
"biz:activationCondition": "biz:RecoveryResourcesConstrained",
"biz:utilityFunction": "biz:WeightedBusinessImpactFunction",
"biz:businessReportingExpectation": ["imo:Degrades", "imo:Complies", "imo:Repaired"],
"biz:applicableRegionProfiles": [
    "Mass-Event-Venue", "Transit-Hub", "Premium-Consumer-Zone", "Enterprise-Critical-Urban"
],
"biz:cellBusinessProfileWeight": 3,
"biz:revenueDensityIndex": 0.85
```

Business intent decomposes to a service intent carrying five expectations (throughput, latency,
stability, video quality of experience, availability), and from there to a healing intent.

### How this compares with ours, which is the interesting part

Three of our four region profiles match theirs by name, independently chosen, and our weight of
3.0 for a mass-event venue matches `biz:cellBusinessProfileWeight` exactly. That is convergence,
not borrowing; the profiles were written before these recordings were available.

The important difference is what each ontology is *for*:

| | BIEM | [ours](ontology/network-healing.ttl) |
|---|---|---|
| indexes | the business layer | the propagation operator |
| key terms | region profiles, revenue density, utility function, activation condition | per-relation coupling, node classes, property hierarchy |
| answers | which fault to heal first | how a fault spreads |
| has a coupling term | **no** | yes, `gnh:coupling` is beta_r in `M = rho I + sum_r beta_r P_r` |

**These are complementary layers, not competing ones.** BIEM is a vocabulary for prioritising
remediation and has nothing to say about physics. Ours is a vocabulary for the operator and has
nothing to say about revenue. A complete system wants both, and nobody appears to have written
the second one down.

---

## What this changes in the comparison

| claim in [industry-comparison.md](industry-comparison.md) | status after watching the demo |
|---|---|
| detection is a service-layer z-score | confirmed, stated by their own UI |
| business utility is profile weight times revenue density | confirmed, exact five-factor formula now known |
| anomalies and faults are node types in the graph | confirmed, they appear in the node-type legend |
| four-layer intent decomposition | confirmed |
| root cause is link prediction over an anomaly node | **wrong**, it is a 13-tool agentic loop including z-scores and a topology walk |
| the twin is at operator scale | **wrong**, 1,367 nodes |
| the execution layer runs against a live orchestrator | **not in this run**, ESOA was mocked |

The honest summary is that the deployed system is less of a graph-learning system than its name
suggests and more of an agentic assurance pipeline with a graph-shaped anomaly detector in front
of it. That makes the comparison in this repository both fairer and more interesting: the
question is not whether a residual beats a GNN, but whether solving the operator beats
orchestrating thirteen queries.
