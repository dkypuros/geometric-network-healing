# How this relates to the industry design

> **Superseded in part.** This page was written from published slides and an article. Since then
> I watched two recordings of the running booth demo, which corrects three claims below, most
> importantly that root cause is *not* link prediction but a 13-tool agentic loop, and that the
> deployed twin is 1,367 nodes rather than operator scale. Read
> **[what-the-demo-shows.md](what-the-demo-shows.md)** alongside this.


A 2026 TM Forum Moonshot Catalyst on business-aware GNN healing (C26.0.965) describes a
production system of the same shape: a service-layer trigger, a GNN over a temporal digital twin
for cross-domain root cause, a business-intent layer that reorders remediation by revenue, and
standards-based execution (TMF921 intents decomposed into TMF641 service orders). This repository
was built independently from the geometric deep learning blueprint and then compared. Chapter 11
of the monograph has the full table; this is the short version.

| Stage | Industry design | This repo | Same math? |
|---|---|---|---|
| Trigger | service-layer z-score vs clean baseline, confirmation gate | same trigger as a baseline (`service_zscore_trigger`), plus a GNN degradation-field detector (Eq. 5.3) | yes, and we add one |
| Localise | link prediction on "originates-from" edges from an anomaly node, traversed in reverse | node softmax over all nodes (Eq. 6.2); transposed relations are the reverse traversal | equivalent for one incident (Remark 6.1) |
| Prioritise | utility = profile weight × revenue density × service impact; one major change at a time | R(v) = Σ w·rev·d̂ over blast radius ÷ resource cost, budget, one change per domain (Eq. 7.3) | theirs is our single-node special case |
| Execute | TMF921 v5 / TMF641 v5 into a vendor orchestrator | TMF921/641-shaped objects into a mock orchestrator | plumbing, not math |
| Verify | reflection stage: complies/degrades, re-trigger | re-simulate, check recovery, re-trigger next candidate (`healing_cycles`) | yes |
| Training data | 13-stage schema-faithful synthetic generator | directed diffusion on the causal graph (Eq. 4.4) | different, ours is the physics prior |
| Scale substrate | managed graph DB + managed training | WholeGraph + cuGraph-PyG + cuDF (docs/nvidia.md) | different vendor path |

What they have that we do not: a production-identical schema for the synthetic corpus, and the
standards contributions (an intent-ontology extension, a business-layer assessment questionnaire,
a proposed silent-degradation lead-time indicator). Neither is needed to reproduce the learning
stack. Both matter for deployment.

What we measure that they report as targets: silent-degradation lead time against calibrated
baselines, root-cause hit@k, triage reduction against the war-room set, and revenue-weighted
exposure with and without early healing. See `results/results.md`.
