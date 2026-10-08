# Edge: LADN, PDU session anchors, and the NVIDIA AI grid

Branch `edge-ai-grid`. Chapter 12 of the monograph has the derivation; this is the operational view.

## Why the edge matters for this use case

Silent-degradation lead time is measured in steps. If every KPI window first travels from the
site to a central lake, the lead time is spent on transport and batching before the model sees it.
NVIDIA's AI grid puts accelerated compute in the central offices, metro hubs and cell sites that
already host the RAN (AI-RAN on the same GPUs). That is where the twin's site subgraph lives, and
where the first pass of detection and localisation should run.

Two 3GPP constructs (TS 23.501) make the edge a first-class part of the twin rather than an
implementation detail:

| Construct | What it is | What it does to the graph |
|---|---|---|
| **PDU Session Anchor (PSA)** | The UPF that terminates N6 for a PDU session. For edge traffic it is co-located at the site, behind an intermediate UPF (I-UPF) on N9. | A `psa` node per site, fed by the site's transport and the regional `iupf`, controlled by the SMF. Its degradation reaches only that site's local services. |
| **Local Area Data Network (LADN)** | A DN reachable only inside a service area (a TA list), typically an enterprise, venue or MEC application. | A `ladn` node per site anchored at the site's PSA, with the site's cells as area members, serving an edge `service`. It carries its own revenue and a profile weight. |
| **SSC mode** | Session and Service Continuity mode of the sessions anchored at a PSA: mode 1 keeps the anchor, mode 2 is break-before-make, mode 3 is make-before-break. | Not a node. It sets the *resource cost* of the remediation "relocate the anchor to the backup PSA": cheap in mode 3, expensive in mode 1. `--ssc-mode` on the pipeline. |

The new fault types are `psa_overload` and `ladn_dn_degrade`; the new remediation actions are
`psa_relocate_ssc_mode{3,2,1}_to_backup` (re-anchor at the next site's PSA),
`iupf_reselect_n9_path`, and `ladn_fallback_to_regional_dn`.

## What runs where

```
cell site / edge POP (one per aggregation site)            metro hub / regional
----------------------------------------------            ---------------------
site subgraph: 23 nodes (owned + 1-hop boundary)          regional merge of site summaries
HealingGNN under Triton / TensorRT  (~ms per step)        full-graph HealingGNN for regional faults
local detect: max d_hat over owned nodes                  BIEM selection across sites (budget B)
local RCA: top-k logits                                   intent -> TMF641 orders -> orchestrator
uplink per step: 1 + 2k scalars (k=3 -> 7)                Dynamo-served LLM agents (see below)
KPI windows never leave the site
```

Measured on the branch (`make pipeline-edge`, see `results/results.md`): per-site inference with a
regional merge versus full-graph inference, same model, same episodes. The partition is a graph
cut; its cost shows up as hit@1 on faults whose origin is a regional node (core routers, I-UPF,
AMF, SMF) that no single site owns. Those are routed to the regional full-graph pass.

## Where NVIDIA Dynamo fits, and where it does not

The GNN is small (tens of thousands of parameters, a 23-node subgraph per site). It does not need
Dynamo. It runs under Triton Inference Server or as a TensorRT engine on the site GPU next to the
AI-RAN workload.

Dynamo serves the *reasoning* around the GNN. The industry design wraps the loop in LLM agents
(sense, localise, prioritise, execute, verify). On an AI grid those agents run as an
OpenAI-compatible service with:

- **Disaggregated prefill and decode.** The long, repeated context (topology description,
  intent vocabulary, standards payload templates) is prefilled once; per-incident decode runs
  on the smaller edge GPUs.
- **KV-aware routing.** Dynamo's router sends an incident to the worker whose KV cache already
  holds that site's context, so a site's agent is "warm" for its own topology.
- **NIXL transfer.** KV state moves between hub and site workers over the grid's fabric.

`deploy/ai-grid/` has an illustrative topology file describing the roles; it is a description of
the deployment shape, not a tested Dynamo configuration. Consult the Dynamo documentation for the
current CLI and config schema.

## Honesty note

Everything in this branch is measured on the synthetic twin on CPU. The AI-grid placement, Triton
packaging and Dynamo serving are documented, not exercised here. The measurable claim is the
partition cost table, which is reproducible with one command.
