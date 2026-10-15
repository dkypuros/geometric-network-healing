# First contact with real telemetry

**Result: the method found nothing, the data had nothing in it to find, and the experiment
produced a tool that says which of those two it is before you deploy.**

Everything else in this repository runs on a simulator written for this repository. This is the
first test against telemetry nobody generated for the purpose. It is a negative result and it is
the most useful thing here.

## What was run

A live five-plane 5G stack on a bare-metal Single Node OpenShift cluster at the Intel Folsom
Acceleration Lab: a 28-pod 5G service-based core, an O-RAN split RAN with O-RU, O-DU, O-CU-CP,
O-CU-UP and a near-RT RIC, an SMO plane, and a BSS/OSS plane. 47 running pods.

| step | what |
|---|---|
| topology | 47 pods, 51 dependency edges derived from 3GPP and O-RAN relations, oriented cause to effect |
| telemetry | real per-pod CPU, memory, network receive and transmit, pulled from the cluster's Thanos at 30-second resolution, 99% coverage |
| control | a healthy 90-minute window with no fault |
| injection | `kubectl rollout restart deployment/odu` in `telco-ran` at 02:29:01Z, graceful and self-healing |
| measurement | a 40-minute window spanning the restart, read off with the same solver used everywhere else |

## What happened

**On the healthy window the method correctly declined to name anything.**

| | healthy real window | synthetic injected fault |
|---|---|---|
| margin over runner-up | 0.106 | 0.767 |
| robust z over noise floor | 5.4 | 15 to 19 |
| confidence | 0.592 | 0.958 |

That is the behaviour you want. No fault, no clear origin, and it says so.

**On the injected window it failed.** The O-DU ranked 13 of 47. Separation was *lower* than on
the healthy window, z of 3.6 and margin of 0.057, so the method was not confidently wrong. It
reported no clear origin, and there was none to report.

## Why, established rather than assumed

Before blaming the method I checked whether the experiment contained a signal. It did not.

**The modelled dependency chain carries no traffic.** Median over the window:

| function | CPU cores | net rx B/s | net tx B/s |
|---|---|---|---|
| O-RU | 0.0008 | **0.0** | **0.0** |
| O-CU-CP | 0.0002 | **0.0** | **0.0** |
| O-CU-UP | 0.0002 | **0.0** | **0.0** |
| AMF | 0.0002 | **0.0** | **0.0** |
| O-DU | 0.0101 | 3,360 | 4,676 |

Most of these pods run the same generic container image standing in for different network
functions. They are placeholders in a lab, not functions passing each other traffic. A pod that
transmits zero bytes cannot propagate a degradation to anything downstream.

**The largest counter movement anywhere in the window was 31%.** The industry demo localised its
fault on a counter that moved 7,327%.

**The dependency graph is not visible in the telemetry.** Strong lagged correlation appears on
0.0% of modelled edges and 0.5% of random pairs. There is no structure to invert.

So the operator I built was fiction. Not wrong in its couplings, fiction in its edges: those
relations are real in 3GPP and absent in these containers.

## The tool this produced

If the precondition can fail silently, it should be testable in advance. It is, on healthy
telemetry alone, with no faults and no labels. `precondition.py` asks whether a cause predicts
its effects one step later more often than an unrelated pair does.

```
python experiments/oberon_live/precondition.py --counters counters.npz
```

| | control, propagation true by construction | the live lab stack |
|---|---|---|
| strong lagged prediction on edges | 9.8% | **0.0%** |
| the same on random pairs | 3.9% | 0.5% |
| lift | **2.5x** | 0x |
| permutation p | 0.0000 | 1.0000 |
| verdict | **PASS** | **FAIL** |

**Two of my own attempts at this check were wrong before this one worked**, and the failures are
worth recording because they are easy to repeat. Contemporaneous correlation on raw counters
fails its own control, because zero-lag correlation is dominated by shared diurnal shape and says
nothing about direction. And the *mean* lagged correlation also fails, scoring 1.11x on the
control against 1.03x on a stack that propagates nothing, because in any real network most pairs
are quiet most of the time and the average is noise. Only the tail separates them: how many edges
carry a strong lagged signal, not how strong the average edge is.

The check also reports something uncomfortable about the control: forward and reversed edges score
the same, 1.01x. Lagged correlation alone does not establish direction. Cause and effect are
separated by the operator's structure, not by any asymmetry visible in the raw statistics.

## What this does and does not say

It does not say the method works on real networks. It has still never been shown to.

It does not say the method failed. It was handed data with no propagation in it and reported no
propagation, which is correct behaviour and the first evidence that its null case is honest on
data I did not construct.

What it does say is that the precondition is real, specific, and checkable, and that any
deployment of this method should run that check first. On a stack that is not carrying load, the
answer is known in advance and no amount of filtering changes it.

## What would make this a real test

A network actually carrying traffic, where the modelled edges are the ones that really couple.
Three candidates, in order of effort:

1. The OAI core with UERANSIM attached and passing user-plane traffic, so the O-RU to O-DU to
   O-CU chain has bytes in it.
2. A load generator against the BSS/OSS plane, where the frontend, gateway and inventory pods do
   measurably depend on each other and already show the largest real movements in this data.
3. Operator telemetry, which remains the ask in the paper.

Run `precondition.py` on each before running anything else.

## Reproduce

```bash
python experiments/oberon_live/topology_from_cluster.py   # 47 pods, 51 3GPP edges
python experiments/oberon_live/fetch_metrics.py --minutes 90 --step 30
python experiments/oberon_live/precondition.py            # check before trusting anything
python experiments/oberon_live/run.py --expect odu        # score against a known fault
```

Needs a cluster login. The topology builder reads whatever is running, so it will reflect your
cluster rather than mine.
