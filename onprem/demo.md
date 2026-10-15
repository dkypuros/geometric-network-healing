# The demo: use case and talk track

Five minutes at a screen. One page, one button, three panels. The live URL is in your local
notes, not in this repository.

---

## Who this is for and what they own

The audience that this lands with hardest is whoever owns **root cause discovery on premises**.
In the published industry architecture that component sits on the vendor's own infrastructure,
not in the cloud half, and it works by orchestrating around thirteen tool calls: performance
counters, alarm correlation, a config check, z-scores over a materialised view, a topology walk,
and a discovery pipeline against a knowledge base.

That is the box this replaces. Not the detector, not the orchestrator, not the intent layer.
One box.

So the frame is never "my system against your system." It is: *here is your discovery step, done
by a subtraction, with a proof instead of a trace.*

---

## The use case, in one paragraph

A degradation starts somewhere in a RAN, Core and Transport network. It breaches no threshold,
so no alarm fires and no ticket opens. It spreads along the dependency graph until a hundred and
forty nodes look unhealthy and a human has to work out which one is actually broken. That triage
is the expensive part: it is cross-domain, it is manual, and today it is answered either by a war
room or by an agent making a dozen queries. This answers it with one matrix subtraction, in
microseconds, deterministically, and hands back the arithmetic as the explanation.

---

## Before you start

- Pick **agg** or **upf** from the selector. Both have a real blast radius. Avoid `sector` on the
  first run: it has one downstream dependent and the picture is unimpressive.
- Run it once before anyone is watching. The fault origin is random each time, so you want to
  know what you are about to see.
- Have the repository open in a second tab for the questions in the last section.

---

## The talk track

### Opening, about twenty seconds

> Everyone building autonomous networks right now is training a model to find the root cause.
> I want to show you what happens if you write down the equation instead. This has zero learned
> parameters and it needs zero labelled incidents. On the same test it matches a graph network
> with two hundred and forty one thousand parameters.

Point at the header: the identity is right there, `s(t) = d(t) − M d(t−1)`.

Then press **Run the incident** and talk through the three panels as they land.

### Panel one — the fault is hiding

> This is every node in the network, four KPI counters each, over time. A transport fault starts
> here at step fifty and ramps up over thirty-five steps. Look at it. There is nothing to see.
> A static threshold on these counters does not fire until step eighty-nine.
>
> That gap is the whole problem. This is a silent degradation, and by the time anything alarms,
> customers have been complaining for an hour.

### Panel two — the field says many nodes are sick

> Now I filter those counters through the dynamics I already know. The topology is known. How a
> fault spreads along it is known, or measurable. That recovers the degradation field.
>
> And now the problem is extremely visible, which is the trap. A hundred and thirty-eight nodes
> are degraded. Every single one of them looks broken. One of them is. The other hundred and
> thirty-seven are just downstream.
>
> This is the state a war room is in.

### Panel three — the operator explains all but one

> Here is the move. I apply the operator and subtract what it predicts.
>
> Every node that is merely suffering the consequences is fully explained by its causes, so it
> goes dark. Everything cancels. What is left is the one node the dynamics cannot account for.
>
> One row stays lit. That is the fault.

Pause here. Let the picture sit. Then scroll to the cards.

### The numbers

> Three things worth taking away.
>
> The operator resolved the origin at step seventy. The threshold alarm fires at eighty-nine.
> That is nineteen steps of lead, and at a five-minute telemetry cadence that is an hour and a
> half before anything would otherwise have paged a human.
>
> A hundred and thirty-eight nodes in the blast radius, all explained but one.
>
> And it cost about four hundred microseconds per step, with no model loaded.

### The explanation, which is the actual pitch

Scroll to the residual bars.

> This is what I hand your engineer. Seven hundred and eighty-one numbers, one per node. Seven
> hundred and eighty of them are flat. One is sixteen times the median.
>
> That is not a confidence score and it is not an attention weight and it is not a traversal
> path. It is arithmetic they can redo by hand for any node on that chart and get the same
> answer. The healing intent carries the same evidence inline, so whatever consumes it can audit
> the claim rather than trust it.

### Close

> No model weights. No training pipeline. No GPU. No labelled incidents. No egress: the container
> holds numpy and a topology file. It is deterministic, so the same counters always give the same
> node, which is the part that makes it certifiable.
>
> It does not want to own your loop. It replaces a discovery step and emits the intent shape your
> executor already takes.

---

## Say the limits out loud, before they ask

This is the part that earns the room, and skipping it costs more than it saves.

> Three things you should know.
>
> First, the counters you just watched are synthetic. The topology is operator-shaped but it is
> not anyone's real network.
>
> Second, I ran this against real telemetry once, on a live five-plane 5G stack, and it found
> nothing. I went and checked why: in that lab the radio and core functions were placeholder
> containers passing zero bytes to each other, so there was no propagation to detect. The method
> correctly reported no clear origin rather than inventing one, and that is the only encouraging
> thing about that result. It is written up in the repository.
>
> Third, I ran an experiment designed to break this. When the assumed physics are wrong, a
> trained graph network beats it: 0.93 against 0.67. A model-based method with the wrong model is
> worse than a learned one with no model. That is measured and it is in the paper.
>
> What I need is a month of real topology and telemetry to find out which of those worlds we are
> actually in.

---

## The questions you will get

**"How do you know the couplings?"**
They can be identified from unlabelled healthy telemetry. On clean data the per-relation fit
recovers every coupling essentially exactly and the memory constant to four decimal places. On
noisy data it recovers the ordering and the magnitude but underestimates, and the gap is
observation noise. Experiment seven in the paper.

**"What if the topology is wrong?"**
Then the method cannot work, and you can find that out before deploying anything. There is a
precondition check that asks, on healthy telemetry alone, whether a cause predicts its effects
one step later more often than an unrelated pair does. It passes a network with real propagation
at 2.5x lift and fails one without at p of 1.0. Run it first.

**"Does it detect, or only localise?"**
Only localise. Pair it with whatever already opens your loop. A service-layer z-score is fine and
is what the industry demo uses.

**"What about non-linear propagation?"**
Honestly, untested. Queueing has knees and this assumes linearity. It is the most likely way this
breaks on real data and it is on the open list.

**"Why is this better than the model?"**
On accuracy, it is not better, it ties. It is better on everything around accuracy: no labels, no
training, no GPU, deterministic, auditable, microseconds, and it transfers to a new topology by
recomputing a matrix rather than retraining.

**"Can it handle two faults at once?"**
Yes, and by construction rather than by luck. The residual support just has two nodes in it. No
change to the method.

---

## If it misses while you are standing there

It can. Say so plainly and use it:

> That one missed. Look at the separation number: it is low. It is telling you it has no clear
> answer rather than handing you a confident wrong one. That is the behaviour I want from
> something that is going to open a change window on a live network.

Then run it again on `agg`.
