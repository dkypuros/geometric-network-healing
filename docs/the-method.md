# The method, in one page

The [monograph](math/geometric-network-healing.pdf) derives this over 29 pages and the
[paper](flatness/four_lines_one_node.pdf) argues it over 8. This is the five-minute version:
what the method is, what is proven, what is measured, and what is still open.

## The object

A fault is four numbers. Origin node, onset step, severity, ramp length. They are hiding inside
`N × K × T` observations, which for a regional network is tens of millions of numbers. The whole
method is recovering the four from the many.

Everything follows from refusing to treat the observations as the object.

## The operator

Faults spread along the topology, and the topology is known. Write that down:

```
M = rho*I + sum over relations r of ( beta_r * P_r ),    P_r = D_in^-1 * A_r^T,    sum_r P_r = P
```

`rho` is memory, how much degradation a node keeps step to step. `beta_r` is coupling, how much
of a cause's degradation its effects inherit along relation `r`. The couplings are declared in
[the ontology](ontology/network-healing.ttl), which is why that file is mathematics and not
documentation: a fibre cut and a signalling storm do not propagate at the same strength, and a
single scalar cannot say so.

Faults attenuate per hop when `max_r beta_r / (1 - rho) < 1`.

## The claim

The dynamics are `d(t+1) = M d(t) + s(t)` with the source `s` nonzero at exactly one node.
Rearrange:

```
d(t+1) - M d(t)  =  s(t),   which is zero everywhere except the origin
```

That is the entire method. Apply the operator, read the support. The proof is the rearrangement.

Two consequences come free. Several simultaneous faults give a support with several nodes, no
change to the method. And the residual at the origin *is* the source, so onset and severity are
read off rather than estimated.

## The difficulty

You never see `d`. You see KPIs, `x_v = n_v + d_v * e_tau(v)`, the field smeared through a
per-type effect direction and buried in diurnal and correlated noise. So the field needs an
estimator before the residual can be taken. That is a Kalman filter, because the model is linear
and the noise is Gaussian, and the residual is then taken on the estimate rather than on the
truth. Everything that is still hard traces back to this one substitution.

## The structural asymmetry with learning

This is the part worth understanding.

A graph network solving the same problem must learn the **adjoint**. Degradation flows cause to
effect, but inference must flow effect back to cause, which is why the architecture carries the
transposed relations explicitly and why it needs labelled incidents to learn what to send along
them.

The read-off needs no adjoint at all. It applies the forward operator once and subtracts.

> One method learns the inverse. The other looks at what the forward model fails to explain.

That is why one needs a quarter of a million parameters and labelled incidents, and the other
needs none of either.

## Where it stands

**Proven.** The flatness identity. The several-origins corollary. That setting every `beta_r`
equal collapses the per-relation operator to the published single-coupling one exactly, which a
test asserts, so turning the feature on cannot move earlier results.

**Measured.** Against a 241,222-parameter relational graph network on the same episodes with the
same calibration, the read-off ties on detection rate, false alarms and root cause, with zero
learned parameters. Per-relation coupling recovers the propagation physics essentially exactly
on a clean field, `rho = 0.8500` and mean coupling error `0.0044`, where a single scalar on the
same clean data collapses to `0.0055` rather than returning an average.

**Open, and stated because it is open.**

| gap | why it matters |
|---|---|
| projection noise is 0.88 of the field's own standard deviation | it is what limits identification; the fix is expectation-maximisation between the filter and the regression, not done here |
| the doubt statistic fails | the argument that a wrong operator makes the field visibly non-flat is right, the statistic built to measure it is not |
| the covariance is dense | quadratic in memory, cubic per step; this is the scaling wall, not compute |
| hit@1 saturates at 1.00 | the twin is too easy for the downstream task to discriminate between operators |

## A result about shape, not implementation

Measured on an Intel Xeon 6767P with 256 vCPU:

| threads | solver | graph network |
|---|---|---|
| 1 | 172 µs | 4.2 ms |
| 16 | 130 µs | 8.6 ms |
| 64 | 128 µs | 22.5 ms |
| 128 | 128 µs | 187 ms |

The solver is flat. The network degrades by a factor of forty-five. That is not a coding
difference. The read-off is dense linear algebra, which parallelises; message passing is scatter
and gather over an edge list, which at this size does not. The two methods have opposite
computational character, and it follows from their mathematical form.

It also means bfloat16 and the matrix extensions, which accelerate dense multiply, help the half
of the system nobody benchmarks and not the half everybody does.

## Reproduce any of it

```bash
make pipeline     # the head-to-head table
make experiments  # E2-E6, including the one the learned model wins
make e7           # per-relation coupling identified from unlabelled telemetry
make ontology     # the relation vocabulary and its couplings
```
