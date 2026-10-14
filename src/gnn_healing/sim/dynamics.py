"""Chapter 4: KPI dynamics and fault propagation as directed diffusion.

Nominal signal per node (Eq. 4.2):
    n_v(t) = a_v sin(2 pi t / P + phi_v) + eps_v(t),  eps AR(1) with coefficient r
Degradation field d(t) in R^N (Eq. 4.4), a damped directed heat flow:
    d(t+1) = rho d(t) + beta * D_in^{-1} A^T d(t) + s(t) 1_{v0}
Observed KPI (Eq. 4.3):
    x_v(t) = n_v(t) + d_v(t) * e_{tau(v)}
The GNN never sees d; it sees x and the graph. Labels come from d and the fault.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ..twin.generator import Twin
from .faults import Fault


@dataclass
class NominalParams:
    noise: float = 0.3       # AR(1) innovation std
    ar: float = 0.6          # AR(1) coefficient
    diurnal_amp: float = 0.4
    period: int = 96         # steps per "day"


@dataclass
class PropagationParams:
    rho: float = 0.85        # memory of the degradation field
    beta: float = 0.13       # coupling; beta/(1-rho) = 0.87 < 1 so faults attenuate per hop
    degraded_threshold: float = 0.5  # d_v > this => node counted as degraded (labels)
    # Per-relation coupling beta_r. None keeps the single-beta operator that the published
    # results use. A dict, or from_ontology=True, makes a fibre cut and a signalling storm
    # propagate at different strengths, which is physically right and which a single scalar
    # cannot represent. See gnn_healing.operator and docs/ontology/network-healing.ttl.
    beta_by_relation: dict | None = None
    from_ontology: bool = False


@dataclass
class Episode:
    X: np.ndarray            # [T, N, K] observed KPIs
    D: np.ndarray            # [T, N] latent degradation field (ground truth)
    fault: Fault | None      # primary fault (first), kept for the single-origin pipeline
    T: int
    faults: list = field(default_factory=list)   # all injected faults (multi-origin experiments)

    @property
    def onset(self) -> int:
        return self.fault.onset if self.fault else self.T

    @property
    def origin(self) -> int:
        return self.fault.origin if self.fault else -1


def simulate(twin: Twin, T: int, fault: "Fault | list[Fault] | None", rng: np.random.Generator,
             nominal: NominalParams = NominalParams(),
             prop: PropagationParams = PropagationParams(),
             remediate_at: int | None = None) -> Episode:
    """Run T steps. If remediate_at is given, the injection is removed from that step on
    (the closed loop of Chapter 8 calls this to measure recovery)."""
    from ..operator import build_operator
    N = twin.n
    # M = rho I + beta P, or rho I + sum_r beta_r P_r when per-relation coupling is on
    M = build_operator(twin, prop.rho, beta=prop.beta,
                       beta_by_relation=prop.beta_by_relation, from_ontology=prop.from_ontology)
    E = twin.effect_matrix()                              # [N, K]
    K = E.shape[1]

    amp = nominal.diurnal_amp * rng.uniform(0.5, 1.5, size=(N, K))
    phase = rng.uniform(0, 2 * np.pi, size=(N, K))
    eps = np.zeros((N, K), dtype=np.float32)
    d = np.zeros(N, dtype=np.float32)
    X = np.zeros((T, N, K), dtype=np.float32)
    D = np.zeros((T, N), dtype=np.float32)

    faults = [] if fault is None else (fault if isinstance(fault, list) else [fault])
    for t in range(T):
        eps = nominal.ar * eps + rng.normal(0, nominal.noise, size=(N, K))
        n = amp * np.sin(2 * np.pi * t / nominal.period + phase) + eps
        inj = np.zeros(N, dtype=np.float32)
        if remediate_at is None or t < remediate_at:
            for f in faults:
                inj[f.origin] += f.injection(t)
        d = M @ d + inj
        D[t] = d
        X[t] = n + d[:, None] * E
    ep = Episode(X=X, D=D, fault=faults[0] if faults else None, T=T)
    ep.faults = faults
    return ep
