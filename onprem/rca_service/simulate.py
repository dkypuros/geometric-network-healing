"""Inject a fault into a topology and emit KPI counters, so the service can be demonstrated
and tested without operator data. Nothing here is used at inference time."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import Topology


@dataclass
class Injection:
    origin: int
    onset: int
    severity: float
    ramp: int

    def at(self, t: int) -> float:
        if t < self.onset:
            return 0.0
        return self.severity * min(1.0, (t - self.onset + 1) / self.ramp)


def inject(topo: Topology, M: np.ndarray, steps: int = 120, origin: int | None = None,
           severity: float = 0.5, ramp: int = 30, onset: int | None = None,
           noise: float = 0.30, ar: float = 0.6, seed: int = 0):
    """Returns (counters [T, N, K], injection). Counters are what an operator would collect:
    per-node KPI time series with diurnal shape and correlated noise, and a slow ramp that
    breaches no fixed threshold for a long time."""
    rng = np.random.default_rng(seed)
    N, K = topo.n, len(topo.kpi_names)
    if origin is None:
        origin = int(rng.integers(0, N))
    if onset is None:
        onset = steps // 3
    inj = Injection(origin, onset, severity, ramp)

    E = np.array([topo.effect[x["type"]] for x in topo.nodes], dtype=np.float64)
    amp = 0.4 * rng.uniform(0.5, 1.5, size=(N, K))
    phase = rng.uniform(0, 2 * np.pi, size=(N, K))
    eps = np.zeros((N, K))
    d = np.zeros(N)
    X = np.zeros((steps, N, K))
    for t in range(steps):
        eps = ar * eps + rng.normal(0, noise, size=(N, K))
        nominal = amp * np.sin(2 * np.pi * t / 96 + phase) + eps
        src = np.zeros(N); src[origin] = inj.at(t)
        d = M @ d + src
        X[t] = nominal + d[:, None] * E
    return X, inj
