"""Chapter 4: fault catalog.

A fault is (origin node v0, onset t0, peak severity s, ramp length T_r) (Eq. 4.1).
Slow ramps are the "silent degradation" regime: the per-node signal stays under a
static alarm threshold for a long time while the *graph* already shows the pattern.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from ..twin.schema import NodeType


class FaultType(str, Enum):
    FIBER_DEGRADE = "fiber_degrade"
    ROUTER_CONGESTION = "router_congestion"
    UPF_OVERLOAD = "upf_overload"
    SLEEPING_CELL = "sleeping_cell"
    AMF_SIGNALING_STORM = "amf_signaling_storm"


# origin type, severity range (per-step injection; steady state at origin = s/(1-rho)), ramp range (steps)
FAULT_CATALOG = {
    FaultType.FIBER_DEGRADE:       (NodeType.LINK,   (0.35, 0.60), (20, 50)),
    FaultType.ROUTER_CONGESTION:   (NodeType.ROUTER, (0.30, 0.55), (15, 40)),
    FaultType.UPF_OVERLOAD:        (NodeType.UPF,    (0.35, 0.60), (20, 45)),
    FaultType.SLEEPING_CELL:       (NodeType.CELL,   (0.40, 0.65), (10, 30)),
    FaultType.AMF_SIGNALING_STORM: (NodeType.AMF,    (0.30, 0.50), (25, 60)),
}


@dataclass(frozen=True)
class Fault:
    kind: FaultType
    origin: int
    onset: int
    severity: float
    ramp: int

    def injection(self, t: int) -> float:
        """s(t) = s * clip((t - t0)/T_r, 0, 1)  (Eq. 4.1)."""
        if t < self.onset:
            return 0.0
        return self.severity * min(1.0, (t - self.onset + 1) / self.ramp)


def sample_fault(nodes, T: int, rng: np.random.Generator, kind: FaultType | None = None) -> Fault:
    kind = kind or FaultType(rng.choice([k.value for k in FaultType]))
    ntype, (s_lo, s_hi), (r_lo, r_hi) = FAULT_CATALOG[kind]
    candidates = nodes.index[nodes["type"] == ntype.value].to_numpy()
    origin = int(rng.choice(candidates))
    onset = int(rng.integers(T // 4, T // 2))
    return Fault(kind=kind, origin=origin, onset=onset,
                 severity=float(rng.uniform(s_lo, s_hi)), ramp=int(rng.integers(r_lo, r_hi)))
