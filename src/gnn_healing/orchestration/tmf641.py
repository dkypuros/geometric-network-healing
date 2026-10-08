"""Chapter 8: execution layer and closed loop.

Intent (TMF921-shaped) decomposes into per-domain service orders (TMF641-shaped). The mock
orchestrator "executes" an order by removing the fault injection in the simulator from the
execution step onward, and we re-simulate to measure recovery and revenue-weighted exposure:
    Loss = sum_t sum_u rev_u * d_u(t)                                   (Eq. 8.1)
"""
from __future__ import annotations
import numpy as np
from ..twin.generator import Twin
from ..sim.dynamics import simulate, Episode, NominalParams, PropagationParams
from ..sim.faults import Fault


def decompose_to_service_orders(intent: dict) -> list[dict]:
    orders = []
    for tgt in intent["expression"]["targets"]:
        orders.append({
            "@type": "ServiceOrder", "id": f"{intent['id']}-so-{tgt['priority']}",
            "relatedIntent": intent["id"], "priority": tgt["priority"],
            "serviceOrderItem": [{"action": "modify", "service": {
                "domain": tgt["domain"], "target": tgt["target"], "operation": tgt["action"]}}],
        })
    return orders


class MockOrchestrator:
    def __init__(self, exec_delay: int = 3):
        self.exec_delay = exec_delay
        self.log: list[dict] = []

    def execute(self, orders: list[dict], t_now: int) -> int:
        """Returns the step at which the first (highest-priority) order takes effect."""
        for o in sorted(orders, key=lambda o: o["priority"]):
            self.log.append({"t": t_now, "order": o["id"], "domain": o["serviceOrderItem"][0]["service"]["domain"],
                             "state": "completed", "effective_at": t_now + self.exec_delay})
        return t_now + self.exec_delay


def closed_loop(twin: Twin, fault: Fault, T: int, remediate_at: int | None, seed: int,
                nominal: NominalParams = NominalParams(), prop: PropagationParams = PropagationParams(),
                recovered_below: float = 0.25) -> dict:
    """Re-simulate with remediation and report recovery time and revenue-weighted exposure."""
    rng = np.random.default_rng(seed)
    ep = simulate(twin, T, fault, rng, nominal, prop, remediate_at=remediate_at)
    rev = twin.nodes["revenue"].to_numpy()
    exposure = float((ep.D * rev[None, :]).sum())
    recovered = None
    if remediate_at is not None:
        for t in range(remediate_at, T):
            if ep.D[t].max() < recovered_below:
                recovered = t; break
    return {"remediate_at": remediate_at, "recovered_at": recovered,
            "time_to_recover": None if recovered is None else recovered - remediate_at,
            "revenue_weighted_exposure": exposure}


def healing_cycles(twin: Twin, fault: Fault, T: int, candidates: list[int], first_effective: int,
                   cycle_delay: int, seed: int, max_cycles: int = 3, **kw) -> dict:
    """Verify-and-re-trigger (a Reflection stage): apply the top candidate; if the field does not
    recover, re-trigger with the next ranked candidate after cycle_delay steps. Returns the cycle
    count and the final closed_loop record. A remediation only removes the injection when it is
    applied at the true origin, so wrong candidates cost time and exposure, never credit."""
    t = first_effective
    for cycle, v in enumerate(candidates[:max_cycles], start=1):
        rec = closed_loop(twin, fault, T, t if v == fault.origin else None, seed=seed, **kw)
        if rec["recovered_at"] is not None:
            rec["cycles"] = cycle; rec["healed"] = True
            return rec
        t += cycle_delay
    rec = closed_loop(twin, fault, T, None, seed=seed, **kw)
    rec["cycles"] = min(len(candidates), max_cycles); rec["healed"] = False
    return rec
