"""Chapter 7: business intent as constrained selection (a BIEM-style extension).

Revenue at risk of a candidate root cause v (Eq. 7.2), with operator-declared profile weights w_u:
    R(v) = sum_{u in reach(v) u {v}} w_u * rev_u * d_hat_u
(w = 1 everywhere reduces to pure revenue density; w from a region-profile catalogue reproduces the
"utility = profile weight x revenue density x service impact" pattern.)
Selection (Eq. 7.3): order branches by R(v)/c(v) and execute greedily under a resource budget B.
Same fault, different response: the ranking depends on rev, which lives in the business layer,
not on the network implementation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from ..twin.generator import Twin
from ..twin.schema import NodeType, DOMAIN_OF

# resource units and cost per remediation target type (business-layer vocabulary)
RESOURCE = {NodeType.CELL: 1, NodeType.GNB: 2, NodeType.ROUTER: 3, NodeType.LINK: 3,
            NodeType.UPF: 4, NodeType.AMF: 5, NodeType.SMF: 4, NodeType.SERVICE: 1}
ACTION = {NodeType.CELL: "cell_reset_and_reparameterize", NodeType.GNB: "gnb_restart_and_reparent",
          NodeType.ROUTER: "reroute_traffic_and_reload", NodeType.LINK: "switch_to_protection_path",
          NodeType.UPF: "scale_out_and_steer_sessions", NodeType.AMF: "throttle_and_scale_signaling",
          NodeType.SMF: "scale_control_plane", NodeType.SERVICE: "no_op"}


@dataclass
class RemediationBranch:
    target: int
    target_type: str
    domain: str
    action: str
    rca_rank: int
    revenue_at_risk: float
    resource_units: int
    priority_score: float


def propose_branches(twin: Twin, ranking: list[int], d_hat: np.ndarray, k: int = 3) -> list[RemediationBranch]:
    out = []
    rev = (twin.nodes["revenue"] * twin.nodes["profile_weight"]).to_numpy()
    for rank, v in enumerate(ranking[:k]):
        t = NodeType(twin.nodes.loc[v, "type"])
        reach = twin.downstream(v) | {v}
        idx = np.fromiter(reach, dtype=int)
        r = float(np.sum(rev[idx] * np.clip(d_hat[idx], 0, None)))
        res = RESOURCE[t]
        out.append(RemediationBranch(target=int(v), target_type=t.value, domain=DOMAIN_OF[t],
                                     action=ACTION[t], rca_rank=rank, revenue_at_risk=r,
                                     resource_units=res, priority_score=r / res))
    return out


def select_order(branches: list[RemediationBranch], budget: int,
                 one_change_per_domain: bool = True) -> list[RemediationBranch]:
    """Greedy by priority_score under the resource budget (Eq. 7.3). With one_change_per_domain
    (a change-risk concurrency policy) at most one high-risk remediation per domain is executed
    per healing cycle; the rest are held as ranked fallbacks."""
    chosen, used, domains = [], 0, set()
    for b in sorted(branches, key=lambda b: -b.priority_score):
        if one_change_per_domain and b.domain in domains:
            continue
        if used + b.resource_units <= budget:
            chosen.append(b); used += b.resource_units; domains.add(b.domain)
    return chosen


def to_tmf921_intent(branches: list[RemediationBranch], intent_id: str = "healing-1") -> dict:
    """A TMF921-shaped intent object (structure, not a full TIO/RDF rendering)."""
    return {
        "id": intent_id, "@type": "Intent", "name": "business-aware-healing",
        "expression": {
            "@type": "IntentExpression",
            "targets": [{"target": f"node:{b.target}", "domain": b.domain, "action": b.action,
                         "priority": i + 1, "revenueAtRisk": round(b.revenue_at_risk, 3),
                         "resourceUnits": b.resource_units} for i, b in enumerate(branches)],
            "expectation": "restore service KPIs to nominal in revenue-priority order",
        },
        "extensions": {"BIEM": {"ordering": "revenue_at_risk/resource_units", "budget_applied": True}},
    }
