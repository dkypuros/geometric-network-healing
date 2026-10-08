"""Chapter 2: the domain.

A telecom network is a heterogeneous directed graph
    G = (V, E, tau, rho)                                   (Eq. 2.1)
with node-type map tau: V -> T and edge-relation map rho: E -> R.
Edges point in the direction a fault *propagates* (cause -> effect).
"""
from __future__ import annotations
from enum import Enum


class NodeType(str, Enum):
    CELL = "cell"        # RAN
    GNB = "gnb"          # RAN
    ROUTER = "router"    # Transport
    LINK = "link"        # Transport (a link is a node so it can carry its own KPIs)
    UPF = "upf"          # Core (user plane)
    AMF = "amf"          # Core (control plane)
    SMF = "smf"          # Core (control plane)
    SERVICE = "service"  # Service layer (the trigger surface; no alarm needed)
    # edge extension (3GPP TS 23.501): intermediate UPF, PDU session anchor UPF at the edge site,
    # Local Area Data Network reachable only through that site's PSA
    IUPF = "iupf"        # Core (regional user plane, N3/N9)
    PSA = "psa"          # Core, edge-hosted PDU Session Anchor UPF (N9 in, N6 out)
    LADN = "ladn"        # Edge data network (enterprise / venue / MEC application)


DOMAIN_OF = {
    NodeType.CELL: "RAN", NodeType.GNB: "RAN",
    NodeType.ROUTER: "Transport", NodeType.LINK: "Transport",
    NodeType.UPF: "Core", NodeType.AMF: "Core", NodeType.SMF: "Core",
    NodeType.SERVICE: "Service",
    NodeType.IUPF: "Core", NodeType.PSA: "Edge", NodeType.LADN: "Edge",
}


class Relation(str, Enum):
    """Typed, directed relations. Direction = fault flow (cause -> effect)."""
    LINK_FEEDS_ROUTER = "link->router"
    CORE_FEEDS_AGG = "router(core)->router(agg)"
    AGG_BACKHAULS_GNB = "router(agg)->gnb"
    GNB_SERVES_CELL = "gnb->cell"
    AMF_CONTROLS_GNB = "amf->gnb"
    SMF_CONTROLS_UPF = "smf->upf"
    CORE_FEEDS_UPF = "router(core)->upf"
    UPF_TERMINATES_SERVICE = "upf->service"
    CELL_CARRIES_SERVICE = "cell->service"
    # edge extension
    CORE_FEEDS_IUPF = "router(core)->iupf"
    SMF_CONTROLS_IUPF = "smf->iupf"
    IUPF_FORWARDS_PSA = "iupf->psa"          # N9
    AGG_FEEDS_PSA = "router(agg)->psa"       # site transport feeds the co-located PSA
    SMF_CONTROLS_PSA = "smf->psa"
    PSA_ANCHORS_LADN = "psa->ladn"           # N6 into the local DN
    CELL_IN_LADN_AREA = "cell->ladn"         # LADN service area (TA list) membership
    LADN_SERVES = "ladn->service"


RELATIONS: list[Relation] = list(Relation)
REL_INDEX = {r: i for i, r in enumerate(RELATIONS)}

# KPI vector per node, standardized so nominal ~ 0 (Eq. 2.3). Same K for every type
# keeps the tensors rectangular; the *effect direction* differs per type (Eq. 4.3).
KPI_NAMES = ["throughput", "latency", "loss", "utilization"]
K = len(KPI_NAMES)

# Effect vector e_tau: how a unit of degradation shows up in each type's KPIs (Eq. 4.3).
EFFECT = {
    NodeType.CELL:    [-1.0, 0.6, 0.5, 0.2],
    NodeType.GNB:     [-0.8, 0.5, 0.4, 0.6],
    NodeType.ROUTER:  [-0.3, 0.9, 0.6, 1.0],
    NodeType.LINK:    [-0.2, 1.0, 1.0, 0.4],
    NodeType.UPF:     [-0.6, 0.8, 0.5, 1.0],
    NodeType.AMF:     [ 0.0, 1.0, 0.3, 0.9],
    NodeType.SMF:     [ 0.0, 0.9, 0.2, 0.8],
    NodeType.SERVICE: [-1.0, 0.8, 0.7, 0.0],
    NodeType.IUPF:    [-0.5, 0.8, 0.5, 1.0],
    NodeType.PSA:     [-0.7, 0.9, 0.5, 1.0],
    NodeType.LADN:    [-1.0, 0.9, 0.6, 0.3],
}
