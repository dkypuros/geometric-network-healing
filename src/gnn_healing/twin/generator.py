"""Chapter 2: synthetic twin generator.

Produces a heterogeneous directed graph with the topology of a small regional network:
cells <- gNBs <- aggregation routers <- core routers, links as first-class nodes,
a small 5G core (AMF/SMF/UPF), and service instances that span RAN and Core.
Every cell carries a revenue density (Eq. 7.1) used by the business intent layer.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from .schema import NodeType, Relation, RELATIONS, REL_INDEX, EFFECT, K


# Region / cell profile catalogue: business-criticality weights declared by the operator (Eq. 7.1).
PROFILES = {"mass_event_venue": 3.0, "transit_hub": 2.0, "enterprise_critical": 2.0, "residential": 1.0}


@dataclass
class Twin:
    nodes: pd.DataFrame            # columns: id, type, domain, revenue, profile, profile_weight
    edges: pd.DataFrame            # columns: src, dst, rel
    graph: nx.DiGraph = field(repr=False)

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def node_type_ids(self) -> np.ndarray:
        types = list(NodeType)
        return self.nodes["type"].map({t: i for i, t in enumerate(types)}).to_numpy()

    def adjacency(self) -> np.ndarray:
        """Cause->effect adjacency A with A[v,u]=1 iff v->u (Eq. 2.2)."""
        A = np.zeros((self.n, self.n), dtype=np.float32)
        A[self.edges["src"].to_numpy(), self.edges["dst"].to_numpy()] = 1.0
        return A

    def effect_matrix(self) -> np.ndarray:
        """E in R^{N x K}: row v is e_tau(v) (Eq. 4.3)."""
        return np.stack([EFFECT[NodeType(t)] for t in self.nodes["type"]]).astype(np.float32)

    def to_pyg(self, add_reverse: bool = True) -> Data:
        """Homogeneous Data with node_type / edge_type ids (relational GNN input, Eq. 3.4).

        add_reverse appends the transposed relations as R extra relation ids so the
        network can also pass messages effect->cause, which the inverse problem of
        Chapter 6 requires.
        """
        src = torch.tensor(self.edges["src"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(self.edges["dst"].to_numpy(), dtype=torch.long)
        rel = torch.tensor([REL_INDEX[Relation(r)] for r in self.edges["rel"]], dtype=torch.long)
        if add_reverse:
            src, dst, rel = (torch.cat([src, dst]), torch.cat([dst, src]),
                             torch.cat([rel, rel + len(RELATIONS)]))
        return Data(
            edge_index=torch.stack([src, dst]),
            edge_type=rel,
            node_type=torch.tensor(self.node_type_ids, dtype=torch.long),
            revenue=torch.tensor(self.nodes["revenue"].to_numpy(), dtype=torch.float32),
            num_nodes=self.n,
        )

    def downstream(self, v: int) -> set[int]:
        """Forward reach of v along cause->effect edges (blast radius, Eq. 7.2)."""
        return nx.descendants(self.graph, v)


def generate_twin(n_core: int = 2, n_agg: int = 6, gnb_per_agg: int = 2,
                  cells_per_gnb: int = 3, n_upf: int = 2, n_services: int = 6,
                  seed: int = 0) -> Twin:
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    rows: list[dict] = []
    erows: list[dict] = []

    def add(t: NodeType, revenue: float = 0.0, profile: str = "n/a") -> int:
        i = len(rows)
        rows.append({"id": i, "type": t.value, "revenue": float(revenue), "profile": profile,
                     "profile_weight": PROFILES.get(profile, 1.0)})
        G.add_node(i, type=t.value)
        return i

    def link(s: int, d: int, r: Relation) -> None:
        erows.append({"src": s, "dst": d, "rel": r.value})
        G.add_edge(s, d, rel=r.value)

    core = [add(NodeType.ROUTER) for _ in range(n_core)]
    agg = [add(NodeType.ROUTER) for _ in range(n_agg)]
    amf, smf = add(NodeType.AMF), add(NodeType.SMF)
    upfs = [add(NodeType.UPF) for _ in range(n_upf)]
    for u in upfs:
        link(smf, u, Relation.SMF_CONTROLS_UPF)
        link(core[rng.integers(n_core)], u, Relation.CORE_FEEDS_UPF)

    # transport: each agg router homed to a core router through a LINK node
    for a in agg:
        c = core[rng.integers(n_core)]
        l = add(NodeType.LINK)
        link(l, a, Relation.LINK_FEEDS_ROUTER)
        link(c, a, Relation.CORE_FEEDS_AGG)
    # core ring links
    for i in range(n_core):
        l = add(NodeType.LINK)
        link(l, core[i], Relation.LINK_FEEDS_ROUTER)

    cells: list[int] = []
    for a in agg:
        for _ in range(gnb_per_agg):
            g = add(NodeType.GNB)
            link(a, g, Relation.AGG_BACKHAULS_GNB)
            link(amf, g, Relation.AMF_CONTROLS_GNB)
            for _ in range(cells_per_gnb):
                # revenue density: heavy-tailed, so a few cells carry most revenue (Eq. 7.1)
                profile = str(rng.choice(list(PROFILES), p=[0.08, 0.12, 0.15, 0.65]))
                c = add(NodeType.CELL, revenue=float(rng.lognormal(mean=0.0, sigma=1.0)), profile=profile)
                link(g, c, Relation.GNB_SERVES_CELL)
                cells.append(c)

    services = []
    for _ in range(n_services):
        s = add(NodeType.SERVICE)
        services.append(s)
        link(upfs[rng.integers(n_upf)], s, Relation.UPF_TERMINATES_SERVICE)
    # every cell carries at least one service (round-robin), plus a few extra random carriers
    for i, c in enumerate(cells):
        link(c, services[i % n_services], Relation.CELL_CARRIES_SERVICE)
    for s in services:
        for c in rng.choice(cells, size=max(1, len(cells) // (2 * n_services)), replace=False):
            if not G.has_edge(int(c), s):
                link(int(c), s, Relation.CELL_CARRIES_SERVICE)

    from .schema import DOMAIN_OF
    nodes = pd.DataFrame(rows)
    nodes["domain"] = nodes["type"].map(lambda t: DOMAIN_OF[NodeType(t)])
    return Twin(nodes=nodes, edges=pd.DataFrame(erows), graph=G)
