"""Edge-partitioned inference for an AI-grid deployment (branch chapter 12).

Each edge site runs the *same* HealingGNN on its own subgraph: the site's nodes plus a one-hop
boundary of regional nodes (core routers, I-UPF, AMF/SMF) that give it context. Only summaries
leave the site: the site's maximum degradation estimate and its top-k root-cause logits. A
regional aggregator merges them. KPI windows never leave the site.

The partition is a graph cut. Its cost is measured, not assumed: compare hit@k and detection
delay against full-graph inference on the same episodes (pipeline --edge).
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from ..twin.generator import Twin
from ..sim.dynamics import Episode
from .backend import device


def site_partition(twin: Twin, base: Data) -> dict[int, dict]:
    """For each site: owned node ids, the induced (owned + 1-hop boundary) subgraph, and a
    relabelled Data skeleton to run the model on."""
    site = twin.nodes["site"].to_numpy()
    ei = base.edge_index
    parts = {}
    for s in twin.sites:
        owned = np.flatnonzero(site == s)
        owned_t = torch.tensor(owned, dtype=torch.long)
        mask = torch.isin(ei[0], owned_t) | torch.isin(ei[1], owned_t)
        boundary = torch.unique(torch.cat([ei[0][mask], ei[1][mask]]))
        nodes = torch.unique(torch.cat([owned_t, boundary]))
        sub_ei, sub_et = subgraph(nodes, ei, edge_attr=base.edge_type, relabel_nodes=True, num_nodes=base.num_nodes)
        parts[s] = {"owned": owned, "nodes": nodes.numpy(), "edge_index": sub_ei, "edge_type": sub_et,
                    "node_type": base.node_type[nodes]}
    return parts


@torch.no_grad()
def edge_scores(model, ep: Episode, parts: dict[int, dict], W: int, k: int = 3):
    """Per step: each site computes (max d_hat over owned nodes, top-k (node, logit) over owned nodes
    and boundary nodes it can see). Returns the regional merge: deg_max [T], rca [T, N] (-inf where
    no site reported), and the number of scalars uplinked per site per step."""
    dev = device(); model.eval()
    T, N, K = ep.X.shape
    X = torch.from_numpy(ep.X)
    deg_max = np.full(T, np.nan, dtype=np.float32)
    rca = np.full((T, N), -np.inf, dtype=np.float32)
    uplink_scalars = 1 + 2 * k
    for t in range(W, T):
        hist_all = X[t - W:t].permute(1, 0, 2).reshape(N, W * K)
        dm = -np.inf
        for s, p in parts.items():
            nodes = p["nodes"]
            out = model(hist_all[nodes].to(dev), p["edge_index"].to(dev), p["edge_type"].to(dev), p["node_type"].to(dev))
            d = out["degradation"].cpu().numpy(); z = out["rca"].cpu().numpy()
            local = np.isin(nodes, p["owned"])
            dm = max(dm, float(d[local].max()))
            top = np.argsort(-z)[:k]                       # site reports its top-k candidates only
            for j in top:
                g = int(nodes[j]); rca[t, g] = max(rca[t, g], float(z[j]))
        deg_max[t] = dm
    return deg_max, rca, uplink_scalars


def edge_calibrate(model, nominal_eps: list[Episode], parts, W: int, margin: float = 1.15) -> float:
    peak = 0.0
    for ep in nominal_eps:
        dm, _, _ = edge_scores(model, ep, parts, W)
        peak = max(peak, float(np.nanmax(dm)))
    return margin * peak


def edge_detect_and_rank(model, ep: Episode, parts, W: int, theta: float, min_run: int = 3,
                         start: int = 0, rca_delay: int = 5):
    dm, rca, uplink = edge_scores(model, ep, parts, W)
    run, t_det = 0, None
    for t in range(max(W, start), ep.T):
        run = run + 1 if dm[t] > theta else 0
        if run >= min_run:
            t_det = t; break
    t_r = min(max((t_det if t_det is not None else ep.onset) + rca_delay, W), ep.T - 1)
    ranking = np.argsort(-rca[t_r]).tolist()
    return t_det, ranking, uplink
