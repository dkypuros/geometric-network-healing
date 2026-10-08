"""Chapter 5/6: supervised windows from episodes.

A sample at time t is (X[t-W:t], target X[t], origin label if t >= t0 else -1).
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from ..twin.generator import Twin
from .faults import sample_fault
from .dynamics import Episode, simulate, NominalParams, PropagationParams


def make_episodes(twin: Twin, n_episodes: int, T: int = 160, nominal_fraction: float = 0.2,
                  seed: int = 0, nominal: NominalParams = NominalParams(),
                  prop: PropagationParams = PropagationParams()) -> list[Episode]:
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n_episodes):
        fault = None if rng.random() < nominal_fraction else sample_fault(twin.nodes, T, rng)
        eps.append(simulate(twin, T, fault, rng, nominal, prop))
    return eps


def windows_from_episode(ep: Episode, base: Data, W: int, stride: int = 1,
                         rca_after: int = 0) -> list[Data]:
    """Each Data shares base.edge_index/edge_type/node_type and carries
    x: [N, W*K] history, y: [N, K] next-step KPIs, origin: scalar (-1 if none / pre-onset)."""
    out = []
    T, N, K = ep.X.shape
    for t in range(W, T, stride):
        hist = torch.from_numpy(ep.X[t - W:t]).permute(1, 0, 2).reshape(N, W * K)
        y = torch.from_numpy(ep.X[t])
        origin = ep.origin if (ep.fault is not None and t >= ep.onset + rca_after) else -1
        d = Data(x=hist, y=y, edge_index=base.edge_index, edge_type=base.edge_type,
                 node_type=base.node_type, num_nodes=N,
                 origin=torch.tensor([origin]), t=torch.tensor([t]))
        out.append(d)
    return out
