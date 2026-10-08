"""Training loop. Loss (Eq. 6.3):
    L = || x_hat - x ||^2  +  || d_hat - d ||^2  +  lambda * CE( softmax_V(z), origin )
The CE term is only applied on post-onset windows of faulty episodes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .model import HealingGNN
from .backend import device
from ..sim.dynamics import Episode
from ..sim.dataset import windows_from_episode


@dataclass
class TrainConfig:
    window: int = 8
    hidden: int = 64
    layers: int = 3
    epochs: int = 30
    batch_size: int = 32
    lr: float = 2e-3
    lam_rca: float = 1.0
    stride: int = 2
    rca_after: int = 5       # start supervising RCA this many steps after onset
    seed: int = 0


def build_windows(episodes: list[Episode], base: Data, cfg: TrainConfig) -> list[Data]:
    out = []
    for ep in episodes:
        ws = windows_from_episode(ep, base, cfg.window, cfg.stride, cfg.rca_after)
        for w in ws:
            w.d = torch.from_numpy(ep.D[int(w.t)])        # latent degradation target
        out.extend(ws)
    return out


def train(episodes: list[Episode], base: Data, cfg: TrainConfig, log=print) -> HealingGNN:
    torch.manual_seed(cfg.seed)
    dev = device()
    N = base.num_nodes
    model = HealingGNN(cfg.window, cfg.hidden, cfg.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loader = DataLoader(build_windows(episodes, base, cfg), batch_size=cfg.batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(cfg.epochs):
        model.train()
        tot = {"forecast": 0.0, "deg": 0.0, "rca": 0.0}
        n = 0
        for batch in loader:
            batch = batch.to(dev)
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.node_type)
            B = batch.num_graphs
            l_f = torch.mean((out["forecast"] - batch.y) ** 2)
            l_d = torch.mean((out["degradation"] - batch.d) ** 2)
            l_r = ce(out["rca"].view(B, N), batch.origin.view(B))
            loss = l_f + l_d + cfg.lam_rca * l_r
            opt.zero_grad(); loss.backward(); opt.step()
            tot["forecast"] += l_f.item() * B; tot["deg"] += l_d.item() * B; tot["rca"] += l_r.item() * B; n += B
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            log(f"epoch {epoch:3d}  forecast {tot['forecast']/n:.4f}  degradation {tot['deg']/n:.4f}  rca {tot['rca']/n:.4f}")
    return model
