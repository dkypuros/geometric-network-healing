"""Chapter 5 and 6 at inference time.

Detection (Eq. 5.3): a(t) = max_v d_hat_v(t); fire at the first t with a(t) > theta on
min_run consecutive steps, theta calibrated on nominal episodes (no alarm, no static KPI threshold).
RCA (Eq. 6.2): rank V by z_v(t_det + delay).
"""
from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from .model import HealingGNN
from .backend import device
from ..sim.dynamics import Episode


@torch.no_grad()
def _scores(model: HealingGNN, ep: Episode, base: Data, W: int):
    """Returns (deg [T, N], rca [T, N]) with NaN rows for t < W."""
    dev = device(); model.eval()
    T, N, K = ep.X.shape
    deg = np.full((T, N), np.nan, dtype=np.float32)
    rca = np.full((T, N), np.nan, dtype=np.float32)
    X = torch.from_numpy(ep.X)
    ei, et, nt = base.edge_index.to(dev), base.edge_type.to(dev), base.node_type.to(dev)
    for t in range(W, T):
        hist = X[t - W:t].permute(1, 0, 2).reshape(N, W * K).to(dev)
        out = model(hist, ei, et, nt)
        deg[t] = out["degradation"].cpu().numpy()
        rca[t] = out["rca"].cpu().numpy()
    return deg, rca


def calibrate_threshold(model, nominal_eps: list[Episode], base: Data, W: int, margin: float = 1.15) -> float:
    """theta = margin * max over nominal episodes of max_v d_hat_v(t): zero false alarms on calibration."""
    peak = 0.0
    for ep in nominal_eps:
        deg, _ = _scores(model, ep, base, W)
        peak = max(peak, float(np.nanmax(deg)))
    return margin * peak


def detect(model, ep: Episode, base: Data, W: int, theta: float, min_run: int = 3, start: int = 0) -> int | None:
    deg, _ = _scores(model, ep, base, W)
    run = 0
    for t in range(max(W, start), ep.T):
        run = run + 1 if np.nanmax(deg[t]) > theta else 0
        if run >= min_run:
            return t
    return None


def rank_root_causes(model, ep: Episode, base: Data, W: int, t: int) -> list[int]:
    _, rca = _scores(model, ep, base, W)
    t = min(max(t, W), ep.T - 1)
    return np.argsort(-rca[t]).tolist()


def degradation_estimate(model, ep: Episode, base: Data, W: int, t: int) -> np.ndarray:
    deg, _ = _scores(model, ep, base, W)
    return deg[min(max(t, W), ep.T - 1)]
