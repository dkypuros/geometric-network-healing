"""Chapter 5: graph-free baselines.

Static threshold alarm (Eq. 5.1): fire at the first t with max_k |x_vk(t)| > theta for any v.
Per-node z-score: same, but on a rolling standardization of each node's own history. Neither
uses the graph, so neither can tell a structural event from a noisy node.
"""
from __future__ import annotations
import numpy as np
from ..sim.dynamics import Episode


def threshold_alarm_detection(ep: Episode, theta: float = 2.5, min_run: int = 3, start: int = 0) -> int | None:
    """First t >= start where some node exceeds theta on min_run consecutive steps. None if never."""
    exceed = (np.abs(ep.X) > theta).any(axis=2)          # [T, N]
    run = np.zeros(ep.X.shape[1], dtype=int)
    for t in range(start, ep.T):
        run = np.where(exceed[t], run + 1, 0)
        if (run >= min_run).any():
            return t
    return None


def _rolling_z(ep: Episode, window: int) -> np.ndarray:
    """max over KPIs of |rolling z| per node: [T, N] (NaN before `window`)."""
    T, N, K = ep.X.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(window, T):
        hist = ep.X[t - window:t]
        mu, sd = hist.mean(axis=0), hist.std(axis=0) + 1e-6
        out[t] = np.abs((ep.X[t] - mu) / sd).max(axis=1)
    return out


def zscore_detection(ep: Episode, window: int = 24, z: float = 3.0, min_run: int = 3, start: int = 0) -> int | None:
    zz = _rolling_z(ep, window)
    run = np.zeros(ep.X.shape[1], dtype=int)
    for t in range(max(window, start), ep.T):
        run = np.where(zz[t] > z, run + 1, 0)
        if (run >= min_run).any():
            return t
    return None


def calibrate_zscore(nominal_eps: list[Episode], window: int = 24, margin: float = 1.15) -> float:
    """z such that the per-node rolling z-score never fires on the nominal calibration episodes."""
    return margin * max(float(np.nanmax(_rolling_z(ep, window))) for ep in nominal_eps)


def degraded_set(ep: Episode, t: int, threshold: float = 0.5) -> set[int]:
    """Ground-truth degraded nodes at t (from the latent field D). Used for triage metrics."""
    return set(np.flatnonzero(ep.D[t] > threshold).tolist())


def service_baseline(nominal_eps: list[Episode], service_idx) -> tuple[np.ndarray, np.ndarray]:
    """Clean per-service, per-KPI mean and std from nominal episodes (a 'clean 30-day baseline')."""
    S = np.concatenate([ep.X[:, service_idx, :] for ep in nominal_eps], axis=0)
    return S.mean(axis=0), S.std(axis=0) + 1e-6


def _service_z(ep: Episode, service_idx, mu, sd) -> np.ndarray:
    S = ep.X[:, service_idx, :]
    return np.abs((S - mu) / sd).max(axis=2)                      # [T, n_services]


def calibrate_service_z(nominal_eps: list[Episode], service_idx, mu, sd, margin: float = 1.15) -> float:
    return margin * max(float(_service_z(ep, service_idx, mu, sd).max()) for ep in nominal_eps)


def service_zscore_trigger(ep: Episode, service_idx, mu, sd, z: float, confirm: int = 3,
                           start: int = 0) -> int | None:
    """Industry-style service-layer trigger: z-score of each *service* node's KPIs against a clean
    baseline, with a confirmation gate of `confirm` consecutive breaches. Graph-free and
    service-only: it sees symptoms, never causes, and cannot name a root cause."""
    zz = _service_z(ep, service_idx, mu, sd)
    run = np.zeros(zz.shape[1], dtype=int)
    for t in range(start, ep.T):
        run = np.where(zz[t] > z, run + 1, 0)
        if (run >= confirm).any():
            return t
    return None
