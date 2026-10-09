"""Flatness root cause: solve the equations instead of learning them.

For a solid rectangle on a grid, the discrete derivative is zero everywhere except on four lines,
and reading the support of the derivative hands back the rectangle (Kypuros, flatness notes, 2026).
The same move on the network: the degradation field obeys d(t+1) = M d(t) + s(t) with
M = rho I + beta P. Away from the origin the field is *flat* under the operator (I - M); the
source s is nonzero only at the origin. So:

    1. estimate the field d from the KPIs       (project onto the effect direction, Kalman-filter
                                                 with the known dynamics; no labels, no training)
    2. apply the operator                        s_hat(t) = d_hat(t) - M d_hat(t-1)
    3. read the support                          origin = argmax_v  sum_{window} s_hat_v

Detection uses the same statistic as the GNN (max_v d_hat_v > theta, calibrated on nominal
episodes) so the two are compared on equal terms. Zero parameters are learned.
"""
from __future__ import annotations
import numpy as np
from ..twin.generator import Twin
from ..sim.dynamics import Episode, PropagationParams


def propagation_matrix(twin: Twin) -> np.ndarray:
    A = twin.adjacency()
    indeg = np.maximum(A.sum(axis=0), 1.0)
    return (A / indeg).T


def identify_operator(twin: Twin, episodes: list[Episode], iters: int = 20, delta: float = 0.1) -> tuple[float, float]:
    """Identify (rho, beta) from UNLABELLED episodes: regress d~(t+1) on [d~(t), P d~(t)] with a Huber
    loss (IRLS). The few (node, step) pairs carrying the source are outliers and get down-weighted;
    no fault labels are used. Needs excitation (some faults present), not labels."""
    P = propagation_matrix(twin)
    E = twin.effect_matrix(); proj = E / np.maximum((E ** 2).sum(axis=1, keepdims=True), 1e-6)
    xs, ys = [], []
    for ep in episodes:
        d = np.einsum("tnk,nk->tn", ep.X, proj)
        d = np.maximum(d, 0.0)
        xs.append(np.stack([d[:-1].ravel(), (d[:-1] @ P.T).ravel()], axis=1)); ys.append(d[1:].ravel())
    X = np.concatenate(xs); y = np.concatenate(ys)
    keep = X[:, 0] > 0.3                      # only steps with a visible field carry information about M
    X, y = X[keep], y[keep]
    w = np.ones(len(y))
    for _ in range(iters):
        Xw = X * w[:, None]
        theta, *_ = np.linalg.lstsq(Xw.T @ X + 1e-6 * np.eye(2), Xw.T @ y, rcond=None)
        res = y - X @ theta
        scale = np.median(np.abs(res)) / 0.6745 + 1e-9
        w = np.minimum(1.0, delta * scale / np.maximum(np.abs(res), 1e-9))
    return float(theta[0]), float(theta[1])


class FlatnessSolver:
    def __init__(self, twin: Twin, prop: PropagationParams = PropagationParams(),
                 q: float = 0.05, r: float = 0.08, window: int = 16, rho: float | None = None, beta: float | None = None):
        P = propagation_matrix(twin)
        self.N = twin.n
        rho = prop.rho if rho is None else rho; beta = prop.beta if beta is None else beta
        self.rho, self.beta = rho, beta
        self.M = rho * np.eye(self.N, dtype=np.float32) + beta * P      # propagation operator
        E = twin.effect_matrix()                                                  # [N, K]
        self.proj = E / np.maximum((E ** 2).sum(axis=1, keepdims=True), 1e-6)    # x -> d along e_tau
        self.q, self.r, self.window = q, r, window

    def field(self, ep: Episode) -> np.ndarray:
        """Kalman-filtered estimate of d(t) from x(t): [T, N]. Causal (no look-ahead)."""
        T, N, K = ep.X.shape
        d_raw = np.einsum("tnk,nk->tn", ep.X, self.proj)                          # projected observation
        d = np.zeros(N, dtype=np.float32); Pm = np.eye(N, dtype=np.float32)
        Q, R = self.q * np.eye(N, dtype=np.float32), self.r * np.eye(N, dtype=np.float32)
        out = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            d_pred = self.M @ d
            P_pred = self.M @ Pm @ self.M.T + Q
            Kg = P_pred @ np.linalg.inv(P_pred + R)
            d = d_pred + Kg @ (d_raw[t] - d_pred)
            d = np.maximum(d, 0.0)                                                # degradation is nonnegative
            Pm = (np.eye(N, dtype=np.float32) - Kg) @ P_pred
            out[t] = d
        return out

    def source(self, d_hat: np.ndarray) -> np.ndarray:
        """Flux under the operator: s_hat(t) = d_hat(t) - M d_hat(t-1)  [T, N]; zero at t=0."""
        s = np.zeros_like(d_hat)
        s[1:] = d_hat[1:] - d_hat[:-1] @ self.M.T
        return np.maximum(s, 0.0)

    def calibrate(self, nominal_eps: list[Episode], margin: float = 1.15) -> float:
        return margin * max(float(self.field(ep).max()) for ep in nominal_eps)

    def detect(self, ep: Episode, theta: float, min_run: int = 3, start: int = 0) -> int | None:
        d_hat = self.field(ep)
        run = 0
        for t in range(start, ep.T):
            run = run + 1 if d_hat[t].max() > theta else 0
            if run >= min_run:
                return t
        return None

    def accumulated_source(self, ep: Episode, t: int) -> np.ndarray:
        s = self.source(self.field(ep))
        t = min(max(t, 1), ep.T - 1)
        return s[max(1, t - self.window + 1):t + 1].sum(axis=0)

    def rank(self, ep: Episode, t: int) -> list[int]:
        return np.argsort(-self.accumulated_source(ep, t)).tolist()

    def doubt(self, ep: Episode, t: int, k: int = 1) -> float:
        """Model-doubt score: share of accumulated source energy OUTSIDE the top-k nodes. Under a
        correct operator the source is concentrated (small); under a wrong operator the field is
        non-flat everywhere and the share grows. The learned model has no analogue."""
        acc = self.accumulated_source(ep, t)
        tot = acc.sum() + 1e-9
        top = np.sort(acc)[::-1][:k].sum()
        return float(1.0 - top / tot)

    def moduli(self, ep: Episode, t: int) -> dict:
        """Read the fault's moduli off the residual: origin, onset (first step the source is
        clearly nonzero at the origin), and severity (plateau of the source there)."""
        d_hat = self.field(ep); s = self.source(d_hat)
        t = min(max(t, 1), ep.T - 1)
        acc = s[max(1, t - self.window + 1):t + 1].sum(axis=0)
        v0 = int(np.argmax(acc))
        trace = s[:t + 1, v0]
        thr = 0.25 * trace.max()
        onset = int(np.argmax(trace > thr)) if trace.max() > 0 else t
        severity = float(np.median(trace[max(onset, t - 8):t + 1])) if t > onset else float(trace[t])
        return {"origin": v0, "onset": onset, "severity": severity}
