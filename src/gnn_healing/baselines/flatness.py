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


from ..operator import (propagation_matrix, relation_matrices, build_operator,
                        design_matrix, design_matrix_grouped, stability, conditioning)
from ..twin.schema import RELATIONS
from .. import ontology


def _fit_huber(X: np.ndarray, y: np.ndarray, iters: int, delta: float) -> np.ndarray:
    """IRLS Huber fit. The (node, step) pairs carrying the source are outliers to a model that
    has no source term, so they get down-weighted automatically. That is what lets this run on
    UNLABELLED data: it needs excitation, not labels saying where the faults were."""
    k = X.shape[1]
    w = np.ones(len(y))
    theta = np.zeros(k)
    for _ in range(iters):
        Xw = X * w[:, None]
        theta, *_ = np.linalg.lstsq(Xw.T @ X + 1e-6 * np.eye(k), Xw.T @ y, rcond=None)
        res = y - X @ theta
        scale = np.median(np.abs(res)) / 0.6745 + 1e-9
        w = np.minimum(1.0, delta * scale / np.maximum(np.abs(res), 1e-9))
    return theta


def _projected_fields(twin: Twin, episodes: list[Episode]) -> list[np.ndarray]:
    E = twin.effect_matrix(); proj = E / np.maximum((E ** 2).sum(axis=1, keepdims=True), 1e-6)
    return [np.maximum(np.einsum("tnk,nk->tn", ep.X, proj), 0.0) for ep in episodes]


def identify_operator(twin: Twin, episodes: list[Episode], iters: int = 20,
                      delta: float = 0.1) -> tuple[float, float]:
    """Identify (rho, beta) from UNLABELLED episodes: regress d~(t+1) on [d~(t), P d~(t)]."""
    Xs, ys = [], []
    for d in _projected_fields(twin, episodes):
        Xs.append(design_matrix(twin, d, per_relation=False)); ys.append(d[1:].ravel())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = X[:, 0] > 0.3                      # only steps with a visible field inform M
    theta = _fit_huber(X[keep], y[keep], iters, delta)
    return float(np.clip(theta[0], 0.0, 0.999)), float(max(theta[1], 0.0))


def identify_operator_grouped(twin: Twin, episodes: list[Episode], iters: int = 20,
                              delta: float = 0.1) -> tuple[float, dict[str, float], dict]:
    """Identify (rho, {beta_r}) with the relations TIED BY THE ONTOLOGY'S PROPERTY HIERARCHY:
    one coefficient per superproperty (feeds, controls, exposes), broadcast back to the
    relations underneath it.

    Nine free coefficients are badly conditioned on a tree-shaped topology, because relations
    along a chain always co-occur and the regression cannot separate them. Three are not. This
    is the hierarchy earning its place: it is a regularisation structure, not a taxonomy."""
    sup = ontology.superproperty_by_relation()
    Xs, ys, groups = [], [], None
    for d in _projected_fields(twin, episodes):
        Xg, groups = design_matrix_grouped(twin, d)
        Xs.append(Xg); ys.append(d[1:].ravel())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = X[:, 0] > 0.3
    Xk = X[keep]
    theta = _fit_huber(Xk, y[keep], iters, delta)
    rho = float(np.clip(theta[0], 0.0, 0.999))
    by_group = {g: float(max(theta[1 + i], 0.0)) for i, g in enumerate(groups)}
    betas = {r.value: by_group.get(sup.get(r.value, ""), 0.0) for r in RELATIONS}
    return rho, betas, {"by_group": by_group, "condition_number": conditioning(Xk)}


def identify_operator_per_relation(twin: Twin, episodes: list[Episode], iters: int = 20,
                                   delta: float = 0.1) -> tuple[float, dict[str, float]]:
    """Identify (rho, {beta_r}) from UNLABELLED episodes. Same regression, one column per
    relation instead of one for the aggregate. This is the thing a single scalar cannot do:
    recover that transport couples several times more strongly than the control plane."""
    Xs, ys = [], []
    for d in _projected_fields(twin, episodes):
        Xs.append(design_matrix(twin, d, per_relation=True)); ys.append(d[1:].ravel())
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = X[:, 0] > 0.3
    Xk = X[keep]
    theta = _fit_huber(Xk, y[keep], iters, delta)
    rho = float(np.clip(theta[0], 0.0, 0.999))
    betas = {r.value: float(max(theta[1 + i], 0.0)) for i, r in enumerate(RELATIONS)}
    identify_operator_per_relation.last_condition_number = conditioning(Xk)
    return rho, betas


def steady_state_gain(M: np.ndarray, q: float, r: float, iters: int = 500,
                      tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """Converged Kalman gain for the time-invariant system d(t+1) = M d(t) + w, y = d + v.

    M, Q and R are constant, so the covariance recursion converges and the gain does not need
    recomputing every step. Iterate the Riccati recursion once at startup:

        P_pred = M P M^T + Q,   K = P_pred (P_pred + R)^-1,   P = (I - K) P_pred

    then every step collapses to two matrix-vector products,

        d  <-  A d + K y,       A = (I - K) M

    which is O(N^2) instead of the O(N^3) inverse the per-step filter pays. Measured on an Intel
    Xeon 6767P, the per-step filter costs 218 ms at 1,222 nodes and 2.8 s at 3,066; this costs
    microseconds at both, because the only cubic work happens once.

    Returns (A, K, iterations_to_converge).
    """
    N = M.shape[0]
    I = np.eye(N, dtype=np.float64)
    Q, R = q * I, r * I
    P = np.eye(N, dtype=np.float64)
    K = np.zeros_like(P)
    for i in range(1, iters + 1):
        P_pred = M @ P @ M.T + Q
        K = P_pred @ np.linalg.inv(P_pred + R)
        P_new = (I - K) @ P_pred
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    A = (I - K) @ M
    return A.astype(np.float32), K.astype(np.float32), i


class FlatnessSolver:
    def __init__(self, twin: Twin, prop: PropagationParams = PropagationParams(),
                 q: float = 0.05, r: float = 0.08, window: int = 16,
                 rho: float | None = None, beta: float | None = None,
                 beta_by_relation: dict[str, float] | None = None, from_ontology: bool = False,
                 steady_state: bool = False):
        self.N = twin.n
        rho = prop.rho if rho is None else rho
        beta = prop.beta if beta is None else beta
        if beta_by_relation is None and from_ontology:
            beta_by_relation = ontology.coupling_by_relation()
        if beta_by_relation is None and prop.beta_by_relation is not None:
            beta_by_relation = prop.beta_by_relation
        self.rho, self.beta, self.beta_by_relation = rho, beta, beta_by_relation
        self.M = build_operator(twin, rho, beta=beta, beta_by_relation=beta_by_relation)
        self.steady_state = steady_state
        self._A = self._K = None
        self.gain_iterations = 0
        if steady_state:
            self._A, self._K, self.gain_iterations = steady_state_gain(
                self.M.astype(np.float64), q, r)
        E = twin.effect_matrix()                                                  # [N, K]
        self.proj = E / np.maximum((E ** 2).sum(axis=1, keepdims=True), 1e-6)    # x -> d along e_tau
        self.q, self.r, self.window = q, r, window

    def field(self, ep: Episode) -> np.ndarray:
        """Kalman-filtered estimate of d(t) from x(t): [T, N]. Causal (no look-ahead)."""
        T, N, K = ep.X.shape
        d_raw = np.einsum("tnk,nk->tn", ep.X, self.proj)                          # projected observation
        if self.steady_state:
            d = np.zeros(N, dtype=np.float32)
            out = np.zeros((T, N), dtype=np.float32)
            for t in range(T):
                d = np.maximum(self._A @ d + self._K @ d_raw[t], 0.0)
                out[t] = d
            return out
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

    def doubt(self, ep: Episode, t: int, theta: float, k: int = 1) -> float:
        """Model-doubt score: flatness *inside the blast radius*. Restrict to nodes whose estimated
        field exceeds the detection threshold theta (the degraded set). Under a correct operator
        every degraded node except the origin is flat, so the share of residual energy off the
        top-k is near zero. Under a wrong operator the downstream nodes stop being flat and the
        share rises. Nodes outside the degraded set carry only noise and are excluded on purpose."""
        d_hat = self.field(ep); t = min(max(t, 1), ep.T - 1)
        resid = np.zeros_like(d_hat); resid[1:] = d_hat[1:] - d_hat[:-1] @ self.M.T   # unsigned: flat means ~0 either way
        acc = np.abs(resid)[max(1, t - self.window + 1):t + 1].sum(axis=0)
        degraded = d_hat[t] > theta
        if degraded.sum() <= k:
            return 0.0
        e = acc[degraded]
        top = np.sort(e)[::-1][:k].sum()
        return float(1.0 - top / (e.sum() + 1e-9))

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
