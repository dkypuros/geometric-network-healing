"""The propagation operator, in its single-coupling and per-relation forms.

Single coupling (Chapter 4, what the published results use):

    M  =  rho I  +  beta P,            P = D_in^-1 A^T

Per-relation coupling (this module's reason to exist):

    M  =  rho I  +  sum_r beta_r P_r,  P_r = D_in^-1 A_r^T

Both use the SAME global in-degree normalisation, so sum_r P_r = P exactly and setting every
beta_r to one value recovers the single-coupling operator to floating-point equality. That is
asserted in tests, and it is why turning this on cannot silently move the published numbers.

beta_r comes from `docs/ontology/network-healing.ttl` by way of `gnn_healing.ontology`.
A fibre cut and a signalling storm do not propagate with the same strength, and a single
scalar asked to average them is a misspecification; measuring that is the point.

Stability: the row sums of sum_r beta_r P_r are convex combinations of the beta_r, so
    max_r beta_r / (1 - rho) < 1
is sufficient for faults to attenuate per hop.
"""
from __future__ import annotations
import numpy as np
from .twin.generator import Twin
from .twin.schema import Relation, RELATIONS
from . import ontology


def in_degree(twin: Twin) -> np.ndarray:
    A = twin.adjacency()
    return np.maximum(A.sum(axis=0), 1.0)


def propagation_matrix(twin: Twin) -> np.ndarray:
    """P = D_in^-1 A^T, the aggregate propagation matrix."""
    return (twin.adjacency() / in_degree(twin)).T


def relation_matrices(twin: Twin) -> dict[str, np.ndarray]:
    """P_r for every relation present, under the GLOBAL in-degree normalisation
    so that sum_r P_r == P exactly."""
    N = twin.n
    indeg = in_degree(twin)
    src = twin.edges["src"].to_numpy()
    dst = twin.edges["dst"].to_numpy()
    rel = twin.edges["rel"].to_numpy()
    out: dict[str, np.ndarray] = {}
    for r in RELATIONS:
        m = rel == r.value
        A_r = np.zeros((N, N), dtype=np.float32)
        if m.any():
            A_r[src[m], dst[m]] = 1.0
        out[r.value] = (A_r / indeg).T
    return out


def build_operator(twin: Twin, rho: float, beta: float | None = None,
                   beta_by_relation: dict[str, float] | None = None,
                   from_ontology: bool = False) -> np.ndarray:
    """M. Exactly one coupling source: a scalar beta, an explicit per-relation dict,
    or the ontology's values."""
    N = twin.n
    if from_ontology and beta_by_relation is None:
        beta_by_relation = ontology.coupling_by_relation()
    if beta_by_relation is None:
        if beta is None:
            raise ValueError("give beta, beta_by_relation, or from_ontology=True")
        return (rho * np.eye(N, dtype=np.float32) + beta * propagation_matrix(twin)).astype(np.float32)
    Pr = relation_matrices(twin)
    M = rho * np.eye(N, dtype=np.float32)
    for r, P_r in Pr.items():
        b = beta_by_relation.get(r)
        if b:
            M += np.float32(b) * P_r
    return M.astype(np.float32)


def stability(rho: float, beta: float | None = None,
              beta_by_relation: dict[str, float] | None = None) -> float:
    """max_r beta_r / (1 - rho). Below 1 means faults attenuate per hop."""
    if beta_by_relation:
        return max(beta_by_relation.values()) / (1.0 - rho)
    return (beta or 0.0) / (1.0 - rho)


def design_matrix(twin: Twin, d: np.ndarray, per_relation: bool) -> np.ndarray:
    """Regressors for identification, given a projected field history d of shape [T, N].

    single:        columns [ d(t), P d(t) ]                       -> (rho, beta)
    per-relation:  columns [ d(t), P_1 d(t), ..., P_R d(t) ]      -> (rho, beta_1..beta_R)

    Returns [ (T-1)*N , 1+R ] stacked over nodes and steps.
    """
    prev = d[:-1]
    cols = [prev.ravel()]
    if per_relation:
        for r in RELATIONS:
            cols.append((prev @ relation_matrices(twin)[r.value].T).ravel())
    else:
        cols.append((prev @ propagation_matrix(twin).T).ravel())
    return np.stack(cols, axis=1)


def group_matrices(twin: Twin) -> dict[str, np.ndarray]:
    """P_g summed over the relations in each superproperty group of the ontology
    (feeds / controls / exposes). sum_g P_g == P, same as the per-relation split,
    but with three columns instead of nine."""
    sup = ontology.superproperty_by_relation()
    Pr = relation_matrices(twin)
    out: dict[str, np.ndarray] = {}
    for rel, P_r in Pr.items():
        g = sup.get(rel, "propagatesTo")
        out[g] = out.get(g, np.zeros_like(P_r)) + P_r
    return out


def design_matrix_grouped(twin: Twin, d: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Regressors [ d(t), P_feeds d(t), P_controls d(t), P_exposes d(t) ]."""
    prev = d[:-1]
    groups = sorted(group_matrices(twin))
    cols = [prev.ravel()] + [(prev @ group_matrices(twin)[g].T).ravel() for g in groups]
    return np.stack(cols, axis=1), groups


def conditioning(X: np.ndarray) -> float:
    """Condition number of the regressor Gram matrix. Large means the columns are
    collinear and the individual coefficients are not separately identifiable."""
    G = X.T @ X
    return float(np.linalg.cond(G))
