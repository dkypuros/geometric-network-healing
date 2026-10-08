"""Chapter 9: metrics.

lead_time       = t_baseline - t_gnn  (steps gained; positive is good)        (Eq. 9.1)
hit@k           = 1[origin in top-k RCA ranking]                               (Eq. 9.2)
triage_reduction= 1 - k / |degraded set at detection|  (nodes a human skips)   (Eq. 9.3)
"""
from __future__ import annotations
import numpy as np


def lead_time(t_baseline: int | None, t_gnn: int | None, T: int) -> int | None:
    """Steps gained over the baseline. None when the baseline never fired (reported separately
    as a detection-rate gap rather than folded into the mean as a censored value)."""
    if t_baseline is None:
        return None
    tg = T if t_gnn is None else t_gnn
    return t_baseline - tg


def hit_at_k(ranking: list[int], origin: int, k: int) -> float:
    return float(origin in ranking[:k])


def triage_reduction(k: int, n_degraded: int) -> float:
    return 0.0 if n_degraded <= 0 else max(0.0, 1.0 - k / n_degraded)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}
