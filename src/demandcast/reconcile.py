"""Hierarchical reconciliation: item x location sums to location sums to total.

Three methods on the same base forecasts:

- bottom_up: keep the item forecasts, add them up.
- top_down: forecast the total, split it by each item-location's average share of the total over the last 28 days.
  Cheap, but a series with no recent history (a new site or item) gets a share of zero.
- mint_lite: weighted least squares reconciliation (MinT with a diagonal covariance, also called WLS-var).
  y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat, with W the in-sample residual variance of each node's base model.
  Every level's own model contributes, weighted by how noisy it is.

Rows of S are ordered [total, locations..., bottom...]. Only series that are open and on the menu on a day take
part, so the summing matrix is built per pattern of active series. Negative reconciled bottom values are clipped
to zero and the upper levels re-summed, so every output is coherent.
"""

from __future__ import annotations

import numpy as np

METHODS = ["bottom_up", "top_down", "mint_lite"]


def summing_matrix(bottom_loc: np.ndarray, n_loc: int) -> np.ndarray:
    nb = len(bottom_loc)
    S = np.zeros((1 + n_loc + nb, nb))
    S[0] = 1
    for li in range(n_loc):
        S[1 + li] = bottom_loc == li
    S[1 + n_loc :] = np.eye(nb)
    return S


def aggregate(bottom: np.ndarray, bottom_loc: np.ndarray, n_loc: int) -> tuple[np.ndarray, np.ndarray]:
    """bottom (n_b, n_days) -> (location (n_loc, n_days), total (n_days,))."""
    loc = np.stack([bottom[bottom_loc == li].sum(axis=0) for li in range(n_loc)])
    return loc, loc.sum(axis=0)


def bottom_up(bottom: np.ndarray, **_) -> np.ndarray:
    return bottom.copy()


def top_down(total: np.ndarray, shares: np.ndarray, active: np.ndarray, **_) -> np.ndarray:
    """shares (n_b,) historical proportions, renormalized over the series active each day."""
    w = shares[:, None] * active
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    return w * total[None, :]


def mint_lite(
    bottom: np.ndarray,
    loc: np.ndarray,
    total: np.ndarray,
    var_bottom: np.ndarray,
    var_loc: np.ndarray,
    var_total: float,
    bottom_loc: np.ndarray,
    active: np.ndarray,
    loc_active: np.ndarray,
    **_,
) -> np.ndarray:
    n_loc = loc.shape[0]
    out = np.zeros_like(bottom)
    patterns: dict[bytes, list[int]] = {}
    for j in range(bottom.shape[1]):
        patterns.setdefault(active[:, j].tobytes() + loc_active[:, j].tobytes(), []).append(j)
    for cols in patterns.values():
        a = active[:, cols[0]]
        la = loc_active[:, cols[0]] & np.array([a[bottom_loc == li].any() for li in range(n_loc)])
        if not a.any():
            continue
        b_idx = np.where(a)[0]
        l_idx = np.where(la)[0]
        S = summing_matrix(bottom_loc[b_idx], n_loc)
        rows = np.concatenate([[0], 1 + l_idx, 1 + n_loc + np.arange(len(b_idx))])
        S = S[rows]
        w = np.concatenate([[var_total], var_loc[l_idx], var_bottom[b_idx]])
        Winv = 1.0 / np.maximum(w, 1e-6)
        StW = S.T * Winv
        G = np.linalg.solve(StW @ S, StW)
        yhat = np.vstack([total[None, cols], loc[l_idx][:, cols], bottom[b_idx][:, cols]])
        out[np.ix_(b_idx, cols)] = G @ yhat
    return out


def reconcile(method: str, **kw) -> dict:
    """Returns coherent {'bottom', 'location', 'total'} for one method."""
    if method == "bottom_up":
        b = bottom_up(**kw)
    elif method == "top_down":
        b = top_down(**kw)
    elif method == "mint_lite":
        b = mint_lite(**kw)
    else:
        raise ValueError(method)
    b = np.maximum(b, 0) * kw["active"]
    loc, tot = aggregate(b, kw["bottom_loc"], kw["loc"].shape[0])
    return {"bottom": b, "location": loc, "total": tot}


def historical_shares(Y_bottom: np.ndarray, Y_total: np.ndarray, cutoff: int, window: int = 28) -> np.ndarray:
    """Average proportion of the total for each bottom series over the window (Gross and Sohl method A)."""
    sl = slice(max(0, cutoff - window + 1), cutoff + 1)
    tot = Y_total[sl]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = Y_bottom[:, sl] / tot[None, :]
    p = np.where(np.isnan(p), 0.0, p)
    ok = ~np.isnan(tot) & (tot > 0)
    return p[:, ok].mean(axis=1) if ok.any() else np.zeros(Y_bottom.shape[0])
