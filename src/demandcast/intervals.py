"""Prediction intervals from conformal residuals.

The nonconformity score is a scaled residual, s = (y - yhat) / (yhat + 1) ** GAMMA. Errors on counts grow with the
level, a bit faster than Poisson noise, so GAMMA = 0.75 makes scores comparable between a 10-unit side and a
600-unit location. An interval is yhat plus the empirical score quantiles times the scale. Scores come only from
past forecasts whose target day is on or before the cutoff (online split conformal), never from the window being
scored.
"""

from __future__ import annotations

import numpy as np

GAMMA = 0.75
NOMINAL = 0.90


def scale(yhat: np.ndarray) -> np.ndarray:
    return (np.maximum(yhat, 0) + 1.0) ** GAMMA


def scores(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return (y - yhat) / scale(yhat)


def score_quantile(s: np.ndarray, p: float) -> float:
    """Empirical quantile with the conformal finite-sample adjustment toward the tails."""
    s = np.sort(np.asarray(s, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    if p >= 0.5:
        k = min(int(np.ceil((n + 1) * p)) - 1, n - 1)
    else:
        k = max(int(np.floor((n + 1) * p)) - 1, 0)
    return float(s[k])


def interval(yhat: np.ndarray, s: np.ndarray, level: float = NOMINAL) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    lo = yhat + score_quantile(s, a) * scale(yhat)
    hi = yhat + score_quantile(s, 1 - a) * scale(yhat)
    return np.maximum(lo, 0), np.maximum(hi, 0)


def quantile(yhat: np.ndarray, s: np.ndarray, p: float) -> np.ndarray:
    return np.maximum(yhat + score_quantile(s, p) * scale(yhat), 0)


def samples(yhat: float, s: np.ndarray) -> np.ndarray:
    """Demand scenarios for one forecast: one per calibration score."""
    return np.maximum(yhat + np.asarray(s) * scale(np.asarray(yhat)), 0)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))
