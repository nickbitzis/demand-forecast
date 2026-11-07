"""Forecasters: seasonal naive, moving average, gradient boosting per panel, and cold start from similar entities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .catalog import CLASSES, ITEMS, LOCATIONS
from .features import Panel, design, masked, training_rows

COLD_LOCATION_DAYS = 28  # use the analog location until the new one has this many days of sales
COLD_ITEM_DAYS = 21  # use the class share prior until the new item has this many days of sales
PSEUDO_DAYS = 7.0
THREADS = (
    4  # small data: more OpenMP threads only add overhead  # weight of the prior in the shrunk ratio, in days of data
)


@dataclass
class GBMConfig:
    max_iter: int = 150
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    seed: int = 0
    normalize: bool = False  # True: model y / level (28-day mean at the info date). Backtested slightly worse.


def seasonal_naive(panel: Panel, cutoff: int, days: np.ndarray) -> np.ndarray:
    """Same weekday in the latest week at or before the cutoff. Falls back a week, then to the 7-day mean."""
    Ym = masked(panel, cutoff)
    out = np.full((Ym.shape[0], len(days)), np.nan)
    for j, t in enumerate(days):
        wk = 7 * int(np.ceil((t - cutoff) / 7))
        for back in (wk, wk + 7, wk + 14):
            v = Ym[:, t - back] if t - back >= 0 else np.full(Ym.shape[0], np.nan)
            out[:, j] = np.where(np.isnan(out[:, j]), v, out[:, j])
    fb = moving_average(panel, cutoff, days)
    out = np.where(np.isnan(out), fb, out)
    return np.where(panel.avail[:, days], np.nan_to_num(out), 0.0)


def moving_average(panel: Panel, cutoff: int, days: np.ndarray, window: int = 7) -> np.ndarray:
    Ym = masked(panel, cutoff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(Ym[:, max(0, cutoff - window + 1) : cutoff + 1], axis=1)
    out = np.repeat(np.nan_to_num(m)[:, None], len(days), axis=1)
    return np.where(panel.avail[:, days], out, 0.0)


class GBM:
    """One HistGradientBoosting model (Poisson loss) for a panel, a shift, and a cutoff."""

    def __init__(self, panel: Panel, shift: int, cutoff: int, cfg: GBMConfig | None = None):
        self.panel, self.shift, self.cutoff = panel, shift, cutoff
        self.cfg = cfg or GBMConfig()
        X, y, self.cols, cat = training_rows(panel, shift, cutoff)
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=self.cfg.max_iter,
            learning_rate=self.cfg.learning_rate,
            max_leaf_nodes=self.cfg.max_leaf_nodes,
            min_samples_leaf=self.cfg.min_samples_leaf,
            l2_regularization=self.cfg.l2_regularization,
            categorical_features=cat,
            random_state=self.cfg.seed,
        )
        self.li = self.cols.index("level")
        with threadpool_limits(THREADS):
            if self.cfg.normalize:
                lvl = np.maximum(X[:, self.li], 1.0)
                self.model.fit(X, y / lvl, sample_weight=lvl)
            else:  # no weights: weighted binning is far slower in scikit-learn
                self.model.fit(X, y)
        self.n_train = len(y)
        self.resid_var = self._resid_var()

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        lvl = X[:, self.li]
        with threadpool_limits(THREADS):
            r = self.model.predict(X)
        if not self.cfg.normalize:
            return r
        return np.where(np.isnan(lvl), 0.0, r * np.maximum(np.nan_to_num(lvl), 1.0))

    def _resid_var(self) -> np.ndarray:
        """In-sample residual variance per series over the last 90 training days, for MinT-lite weights."""
        n = self.panel.Y.shape[0]
        days = np.arange(max(self.cutoff - 89, 0), self.cutoff + 1)
        Xd, _, _ = design(self.panel, days, self.shift, self.cutoff)
        yd = self.panel.Y[:, days]
        pd_ = self._raw_predict(Xd).reshape(n, len(days))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean((yd - pd_) ** 2, axis=1)
        return np.where(np.isnan(v), np.nanmax(v) if np.isfinite(v).any() else 1.0, v)

    def predict(self, days: np.ndarray, cutoff: int | None = None, overrides: dict | None = None) -> np.ndarray:
        """(n_series, n_days). Features use data up to `cutoff` (default: the training cutoff). Zero when closed."""
        c = self.cutoff if cutoff is None else cutoff
        assert np.all(days - c <= self.shift), "target is further ahead than this model's shift"
        X, _, _ = design(self.panel, days, self.shift, c, overrides)
        p = self._raw_predict(X).reshape(self.panel.Y.shape[0], len(days))
        return np.where(self.panel.avail[:, days], np.maximum(p, 0), 0.0)


def cold_threshold(limit: int, days: np.ndarray, cutoff: int) -> int:
    """Days of history a series needs before the GBM takes over: its lags must cover a week past the horizon."""
    return min(limit, int(days.max() - cutoff) + 7)


def observed_days(panel: Panel, series: int, cutoff: int) -> int:
    return int((~np.isnan(panel.Y[series, : cutoff + 1])).sum())


def cold_start_location(
    item_fc: np.ndarray, loc_fc: np.ndarray, items: Panel, locs: Panel, cutoff: int, days: np.ndarray
) -> list[dict]:
    """Replace forecasts of a location with too little history by its analog's forecast times a shrunk ratio.

    ratio = (sum of new-site sales + prior * K * analog daily mean) / (analog sales on the same days + K * mean).
    With no sales yet this is the configured prior. Edits the arrays in place and returns what it did.
    """
    notes = []
    for li, loc in enumerate(LOCATIONS):
        if loc.analog is None:
            continue
        n_obs = observed_days(locs, li, cutoff)
        if n_obs >= cold_threshold(COLD_LOCATION_DAYS, days, cutoff):
            continue
        ai = [i for i, x in enumerate(LOCATIONS) if x.id == loc.analog][0]
        y_new, y_an = locs.Y[li, : cutoff + 1], locs.Y[ai, : cutoff + 1]
        both = ~np.isnan(y_new) & ~np.isnan(y_an)
        an_mean = float(np.nanmean(y_an[-28:])) if np.isfinite(y_an[-28:]).any() else 1.0
        ratio = (y_new[both].sum() + loc.prior_ratio * PSEUDO_DAYS * an_mean) / (
            y_an[both].sum() + PSEUDO_DAYS * an_mean
        )
        av = locs.avail[li, days]
        loc_fc[li] = np.where(av, loc_fc[ai] * ratio, 0.0)
        rows = np.where(items.loc_index == li)[0]
        for r in rows:
            k = items.item_index[r]
            ra = np.where((items.loc_index == ai) & (items.item_index == k))[0][0]
            item_fc[r] = np.where(items.avail[r, days], item_fc[ra] * ratio, 0.0)
        notes.append({"location": loc.id, "analog": loc.analog, "days_observed": n_obs, "ratio": round(ratio, 3)})
    return notes


def cold_start_item(item_fc: np.ndarray, items: Panel, cutoff: int, days: np.ndarray) -> list[dict]:
    """New menu item with little history: a shrunk share of its class's forecast at the same location."""
    notes = []
    for k, it in enumerate(ITEMS):
        if it.launch_days_before_end is None:
            continue
        cls_items = [j for j, x in enumerate(ITEMS) if x.cls == it.cls and j != k]
        for li in range(len(LOCATIONS)):
            r = np.where((items.loc_index == li) & (items.item_index == k))[0][0]
            n_obs = observed_days(items, r, cutoff)
            if n_obs >= cold_threshold(COLD_ITEM_DAYS, days, cutoff) or not items.avail[r, days].any():
                continue
            peers = np.where((items.loc_index == li) & np.isin(items.item_index, cls_items))[0]
            y_new = items.Y[r, : cutoff + 1]
            y_cls = np.nansum(items.Y[peers, : cutoff + 1], axis=0)
            seen = ~np.isnan(y_new)
            base = float(np.mean(y_cls[-28:])) if len(y_cls) else 1.0
            den = y_cls[seen].sum() + PSEUDO_DAYS * base
            share = (y_new[seen].sum() + it.analog_share * PSEUDO_DAYS * base) / den if den > 0 else it.analog_share
            item_fc[r] = np.where(items.avail[r, days], item_fc[peers].sum(axis=0) * share, 0.0)
            notes.append(
                {"item": it.id, "location": LOCATIONS[li].id, "days_observed": n_obs, "share": round(share, 3)}
            )
    return notes


def class_index(items: Panel) -> np.ndarray:
    return np.array([CLASSES.index(ITEMS[k].cls) for k in items.item_index])
