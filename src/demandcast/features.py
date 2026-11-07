"""Panels and leakage-safe features.

A panel is a set of daily series (item x location, location, or total) with known-in-advance covariates. Every
feature matrix is built from a copy of the target array with everything after the cutoff set to NaN, and every lag
is at least `shift` days old, so a model trained or scored at a cutoff can never see the future. The leakage test
perturbs the data after the cutoff and checks the forecasts do not move.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .catalog import CLASSES, ITEMS, LOCATIONS
from .generator import SITE_KINDS, Dataset

HOLIDAYS = [
    "",
    "New Year's Day",
    "MLK Day",
    "Presidents Day",
    "Memorial Day",
    "Independence Day",
    "Labor Day",
    "Thanksgiving",
    "Black Friday",
    "Christmas Eve",
    "Christmas",
    "New Year's Eve",
]
MIN_HISTORY = 36  # rows need this many days before them for the 28-day rolling mean at the longest shift


@dataclass
class Panel:
    level: str  # item, location, total
    names: list[str]  # series names
    loc_index: np.ndarray  # (n_series,) location index, -1 for total
    item_index: np.ndarray  # (n_series,) item index, -1 above item level
    Y: np.ndarray  # (n_series, n_all) float, NaN where unknown, closed, or not on the menu
    avail: np.ndarray  # (n_series, n_all) bool, open and on the menu (known in advance)
    static: dict  # name -> (n_series,) int
    exog: dict  # name -> (n_series, n_all) float
    categorical: tuple = ("loc", "item", "cls", "site", "holiday")


def _calendar(ds: Dataset) -> dict:
    d = ds.dates
    doy = d.dayofyear.values
    return {
        "dow": d.day_of_week.values.astype(float),
        "month": d.month.values.astype(float),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "holiday": np.array([HOLIDAYS.index(h) for h in ds.holiday], dtype=float),
        "temp": ds.temp.astype(float),
        "precip": ds.precip.astype(float),
        "snow": ds.snow.astype(float),
    }


def _days_since(mask: np.ndarray, cap: int) -> np.ndarray:
    """Days since the first True along the last axis, capped; -1 before."""
    first = np.where(mask.any(axis=-1), mask.argmax(axis=-1), mask.shape[-1])
    t = np.arange(mask.shape[-1])
    out = t - first[..., None]
    return np.where(out >= 0, np.minimum(out, cap), -1).astype(float)


def item_panel(ds: Dataset) -> Panel:
    n_loc, n_items, n_all = len(LOCATIONS), len(ITEMS), len(ds.dates)
    cal = _calendar(ds)
    daily = ds.daily.astype(float)
    avail = ds.open_[:, None, :] & ds.on_menu  # (loc, item, day)
    Y = np.full((n_loc, n_items, n_all), np.nan)
    Y[:, :, : ds.n_hist] = daily
    Y[~avail] = np.nan
    days_open = _days_since(ds.open_, 120)
    days_menu = _days_since(ds.on_menu, 60)
    ex = {k: np.broadcast_to(v, (n_loc, n_items, n_all)) for k, v in cal.items()}
    ex["pressure"] = np.broadcast_to(ds.pressure[:, None, :], (n_loc, n_items, n_all))
    ex["event_start"] = np.broadcast_to(ds.event_start[:, None, :], (n_loc, n_items, n_all))
    ex["site"] = np.broadcast_to(ds.site[:, None, :].astype(float), (n_loc, n_items, n_all))
    ex["promo"] = np.broadcast_to(ds.promo[None].astype(float), (n_loc, n_items, n_all))
    ex["days_open"] = np.broadcast_to(days_open[:, None, :], (n_loc, n_items, n_all))
    ex["days_on_menu"] = days_menu
    flat = lambda a: np.ascontiguousarray(a).reshape(n_loc * n_items, n_all)  # noqa: E731
    li = np.repeat(np.arange(n_loc), n_items)
    ki = np.tile(np.arange(n_items), n_loc)
    return Panel(
        level="item",
        names=[f"{LOCATIONS[a].id}/{ITEMS[b].id}" for a, b in zip(li, ki, strict=True)],
        loc_index=li,
        item_index=ki,
        Y=flat(Y),
        avail=flat(avail),
        static={"loc": li, "item": ki, "cls": np.array([CLASSES.index(ITEMS[k].cls) for k in ki])},
        exog={k: flat(v).astype(float) for k, v in ex.items()},
    )


def location_panel(ds: Dataset) -> Panel:
    n_loc, n_all = len(LOCATIONS), len(ds.dates)
    cal = _calendar(ds)
    Y = np.full((n_loc, n_all), np.nan)
    Y[:, : ds.n_hist] = ds.daily.sum(axis=1)
    Y[~ds.open_] = np.nan
    ex = {k: np.broadcast_to(v, (n_loc, n_all)).astype(float) for k, v in cal.items()}
    ex["pressure"] = ds.pressure.astype(float)
    ex["event_start"] = ds.event_start.astype(float)
    ex["site"] = ds.site.astype(float)
    ex["promo"] = np.broadcast_to(ds.promo.sum(axis=0), (n_loc, n_all)).astype(float)
    ex["days_open"] = _days_since(ds.open_, 120)
    ex["new_items"] = np.broadcast_to(_days_since(ds.on_menu, 60).max(axis=1), (n_loc, n_all)).astype(float)
    li = np.arange(n_loc)
    return Panel(
        level="location",
        names=[loc.id for loc in LOCATIONS],
        loc_index=li,
        item_index=np.full(n_loc, -1),
        Y=Y,
        avail=ds.open_.copy(),
        static={"loc": li},
        exog=ex,
    )


def total_panel(ds: Dataset) -> Panel:
    n_all = len(ds.dates)
    cal = _calendar(ds)
    Y = np.full((1, n_all), np.nan)
    Y[0, : ds.n_hist] = ds.daily.sum(axis=(0, 1))
    any_open = ds.open_.any(axis=0)
    Y[0, ~any_open] = np.nan
    brick = [i for i, loc in enumerate(LOCATIONS) if not loc.mobile]
    truck = [i for i, loc in enumerate(LOCATIONS) if loc.mobile][0]
    ex = {k: v[None].astype(float) for k, v in cal.items()}
    ex["pressure"] = ds.pressure[brick].sum(axis=0)[None]
    ex["truck_pressure"] = ds.pressure[truck][None]
    ex["site"] = ds.site[truck][None].astype(float)
    ex["n_open"] = ds.open_.sum(axis=0)[None].astype(float)
    ex["promo"] = ds.promo.sum(axis=0)[None].astype(float)
    ex["days_open"] = _days_since(ds.open_, 120).min(axis=0)[None]
    return Panel(
        level="total",
        names=["total"],
        loc_index=np.array([-1]),
        item_index=np.array([-1]),
        Y=Y,
        avail=any_open[None],
        static={},
        exog=ex,
    )


def _roll_mean(Y: np.ndarray, w: int) -> np.ndarray:
    """Mean of non-NaN values over [t - w + 1, t] for every t, NaN when the window is empty."""
    v = np.nan_to_num(Y)
    c = (~np.isnan(Y)).astype(float)
    cs = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(v, axis=1)], axis=1)
    cc = np.concatenate([np.zeros((Y.shape[0], 1)), np.cumsum(c, axis=1)], axis=1)
    idx = np.arange(Y.shape[1])
    lo = np.maximum(idx - w + 1, 0)
    s = cs[:, idx + 1] - cs[:, lo]
    n = cc[:, idx + 1] - cc[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _take(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A[:, t] with NaN for negative t."""
    out = A[:, np.clip(t, 0, None)]
    out[:, t < 0] = np.nan
    return out


def lag_features(Ym: np.ndarray, days: np.ndarray, shift: int) -> dict:
    """Lags for target days, using only values at least `shift` days before each target."""
    wk = 7 * int(np.ceil(shift / 7))
    r7 = _roll_mean(Ym, 7)
    r28 = _roll_mean(Ym, 28)
    dow = np.nanmean(np.stack([_take(Ym, days - wk - 7 * j) for j in range(4)]), axis=0)
    level = _take(r28, days - shift)
    # Fall back to the mean of whatever is observed up to each target's info date (new sites and items).
    r_all = _roll_mean(Ym, 10_000)
    level = np.where(np.isnan(level), _take(r_all, days - shift), level)
    raw = {
        "lag_recent": _take(Ym, days - shift),
        "lag_week": _take(Ym, days - wk),
        "lag_week2": _take(Ym, days - wk - 7),
        "roll7": _take(r7, days - shift),
        "dow_mean4": dow,
    }
    # Lags as ratios to the level, so trees see shapes, not sizes, and a new site or growth does not leave the
    # training range. The model predicts y / level too.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {k: v / np.maximum(level, 1.0) for k, v in raw.items()}
    out["level"] = level
    return out


def masked(panel: Panel, cutoff: int) -> np.ndarray:
    Ym = panel.Y.copy()
    Ym[:, cutoff + 1 :] = np.nan
    return Ym


def design(panel: Panel, days: np.ndarray, shift: int, cutoff: int, overrides: dict | None = None):
    """Feature matrix for every series on the given days. Rows are series-major: (series, day).

    Returns (X, columns, categorical mask). `overrides` replaces exogenous columns, for example zeroing the event
    pressure to get a no-event counterfactual.
    """
    Ym = masked(panel, cutoff)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lags = lag_features(Ym, days, shift)
    n = panel.Y.shape[0]
    cols, mats = [], []
    for k, v in panel.static.items():
        cols.append(k)
        mats.append(np.repeat(v[:, None], len(days), axis=1).astype(float))
    for k, v in panel.exog.items():
        cols.append(k)
        vv = v[:, days]
        if overrides and k in overrides:
            vv = np.broadcast_to(overrides[k], vv.shape)
        mats.append(vv)
    for k, v in lags.items():
        cols.append(k)
        mats.append(v)
    X = np.stack([m.reshape(n * len(days)) for m in mats], axis=1)
    cat = np.array([c in panel.categorical for c in cols])
    return X, cols, cat


def training_rows(panel: Panel, shift: int, cutoff: int, min_day: int = MIN_HISTORY):
    """Training design for a model with this shift at this cutoff: every observed day up to the cutoff."""
    days = np.arange(min_day + shift, cutoff + 1)
    X, cols, cat = design(panel, days, shift, cutoff)
    y = panel.Y[:, days].reshape(-1)
    level = X[:, cols.index("level")]
    keep = ~np.isnan(y) & ~np.isnan(level)
    return X[keep], y[keep], cols, cat


SITE_INDEX = {k: i for i, k in enumerate(SITE_KINDS)}
