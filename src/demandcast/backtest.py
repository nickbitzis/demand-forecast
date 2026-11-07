"""Rolling-origin backtest.

At each origin (a cutoff day), every model is fit on data up to the origin only and forecasts the next H days for
H in (1, 7, 28). The GBM for horizon H uses lags at least H days old, so a 28-day forecast never uses a value it
would not have had 28 days earlier. Origins step back from the end of history by `step` days.

Metrics, on every open series-day:
- WAPE = sum |y - yhat| / sum y
- MASE = mean of |y - yhat| / s, where s is the series' mean absolute weekly change (|y_t - y_t-7|) before the origin
- bias = sum (yhat - y) / sum y
- coverage = share of actuals inside the 90% interval. Intervals use conformal scores from earlier origins whose
  target days are on or before the current origin, so the first `warmup` origins only calibrate.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import intervals
from .catalog import ITEMS, LOCATIONS, N_SLOTS
from .forecast import LEVELS, Panels, base_forecasts, fit, intraday_profiles, reconcile_all, split_intraday
from .generator import Dataset
from .models import GBMConfig, cold_start_item, cold_start_location, moving_average, seasonal_naive
from .reconcile import METHODS

HORIZONS = (1, 7, 28)
BASELINES = ["seasonal_naive", "moving_average"]
EVENT_PRESSURE = 0.5  # a location-day with at least this much nearby attendance (10,000s, distance-weighted)


@dataclass
class BacktestConfig:
    origins: int = 12
    step: int = 14
    horizons: tuple = HORIZONS
    warmup: int = 2
    gbm: GBMConfig = field(default_factory=GBMConfig)


def origin_days(ds: Dataset, cfg: BacktestConfig) -> list[int]:
    last = ds.n_hist - 1 - max(cfg.horizons)
    return sorted(last - cfg.step * k for k in range(cfg.origins))


def _mase_scale(Y: np.ndarray, cutoff: int) -> np.ndarray:
    sl = Y[:, max(7, cutoff - 364) : cutoff + 1]
    prev = Y[:, max(0, cutoff - 371) : cutoff - 6][:, : sl.shape[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanmean(np.abs(sl - prev), axis=1)
    return np.where(np.isfinite(s) & (s > 0), s, np.nan)


def _rows(origin, H, model, level, panel, days, yhat, scale, ds) -> dict:
    n = panel.Y.shape[0]
    y = panel.Y[:, days]
    li = np.repeat(panel.loc_index[:, None], len(days), axis=1)
    pressure = np.where(
        li >= 0, ds.pressure[np.clip(li, 0, None), np.broadcast_to(days, (n, len(days)))], ds.pressure[:, days].max(0)
    )
    return {
        "origin": np.full(y.size, origin),
        "H": np.full(y.size, H),
        "model": np.full(y.size, model),
        "level": np.full(y.size, level),
        "series": np.repeat(np.arange(n), len(days)),
        "loc": li.reshape(-1),
        "item": np.repeat(panel.item_index[:, None], len(days), axis=1).reshape(-1),
        "day": np.tile(days, n),
        "y": y.reshape(-1),
        "yhat": yhat.reshape(-1),
        "scale": np.repeat(scale[:, None], len(days), axis=1).reshape(-1),
        "event": (pressure >= EVENT_PRESSURE).reshape(-1),
    }


def run(ds: Dataset, cfg: BacktestConfig | None = None, log=None) -> dict:
    cfg = cfg or BacktestConfig()
    P = Panels.build(ds)
    origins = origin_days(ds, cfg)
    chunks, hourly, cold_notes = [], [], []
    t0 = time.perf_counter()
    for o in origins:
        for H in cfg.horizons:
            days = np.arange(o + 1, o + 1 + H)
            models = fit(P, o, H, cfg.gbm)
            raw = base_forecasts(models, P, days, cold_start=False)
            base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()}
            cold_notes += cold_start_item(base["bottom"], P.bottom, o, days)
            cold_notes += cold_start_location(base["bottom"], base["location"], P.bottom, P.location, o, days)
            rec = reconcile_all(base, models, P, o, days, METHODS)
            for lvl in LEVELS:
                pan = P[lvl]
                sc = _mase_scale(pan.Y, o)
                for name, fn in (("seasonal_naive", seasonal_naive), ("moving_average", moving_average)):
                    chunks.append(_rows(o, H, name, lvl, pan, days, fn(pan, o, days), sc, ds))
                for m in METHODS:
                    f = rec[m][lvl]
                    chunks.append(_rows(o, H, f"gbm_{m}", lvl, pan, days, np.atleast_2d(f), sc, ds))
                if lvl == "bottom":
                    bu = raw["bottom"]
                    chunks.append(_rows(o, H, "gbm_no_cold_start", lvl, pan, days, bu, sc, ds))
            if H == 1:
                prof = intraday_profiles(ds, o)
                no_ev = models["location"].predict(days, o, overrides={"pressure": 0.0, "event_start": 0.0})
                d = days[0]
                for li in range(len(LOCATIONS)):
                    actual = ds.units[li, :, d].sum(axis=0).astype(float)
                    if not ds.open_[li, d]:
                        continue
                    f_rec = {m: rec[m]["location"][li, 0] for m in METHODS}
                    ratio = no_ev[li, 0] / max(raw["location"][li, 0], 1e-9)
                    for m, fv in f_rec.items():
                        slots = split_intraday(ds, prof, li, d, fv, fv * ratio)
                        for h in range(N_SLOTS // 4):
                            sl = slice(4 * h, 4 * h + 4)
                            hourly.append((o, m, li, h, actual[sl].sum(), slots[sl].sum()))
        if log:
            log(f"origin {ds.dates[o].date()} done, {time.perf_counter() - t0:.0f}s")
    df = pd.DataFrame({k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]})
    df = df[~np.isnan(df["y"])].reset_index(drop=True)
    hdf = pd.DataFrame(hourly, columns=["origin", "model", "loc", "hour", "y", "yhat"])
    return summarize(ds, df, hdf, origins, cfg, cold_notes, time.perf_counter() - t0)


def add_intervals(df: pd.DataFrame, origins: list[int], warmup: int) -> pd.DataFrame:
    """Online conformal 90% intervals for GBM rows, per model, level, and horizon."""
    df = df.copy()
    df["lo"] = np.nan
    df["hi"] = np.nan
    df["score"] = intervals.scores(df["y"].values, df["yhat"].values)
    g = df[df["model"].str.startswith("gbm")]
    for (_m, _lvl, _H), part in g.groupby(["model", "level", "H"]):
        for k, o in enumerate(origins):
            if k < warmup:
                continue
            cal = part[(part["origin"] < o) & (part["day"] <= o)]["score"].values
            if len(cal) < 30:
                continue
            idx = part.index[part["origin"] == o]
            lo, hi = intervals.interval(df.loc[idx, "yhat"].values, cal)
            df.loc[idx, "lo"] = lo
            df.loc[idx, "hi"] = hi
    return df


def metrics(part: pd.DataFrame) -> dict:
    y, f = part["y"].values, part["yhat"].values
    e = np.abs(y - f)
    sy = y.sum()
    sc = part["scale"].values
    ok = np.isfinite(sc)
    out = {
        "wape": float(e.sum() / sy) if sy > 0 else None,
        "mase": float(np.mean(e[ok] / sc[ok])) if ok.any() else None,
        "bias": float((f - y).sum() / sy) if sy > 0 else None,
        "n": int(len(part)),
    }
    if "lo" in part and part["lo"].notna().any():
        c = part[part["lo"].notna()]
        out["coverage"] = intervals.coverage(c["y"].values, c["lo"].values, c["hi"].values)
    return out


def _r(v, d=4):
    return None if v is None else round(v, d)


def summarize(ds, df, hdf, origins, cfg, cold_notes, seconds) -> dict:
    df = add_intervals(df, origins, cfg.warmup)
    models = BASELINES + [f"gbm_{m}" for m in METHODS]
    # Pick the reconciliation: lowest mean WAPE over the three levels at the 7-day horizon (or the middle one).
    Hsel = 7 if 7 in cfg.horizons else cfg.horizons[len(cfg.horizons) // 2]
    rec_table = []
    for H in cfg.horizons:
        for m in METHODS:
            row = {"H": H, "method": m}
            for lvl in LEVELS:
                part = df[(df["model"] == f"gbm_{m}") & (df["level"] == lvl) & (df["H"] == H)]
                row[lvl] = _r(metrics(part)["wape"])
            row["mean"] = _r(float(np.mean([row[lvl] for lvl in LEVELS])))
            rec_table.append(row)
    sel = [r for r in rec_table if r["H"] == Hsel]
    chosen = min(sel, key=lambda r: r["mean"])["method"]
    best = f"gbm_{chosen}"

    main = []
    for lvl in LEVELS:
        for H in cfg.horizons:
            sn = metrics(df[(df["model"] == "seasonal_naive") & (df["level"] == lvl) & (df["H"] == H)])
            for m in models:
                r = metrics(df[(df["model"] == m) & (df["level"] == lvl) & (df["H"] == H)])
                main.append(
                    {
                        "level": lvl,
                        "H": H,
                        "model": m,
                        "wape": _r(r["wape"]),
                        "mase": _r(r["mase"], 3),
                        "bias": _r(r["bias"]),
                        "coverage": _r(r.get("coverage"), 3),
                        "improvement_vs_snaive": _r(1 - r["wape"] / sn["wape"]) if r["wape"] is not None else None,
                        "n": r["n"],
                    }
                )

    def compare(mask, label_fn, groups):
        out = []
        for g in groups:
            for H in cfg.horizons:
                row = {**label_fn(g), "H": H}
                for m in ("seasonal_naive", best):
                    part = df[mask(g) & (df["model"] == m) & (df["H"] == H)]
                    row["snaive" if m == "seasonal_naive" else "gbm"] = _r(metrics(part)["wape"]) if len(part) else None
                    if m == best and len(part):
                        row["coverage"] = _r(metrics(part).get("coverage"), 3)
                out.append(row)
        return out

    bottom = df["level"] == "bottom"
    by_loc = compare(
        lambda li: bottom & (df["loc"] == li), lambda li: {"location": LOCATIONS[li].id}, range(len(LOCATIONS))
    )
    classes = ["bowl", "wrap", "side", "drink"]
    cls_of = np.array([ITEMS[k].cls if k >= 0 else "" for k in range(len(ITEMS))])
    item_cls = np.where(df["item"].values >= 0, cls_of[np.clip(df["item"].values, 0, None)], "")
    by_cls = compare(lambda c: bottom & (item_cls == c), lambda c: {"class": c}, classes)
    locl = df["level"] == "location"
    by_event = compare(
        lambda e: locl & (df["event"] == e), lambda e: {"day_type": "event day" if e else "normal day"}, [True, False]
    )
    by_event_item = compare(
        lambda e: bottom & (df["event"] == e),
        lambda e: {"day_type": "event day" if e else "normal day"},
        [True, False],
    )

    # Cold start segments: Highlands in its first 8 weeks, the new bowl in its first 4 weeks.
    cold = []
    hl = [i for i, x in enumerate(LOCATIONS) if x.analog][0]
    open_day = int(np.argmax(ds.open_[hl])) if ds.open_[hl].any() else ds.n_hist
    newk = [k for k, it in enumerate(ITEMS) if it.launch_days_before_end is not None][0]
    launch = ds.n_hist - ITEMS[newk].launch_days_before_end
    segs = [
        ("Highlands, first 8 weeks", bottom & (df["loc"] == hl) & (df["day"] < open_day + 56)),
        ("New bowl, first 4 weeks", bottom & (df["item"] == newk) & (df["day"] < launch + 28)),
    ]
    for name, mask in segs:
        for H in cfg.horizons:
            row = {"segment": name, "H": H}
            for m, key in (
                ("seasonal_naive", "snaive"),
                ("gbm_no_cold_start", "gbm_no_cold_start"),
                ("gbm_bottom_up", "gbm_cold_start"),
            ):
                part = df[mask & (df["model"] == m) & (df["H"] == H)]
                row[key] = _r(metrics(part)["wape"]) if len(part) else None
                row["n"] = int(len(part))
            cold.append(row)

    # Hourly scores for the chosen method (H=1, location level) and 90% coverage of the hourly band.
    h = hdf[hdf["model"] == chosen]
    hs = intervals.scores(h["y"].values, h["yhat"].values)
    hourly_cov = None
    if len(origins) > cfg.warmup:
        cal = hs[h["origin"].values < origins[cfg.warmup]]
        test = h["origin"].values >= origins[cfg.warmup]
        if len(cal) > 30 and test.any():
            lo, hi = intervals.interval(h["yhat"].values[test], cal)
            hourly_cov = intervals.coverage(h["y"].values[test], lo, hi)

    # Calibration scores the live forecast uses: the chosen method, all origins, per level and horizon.
    sc = {}
    for lvl in LEVELS:
        for H in cfg.horizons:
            s = df[(df["model"] == best) & (df["level"] == lvl) & (df["H"] == H)]["score"].values
            sc[f"{lvl}:{H}"] = np.round(s, 3).tolist()
    sc["hourly:1"] = np.round(hs, 3).tolist()

    gate_row = next(r for r in main if r["level"] == "bottom" and r["H"] == Hsel and r["model"] == best)
    sn_row = next(r for r in main if r["level"] == "bottom" and r["H"] == Hsel and r["model"] == "seasonal_naive")
    return {
        "origins": [ds.dates[o].date().isoformat() for o in origins],
        "horizons": list(cfg.horizons),
        "chosen_reconciliation": chosen,
        "model": best,
        "main": main,
        "reconciliation": rec_table,
        "by_location": by_loc,
        "by_class": by_cls,
        "by_event": by_event,
        "by_event_item": by_event_item,
        "cold_start": cold,
        "hourly_coverage": _r(hourly_cov, 3),
        "gate": {
            "horizon": Hsel,
            "level": "bottom",
            "model_wape": gate_row["wape"],
            "snaive_wape": sn_row["wape"],
            "passed": gate_row["wape"] < sn_row["wape"],
        },
        "cold_start_notes": cold_notes[-8:],
        "scores": sc,
        "seconds": round(seconds, 1),
        "rows": int(len(df)),
        "_frame": df,
    }
