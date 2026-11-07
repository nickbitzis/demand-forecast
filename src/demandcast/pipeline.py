"""End to end: generate, backtest, forecast the next 28 days, plan prep, orders, and staff, simulate a month.

`build()` returns one JSON-serializable dict (the artifacts) that the API serves and `demandcast train` writes to
data/artifacts.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import backtest, intervals, labor, ordering, prep, simulate
from .catalog import ING, ITEMS, LOCATIONS, N_SLOTS, SUPPLIER, VENUE, raw_per_portion, slot_label
from .forecast import LEVELS, Panels, base_forecasts, fit, intraday_profiles, reconcile_all, split_intraday
from .generator import SITE_KINDS, Dataset, GenConfig, generate
from .models import GBMConfig

SIM_DAYS = 28
RETRAIN_EVERY = 7


@dataclass
class PipelineConfig:
    gen: GenConfig = field(default_factory=GenConfig)
    backtest: backtest.BacktestConfig = field(default_factory=lambda: backtest.BacktestConfig(origins=12))
    sim_days: int = SIM_DAYS

    @classmethod
    def quick(cls, seed: int = 42) -> PipelineConfig:
        """Small and fast, for CI and tests: shorter history, fewer origins, fewer trees."""
        g = GBMConfig(max_iter=60)
        return cls(
            gen=GenConfig(seed=seed, days=420),
            backtest=backtest.BacktestConfig(origins=4, step=14, warmup=1, gbm=g),
            sim_days=14,
        )


def _horizon_bucket(lead: int) -> int:
    return 1 if lead <= 1 else (7 if lead <= 7 else 28)


def _scores(bt: dict, key: str, fallback: str) -> np.ndarray:
    s = np.array(bt["scores"].get(key) or [])
    if len(s) < 50:
        s = np.array(bt["scores"][fallback])
    return s


def live_forecast(ds: Dataset, P: Panels, bt: dict, gbm: GBMConfig) -> dict:
    """Forecast of the 28 days after the last history day, reconciled with the chosen method, with intervals."""
    method = bt["chosen_reconciliation"]
    c = ds.n_hist - 1
    H = ds.cfg.horizon
    days_all = np.arange(c + 1, c + 1 + H)
    out = {lvl: np.zeros((P[lvl].Y.shape[0], H)) for lvl in LEVELS}
    lo = {lvl: np.zeros_like(v) for lvl, v in out.items()}
    hi = {lvl: np.zeros_like(v) for lvl, v in out.items()}
    no_event_loc = np.zeros((len(LOCATIONS), H))
    raw_loc = np.zeros((len(LOCATIONS), H))
    cold = []
    for shift, (a, b) in ((1, (0, 1)), (7, (1, min(7, H))), (28, (min(7, H), H))):
        if a >= b:
            continue
        days = days_all[a:b]
        models = fit(P, c, shift, gbm)
        base = base_forecasts(models, P, days)
        cold += base["cold_start"]
        rec = reconcile_all(base, models, P, c, days, [method])[method]
        raw_loc[:, a:b] = models["location"].predict(days)
        no_event_loc[:, a:b] = models["location"].predict(days, overrides={"pressure": 0.0, "event_start": 0.0})
        for lvl in LEVELS:
            v = np.atleast_2d(rec[lvl])
            out[lvl][:, a:b] = v
            s = _scores(bt, f"{lvl}:{shift}", f"{lvl}:7" if f"{lvl}:7" in bt["scores"] else "bottom:7")
            lo_, hi_ = intervals.interval(v, s)
            lo[lvl][:, a:b] = lo_
            hi[lvl][:, a:b] = hi_
    return {
        "days": days_all,
        "mean": out,
        "lo": lo,
        "hi": hi,
        "no_event_ratio": np.where(raw_loc > 0, no_event_loc / np.maximum(raw_loc, 1e-9), 1.0),
        "cold_start": cold,
    }


def synthetic_on_hand(today, usage: dict, rng) -> dict:
    """Stock at close today: enough to reach the supplier's next delivery, plus a random part-day to a few days."""
    out = {}
    for ing, u in usage.items():
        sup = SUPPLIER[ING[ing].supplier]
        first = ordering.opportunities(today, sup.order_weekdays, 8)[0] + sup.lead_days
        extra = rng.uniform(0.2, 1.2) if ING[ing].shelf_life_days < 30 else rng.uniform(1.0, 6.0)
        out[ing] = float(u[:first].sum() + u[:3].mean() * extra)
    return out


def _r(x, d=1):
    return round(float(x), d)


def build(cfg: PipelineConfig | None = None, log=print) -> dict:
    cfg = cfg or PipelineConfig()
    t0 = time.perf_counter()
    ds = generate(cfg.gen)
    P = Panels.build(ds)
    log(f"generated {ds.units.size:,} slot cells, {int(ds.units.sum()):,} units, {len(ds.events)} events")
    bt = backtest.run(ds, cfg.backtest, log=log)
    frame = bt.pop("_frame")
    log(f"backtest done in {bt['seconds']}s, chosen reconciliation {bt['chosen_reconciliation']}, gate {bt['gate']}")

    fc = live_forecast(ds, P, bt, cfg.backtest.gbm)
    c = ds.n_hist - 1
    tomorrow = c + 1
    dates = [d.date().isoformat() for d in ds.dates]
    today = ds.dates[c].date()
    prof = intraday_profiles(ds, c)
    hourly_scores = np.array(bt["scores"]["hourly:1"])
    item_scores = _scores(bt, "bottom:1", "bottom:7")
    rng = np.random.default_rng(cfg.gen.seed + 1)

    locations = {}
    for li, loc in enumerate(LOCATIONS):
        rows = np.where(P.bottom.loc_index == li)[0]
        daily = [
            {
                "date": dates[d],
                "mean": _r(fc["mean"]["location"][li, j]),
                "lo": _r(fc["lo"]["location"][li, j]),
                "hi": _r(fc["hi"]["location"][li, j]),
                "open": bool(ds.open_[li, d]),
                "site": SITE_KINDS[ds.site[li, d]],
                "event_pressure": _r(ds.pressure[li, d], 2),
            }
            for j, d in enumerate(fc["days"])
        ]
        items = {
            ITEMS[P.bottom.item_index[r]].id: {
                "mean": [_r(v) for v in fc["mean"]["bottom"][r]],
                "lo": [_r(v) for v in fc["lo"]["bottom"][r]],
                "hi": [_r(v) for v in fc["hi"]["bottom"][r]],
            }
            for r in rows
        }
        # Tomorrow by 15 minutes and by hour.
        f_tom = fc["mean"]["location"][li, 0]
        slots = split_intraday(ds, prof, li, tomorrow, f_tom, f_tom * fc["no_event_ratio"][li, 0])
        hours = []
        for h in range(N_SLOTS // 4):
            m = float(slots[4 * h : 4 * h + 4].sum())
            lo_, hi_ = intervals.interval(np.array([m]), hourly_scores) if m > 0 else (np.zeros(1), np.zeros(1))
            hours.append({"hour": slot_label(4 * h), "mean": _r(m), "lo": _r(lo_[0]), "hi": _r(hi_[0])})
        shape = slots / slots.sum() if slots.sum() > 0 else np.zeros(N_SLOTS)
        item_slots = {ITEMS[P.bottom.item_index[r]].id: fc["mean"]["bottom"][r, 0] * shape for r in rows}
        staff = (
            labor.plan(item_slots, loc.mobile)
            if f_tom > 0
            else {"requirements": {}, "shifts": [], "labor_hours": 0, "peak_staff": 0, "slots": []}
        )
        # Prep list for tomorrow.
        tom_fc = {
            ITEMS[P.bottom.item_index[r]].id: {
                "mean": fc["mean"]["bottom"][r, 0],
                "lo": fc["lo"]["bottom"][r, 0],
                "hi": fc["hi"]["bottom"][r, 0],
            }
            for r in rows
        }
        last_week = {ITEMS[P.bottom.item_index[r]].id: float(np.nan_to_num(P.bottom.Y[r, tomorrow - 7])) for r in rows}
        plist = prep.prep_list(tom_fc, item_scores, last_week)
        # Orders: expected ingredient usage and its spread over the next 28 days.
        usage, sd = {}, {}
        for r in rows:
            it = ITEMS[P.bottom.item_index[r]]
            m = fc["mean"]["bottom"][r]
            s = (fc["hi"]["bottom"][r] - fc["lo"]["bottom"][r]) / (2 * 1.645)
            for ing, q in raw_per_portion(it).items():
                usage[ing] = usage.get(ing, 0) + m * q
                sd[ing] = sd.get(ing, 0) + s * q
        on_hand = synthetic_on_hand(today, usage, rng)
        orders = ordering.plan(today, usage, sd, on_hand) if usage else {"next": [], "lines": []}
        d_tom = daily[0]
        locations[loc.id] = {
            "id": loc.id,
            "name": loc.name,
            "mobile": loc.mobile,
            "tomorrow": {**d_tom, "hours": hours, "slots": [_r(v, 2) for v in slots]},
            "daily": daily,
            "items": items,
            "prep": plist,
            "orders": orders,
            "labor": staff,
        }

    sim = run_simulation(ds, P, bt, frame, cfg)
    log(f"simulation done: {[(p['policy'], round(p['total_cost'])) for p in sim['policies']]}")
    tot = [
        {
            "date": dates[d],
            "mean": _r(fc["mean"]["total"][0, j]),
            "lo": _r(fc["lo"]["total"][0, j]),
            "hi": _r(fc["hi"]["total"][0, j]),
        }
        for j, d in enumerate(fc["days"])
    ]
    events_ahead = [e for e in ds.events if c < e["day"] <= c + 7]
    return {
        "generated": {
            "seed": cfg.gen.seed,
            "history_days": ds.n_hist,
            "start": dates[0],
            "end": dates[c],
            "units": int(ds.units.sum()),
            "events": len(ds.events),
        },
        "today": dates[c],
        "tomorrow": dates[tomorrow],
        "weather_tomorrow": {
            "temp_max": float(ds.temp[tomorrow]),
            "precip_mm": float(ds.precip[tomorrow]),
            "snow_cm": float(ds.snow[tomorrow]),
            "holiday": ds.holiday[tomorrow],
        },
        "events_ahead": [
            {
                "venue_name": VENUE[e["venue"]].name,
                **{k: e[k] for k in ("date", "venue", "kind", "attendance", "start")},
            }
            for e in events_ahead
        ],
        "locations": locations,
        "total": tot,
        "cold_start": fc["cold_start"],
        "backtest": bt,
        "simulation": sim,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def run_simulation(ds: Dataset, P: Panels, bt: dict, frame, cfg: PipelineConfig) -> dict:
    """Replay the last `sim_days` of history with one-day-ahead forecasts, models retrained weekly."""
    method = bt["chosen_reconciliation"]
    S = ds.n_hist - cfg.sim_days
    sim_days = np.arange(S, ds.n_hist)
    F = np.zeros((P.bottom.Y.shape[0], len(sim_days)))
    models = None
    for j, d in enumerate(sim_days):
        if j % RETRAIN_EVERY == 0:
            models = fit(P, d - 1, 1, cfg.backtest.gbm)
        days = np.array([d])
        base = base_forecasts(models, P, days, cutoff=d - 1)
        F[:, j] = reconcile_all(base, models, P, d - 1, days, [method])[method]["bottom"][:, 0]
    Y = P.bottom.Y[:, sim_days]
    gut = P.bottom.Y[:, sim_days - 7]
    gut = np.where(np.isnan(gut), P.bottom.Y[:, sim_days - 14], gut)
    sel = frame[(frame["model"] == f"gbm_{method}") & (frame["level"] == "bottom") & (frame["day"] < S)]
    s1 = sel[sel["H"] == 1]["score"].values
    scores = s1 if len(s1) >= 150 else sel[sel["H"] == 7]["score"].values
    out = simulate.run(
        Y,
        F,
        gut,
        P.bottom.item_index,
        P.bottom.loc_index,
        P.bottom.avail[:, sim_days],
        scores,
        [d.date().isoformat() for d in ds.dates[sim_days]],
    )
    out["calibration_scores"] = int(len(scores))
    return out


def save(art: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, separators=(",", ":"), default=_default))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
