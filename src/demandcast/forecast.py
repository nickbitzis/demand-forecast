"""Forecast engine: fit the three level models at a cutoff, apply cold start, reconcile, and split into 15 minutes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .catalog import LOCATIONS, N_SLOTS
from .features import Panel, item_panel, location_panel, total_panel
from .generator import SITE_KINDS, Dataset, event_kernel
from .models import GBM, GBMConfig, cold_start_item, cold_start_location
from .reconcile import historical_shares, reconcile

LEVELS = ("bottom", "location", "total")


@dataclass
class Panels:
    bottom: Panel
    location: Panel
    total: Panel

    @classmethod
    def build(cls, ds: Dataset) -> Panels:
        return cls(item_panel(ds), location_panel(ds), total_panel(ds))

    def __getitem__(self, k: str) -> Panel:
        return getattr(self, k)


def fit(panels: Panels, cutoff: int, shift: int, cfg: GBMConfig | None = None) -> dict[str, GBM]:
    return {lvl: GBM(panels[lvl], shift, cutoff, cfg) for lvl in LEVELS}


def base_forecasts(
    models: dict[str, GBM], panels: Panels, days: np.ndarray, cutoff: int | None = None, cold_start: bool = True
) -> dict:
    """Base forecasts per level for target days, features as of `cutoff` (default: the training cutoff)."""
    c = models["bottom"].cutoff if cutoff is None else cutoff
    out = {lvl: models[lvl].predict(days, c) for lvl in LEVELS}
    out["total"] = out["total"][0]
    notes = []
    if cold_start:
        notes += cold_start_item(out["bottom"], panels.bottom, c, days)
        notes += cold_start_location(out["bottom"], out["location"], panels.bottom, panels.location, c, days)
    out["cold_start"] = notes
    return out


def reconcile_all(base: dict, models: dict[str, GBM], panels: Panels, cutoff: int, days: np.ndarray, methods):
    b = panels.bottom
    kw = dict(
        bottom=base["bottom"],
        loc=base["location"],
        total=base["total"],
        var_bottom=models["bottom"].resid_var,
        var_loc=models["location"].resid_var,
        var_total=float(models["total"].resid_var[0]),
        bottom_loc=b.loc_index,
        active=b.avail[:, days],
        loc_active=panels.location.avail[:, days],
        shares=historical_shares(b.Y, panels.total.Y[0], cutoff),
    )
    return {m: reconcile(m, **kw) for m in methods}


# ---------------------------------------------------------------------------------------------------------------
# Intraday: daily forecast to 15-minute slots.


def _daytype(dow: int) -> int:
    return 0 if dow < 5 else (1 if dow == 5 else 2)


def intraday_profiles(ds: Dataset, cutoff: int, weeks: int = 8) -> dict:
    """Learned shapes as of the cutoff.

    base[loc, daytype, slot]: average slot share on non-event days over the last `weeks` weeks (for the truck,
    by parking spot instead of day type). event[kind]: average shape of the excess over the base profile on event
    days, aligned to the event start, pooled over locations and the full history up to the cutoff.
    """
    units = ds.units[:, :, : cutoff + 1].sum(axis=1).astype(float)  # (loc, day, slot)
    tot = units.sum(axis=2)
    dow = ds.dates.day_of_week.values
    L = len(LOCATIONS)
    base = np.zeros((L, 3, N_SLOTS))
    lo = max(0, cutoff + 1 - 7 * weeks)
    for li, loc in enumerate(LOCATIONS):
        for dt in range(3):
            if loc.mobile:
                spot = SITE_KINDS.index("office" if dt == 0 else "brewery")
                pool = [d for d in range(0, cutoff + 1) if ds.site[li, d] == spot and tot[li, d] > 0][-60:]
            else:
                pool = [
                    d
                    for d in range(lo, cutoff + 1)
                    if _daytype(dow[d]) == dt and ds.pressure[li, d] < 0.2 and tot[li, d] > 0
                ]
            if not pool:
                pool = [d for d in range(0, cutoff + 1) if tot[li, d] > 0][-28:]
            if not pool:
                base[li, dt] = 1.0 / N_SLOTS
                continue
            shares = units[li, pool] / tot[li, pool][:, None]
            base[li, dt] = shares.mean(axis=0)
    # Event excess aligned to the start slot, per kind, on a +-48 slot offset grid.
    ev = {}
    for kind in ("timed", "festival"):
        acc, n = np.zeros(2 * N_SLOTS), 0
        for li, loc in enumerate(LOCATIONS):
            for d in range(cutoff + 1):
                if ds.pressure[li, d] < 0.5 or tot[li, d] == 0:
                    continue
                is_fest = ds.event_start[li, d] == 11.0
                if (kind == "festival") != is_fest:
                    continue
                if loc.mobile:
                    excess = units[li, d]
                else:
                    excess = np.maximum(units[li, d] - base[li, _daytype(dow[d])] * tot[li, d] * 0.8, 0)
                if excess.sum() <= 0:
                    continue
                s0 = int(round((ds.event_start[li, d] - 10.0) * 4))
                for s in range(N_SLOTS):
                    acc[s - s0 + N_SLOTS] += excess[s] / excess.sum()
                n += 1
        ev[kind] = acc / n if n else None
    return {"base": base, "event": ev, "cutoff": cutoff}


def event_shape(profiles: dict, start: float) -> np.ndarray:
    kind = "festival" if start == 11.0 else "timed"
    acc = profiles["event"].get(kind)
    if acc is None:
        return event_kernel("festival" if kind == "festival" else "baseball", start)
    s0 = int(round((start - 10.0) * 4))
    out = np.array([acc[s - s0 + N_SLOTS] for s in range(N_SLOTS)])
    out[:2] = 0
    out[-2:] = 0
    return out / out.sum() if out.sum() > 0 else event_kernel("baseball", start)


def split_intraday(ds: Dataset, profiles: dict, li: int, day: int, daily: float, daily_no_event: float) -> np.ndarray:
    """15-minute forecast for one location-day: base profile for the no-event part, event shape for the uplift."""
    loc = LOCATIONS[li]
    if daily <= 0:
        return np.zeros(N_SLOTS)
    dow = int(ds.dates[day].day_of_week)
    start = float(ds.event_start[li, day])
    if loc.mobile:
        if start > 0:
            return daily * event_shape(profiles, start)
        return daily * profiles["base"][li, 0 if dow < 5 else 1]
    base = profiles["base"][li, _daytype(dow)]
    uplift = max(daily - daily_no_event, 0.0) if start > 0 else 0.0
    out = (daily - uplift) * base
    if uplift > 0:
        out = out + uplift * event_shape(profiles, start)
    return out
