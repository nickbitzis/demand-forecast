"""Seeded synthetic history: two years of 15-minute item sales per location, plus the known future calendar.

The data generating process, in order:

1. Weather per day: temperature with a seasonal curve and AR(1) noise, precipitation by month, snow below 2 C.
2. Calendar: US holidays (closed on Thanksgiving and Christmas), weekday effects per location, gentle growth.
3. Events: baseball, football, arena, amphitheater, and festival days with attendance and a start time. Each brick
   location gets a lift of 0.012 units per attendee, decaying with distance as exp(-km / 0.8).
4. Channels: dine in, patio, takeout, delivery. Rain and cold close most of the patio (35% of those guests move
   inside), snow cuts walk-in traffic by 28% and raises delivery by 45%.
5. The food truck parks at the biggest event of the day, or at an office lot on weekdays and a brewery row on
   weekends. It is closed on Mondays without an event and on snow days. Its demand is driven by where it parks.
6. Items: base shares, drinks up in heat, soup up in cold, promotions (+40% on one item for a week), a new bowl
   launched 53 days before the end with no history, partly cannibalizing the other bowls.
7. Highlands opens 68 days before the end (cold start), with a grand opening bump.
8. Noise: lognormal at the location-day and location-item-day level, then Poisson counts per 15-minute slot.

Future days (the forecast horizon) get weather, events, promotions, and the truck schedule but no sales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .catalog import (
    CLOSE_SLOT,
    ITEMS,
    LOCATIONS,
    N_SLOTS,
    OPEN_SLOT,
    VENUE,
    VENUES,
    distance_km,
    slot_hour,
)

CHANNELS = ["dine_in", "patio", "takeout", "delivery"]
SITE_KINDS = ["store", "office", "brewery", "baseball", "football", "arena", "amphitheater", "festival", "closed"]
EVENT_DECAY_KM = 0.8
EVENT_UNITS_PER_ATTENDEE = 0.012

DOW = {  # Monday first
    "lodo": [0.97, 1.00, 1.03, 1.05, 1.12, 0.90, 0.78],
    "cap-hill": [0.90, 0.94, 0.98, 1.03, 1.16, 1.18, 1.00],
    "highlands": [0.88, 0.92, 0.97, 1.02, 1.15, 1.22, 1.08],
}
SEASON = [0.88, 0.90, 0.95, 0.99, 1.03, 1.06, 1.07, 1.06, 1.03, 1.00, 0.95, 0.91]


@dataclass
class GenConfig:
    seed: int = 42
    days: int = 730
    horizon: int = 28
    end: date = date(2026, 11, 7)


@dataclass
class Dataset:
    cfg: GenConfig
    dates: pd.DatetimeIndex  # history plus horizon
    n_hist: int
    units: np.ndarray  # (loc, item, hist day, slot) int16 sales
    open_: np.ndarray  # (loc, day) bool, known in advance
    on_menu: np.ndarray  # (loc, item, day) bool
    promo: np.ndarray  # (item, day) bool
    temp: np.ndarray  # (day,) daily high, C
    precip: np.ndarray  # (day,) mm water
    snow: np.ndarray  # (day,) cm
    holiday: list  # (day,) name or ""
    events: list  # dicts: day, venue, kind, attendance, start
    pressure: np.ndarray  # (loc, day) distance-weighted attendance / 10,000, or the truck's own event
    event_kernel: np.ndarray  # (loc, day, slot) intraday shape of the event lift, sums to 1 when there is one
    event_start: np.ndarray  # (loc, day) start hour of the main event, 0 if none
    site: np.ndarray  # (loc, day) index into SITE_KINDS
    channels: np.ndarray  # (loc, hist day, channel) expected units by channel, scaled to the day's sales
    profile_truth: np.ndarray = field(repr=False, default=None)  # (loc, day, slot) base intraday shape

    @property
    def daily(self) -> np.ndarray:
        """(loc, item, hist day) daily units."""
        return self.units.sum(axis=3, dtype=np.int32)

    def day_index(self, d: date | str) -> int:
        return int(self.dates.get_loc(pd.Timestamp(d)))


def us_holidays(years: range) -> dict[date, str]:
    def nth(y, m, wd, n):
        d = date(y, m, 1)
        d += timedelta(days=(wd - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last(y, m, wd):
        d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
        return d - timedelta(days=(d.weekday() - wd) % 7)

    out = {}
    for y in years:
        out[date(y, 1, 1)] = "New Year's Day"
        out[nth(y, 1, 0, 3)] = "MLK Day"
        out[nth(y, 2, 0, 3)] = "Presidents Day"
        out[last(y, 5, 0)] = "Memorial Day"
        out[date(y, 7, 4)] = "Independence Day"
        out[nth(y, 9, 0, 1)] = "Labor Day"
        tg = nth(y, 11, 3, 4)
        out[tg] = "Thanksgiving"
        out[tg + timedelta(days=1)] = "Black Friday"
        out[date(y, 12, 24)] = "Christmas Eve"
        out[date(y, 12, 25)] = "Christmas"
        out[date(y, 12, 31)] = "New Year's Eve"
    return out


HOLIDAY_FACTOR = {
    "New Year's Day": 0.62,
    "MLK Day": 0.95,
    "Presidents Day": 0.95,
    "Memorial Day": 0.84,
    "Independence Day": 0.72,
    "Labor Day": 0.86,
    "Thanksgiving": 0.0,
    "Black Friday": 0.88,
    "Christmas Eve": 0.55,
    "Christmas": 0.0,
    "New Year's Eve": 0.80,
}


def _weather(rng, dates):
    doy = dates.dayofyear.values
    mean = 19.0 - 12.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    noise = np.zeros(len(dates))
    for t in range(1, len(dates)):
        noise[t] = 0.65 * noise[t - 1] + rng.normal(0, 4.2)
    temp = mean + noise
    month = dates.month.values
    p_wet = np.array([0.14, 0.16, 0.20, 0.24, 0.30, 0.28, 0.33, 0.32, 0.20, 0.16, 0.14, 0.13])[month - 1]
    wet = rng.random(len(dates)) < p_wet
    amount = np.where(wet, rng.gamma(1.3, 4.5, len(dates)), 0.0)
    snow = np.where(temp < 2.0, amount * 1.0, 0.0)  # mm water to cm snow at 10:1
    precip = amount
    return np.round(temp, 1), np.round(precip, 1), np.round(snow, 1)


def _events(rng, dates):
    ev = []
    for t, ts in enumerate(dates):
        d = ts.date()
        wd = d.weekday()
        m = d.month
        if 4 <= m <= 9 and rng.random() < 0.45:
            att = int(rng.uniform(26000, 40000) * (1.12 if wd >= 4 else 1.0))
            ev.append({"day": t, "venue": "ballpark", "attendance": att, "start": 13.1 if wd >= 5 else 18.7})
        if (m >= 9 or m == 1) and wd == 6 and (d.isocalendar()[1] % 2 == 0) and not (m == 1 and d.day > 10):
            ev.append(
                {
                    "day": t,
                    "venue": "stadium",
                    "attendance": int(rng.uniform(71000, 76000)),
                    "start": 14.1 if rng.random() < 0.7 else 18.3,
                }
            )
        if (m >= 10 or m <= 4) and wd in (1, 3, 5) and rng.random() < 0.6:
            ev.append({"day": t, "venue": "arena", "attendance": int(rng.uniform(14000, 19000)), "start": 19.0})
        elif rng.random() < 0.07:
            ev.append({"day": t, "venue": "arena", "attendance": int(rng.uniform(12000, 17000)), "start": 19.5})
        if 5 <= m <= 9 and rng.random() < 0.55:
            ev.append({"day": t, "venue": "amphitheater", "attendance": int(rng.uniform(8000, 9500)), "start": 19.5})
        # Festival weekends: the second Saturday and Sunday of May to September.
        if 5 <= m <= 9 and wd in (5, 6) and 8 <= d.day - (wd - 5) <= 14:
            ev.append({"day": t, "venue": "civic", "attendance": int(rng.uniform(55000, 85000)), "start": 11.0})
    for e in ev:
        e["kind"] = VENUE[e["venue"]].kind
        e["date"] = dates[e["day"]].date().isoformat()
    return ev


def _gauss(center, sd):
    h = np.array([slot_hour(s) + 0.125 for s in range(N_SLOTS)])
    return np.exp(-0.5 * ((h - center) / sd) ** 2)


def _mask_open(p):
    p = p.copy()
    p[:OPEN_SLOT] = 0
    p[CLOSE_SLOT:] = 0
    return p / p.sum()


def base_profile(lunch_weight: float, weekend: bool) -> np.ndarray:
    if weekend:
        p = lunch_weight * 0.8 * _gauss(12.8, 1.3) + (1 - lunch_weight * 0.8) * _gauss(18.0, 1.4) + 0.12
    else:
        p = lunch_weight * _gauss(12.25, 0.8) + (1 - lunch_weight) * 0.9 * _gauss(18.1, 1.1) + 0.06
    return _mask_open(p)


def event_kernel(kind: str, start: float) -> np.ndarray:
    if kind == "festival":
        p = _gauss(14.5, 2.6)
    else:
        p = _gauss(start - 1.3, 0.75) + 0.25 * _gauss(start + 2.6, 0.5)
    return _mask_open(p)


TRUCK_PREF = {"festival": 1.3, "football": 1.2, "baseball": 1.0, "amphitheater": 1.1, "arena": 0.8}
TRUCK_RATE = {"festival": 0.0042, "football": 0.0032, "baseball": 0.0055, "amphitheater": 0.016, "arena": 0.0075}


def generate(cfg: GenConfig | None = None) -> Dataset:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    start = cfg.end - timedelta(days=cfg.days - 1)
    dates = pd.date_range(start, periods=cfg.days + cfg.horizon, freq="D")
    n_all, n_hist = len(dates), cfg.days
    L, n_items = len(LOCATIONS), len(ITEMS)

    temp, precip, snow = _weather(rng, dates)
    hol_map = us_holidays(range(dates[0].year, dates[-1].year + 1))
    holiday = [hol_map.get(ts.date(), "") for ts in dates]
    events = _events(rng, dates)
    dow = dates.day_of_week.values
    years = (np.arange(n_all) - n_all) / 365.25  # negative, growth toward the end

    open_ = np.ones((L, n_all), dtype=bool)
    closed_hol = np.array([HOLIDAY_FACTOR.get(h, 1.0) == 0.0 for h in holiday])
    open_[:, closed_hol] = False
    on_menu = np.ones((L, n_items, n_all), dtype=bool)
    promo = np.zeros((n_items, n_all), dtype=bool)
    pressure = np.zeros((L, n_all))
    kernel = np.zeros((L, n_all, N_SLOTS))
    ev_start = np.zeros((L, n_all))
    site = np.zeros((L, n_all), dtype=np.int16)
    profile = np.zeros((L, n_all, N_SLOTS))

    # Promotions: a 7-day promo on one food item roughly every 6 weeks.
    food = [k for k, it in enumerate(ITEMS) if it.cls in ("bowl", "wrap", "side") and it.launch_days_before_end is None]
    t = int(rng.integers(10, 30))
    while t < n_all:
        promo[int(rng.choice(food)), t : t + 7] = True
        t += int(rng.integers(35, 50))

    new_items = {}
    for k, it in enumerate(ITEMS):
        if it.launch_days_before_end is not None:
            launch = n_hist - it.launch_days_before_end
            on_menu[:, k, :launch] = False
            new_items[k] = launch
    k_soup = [k for k, it in enumerate(ITEMS) if it.id == "soup"][0]
    truck = [li for li, loc in enumerate(LOCATIONS) if loc.mobile][0]
    on_menu[truck, k_soup, :] = False

    # Event lift for brick locations, and the truck's site choice.
    by_day: dict[int, list] = {}
    for e in events:
        by_day.setdefault(e["day"], []).append(e)
    lift = np.zeros((L, n_all))
    for li, loc in enumerate(LOCATIONS):
        if loc.mobile:
            continue
        for d, evs in by_day.items():
            best = 0.0
            for e in evs:
                w = np.exp(-distance_km(loc, VENUE[e["venue"]]) / EVENT_DECAY_KM)
                units = EVENT_UNITS_PER_ATTENDEE * e["attendance"] * w
                if units < 1:
                    continue
                lift[li, d] += units
                pressure[li, d] += e["attendance"] / 10000 * w
                kernel[li, d] += units * event_kernel(e["kind"], e["start"])
                if units > best:
                    best, ev_start[li, d] = units, e["start"]
            if lift[li, d] > 0:
                kernel[li, d] /= kernel[li, d].sum()

    truck_level = np.zeros(n_all)
    for d in range(n_all):
        wd = int(dow[d])
        evs = [e for e in by_day.get(d, []) if e["attendance"] >= 8000]
        if not open_[truck, d] or snow[d] > 3:
            open_[truck, d] = False
            site[truck, d] = SITE_KINDS.index("closed")
            continue
        if evs:
            e = max(evs, key=lambda e: e["attendance"] * TRUCK_PREF[e["kind"]])
            site[truck, d] = SITE_KINDS.index(e["kind"])
            pressure[truck, d] = e["attendance"] / 10000
            ev_start[truck, d] = e["start"]
            truck_level[d] = 40 + TRUCK_RATE[e["kind"]] * e["attendance"]
            kernel[truck, d] = event_kernel(e["kind"], e["start"])
            profile[truck, d] = kernel[truck, d]
        elif wd == 0:
            open_[truck, d] = False
            site[truck, d] = SITE_KINDS.index("closed")
        elif wd <= 4:
            site[truck, d] = SITE_KINDS.index("office")
            truck_level[d] = LOCATIONS[truck].base_daily
            profile[truck, d] = _mask_open(_gauss(12.2, 0.7) + 0.05)
        else:
            site[truck, d] = SITE_KINDS.index("brewery")
            truck_level[d] = LOCATIONS[truck].base_daily * 1.25
            profile[truck, d] = _mask_open(_gauss(16.5, 2.0))

    # Location-day totals.
    rain = (precip > 1.0) & (snow == 0)
    snowy = snow > 2.0
    expected_total = np.zeros((L, n_all))
    chan = np.zeros((L, n_all, 4))
    for li, loc in enumerate(LOCATIONS):
        if loc.mobile:
            season = np.array(SEASON)[dates.month.values - 1]
            wx = np.where(rain, 0.6, 1.0) * np.where(temp < 5, 0.7, 1.0)
            event_day = site[li] >= SITE_KINDS.index("baseball")
            mult = np.where(event_day, 1.0, season * wx)
            expected_total[li] = truck_level * mult * np.exp(0.05 * years)
            chan[li, :, 2] = expected_total[li]
            continue
        base = (
            loc.base_daily
            * np.exp(0.06 * years)
            * np.array(DOW[loc.id])[dow]
            * np.array(SEASON)[dates.month.values - 1]
            * np.array([HOLIDAY_FACTOR.get(h, 1.0) for h in holiday])
        )
        sh = loc.channels
        patio_f = np.where(temp < 10, 0.25, np.where(temp < 16, 0.6, 1.0)) * np.where(rain, 0.22, 1.0)
        patio_f = np.where(snowy, 0.0, patio_f) * np.where((temp > 22) & (temp < 31) & ~rain, 1.12, 1.0)
        displaced = sh["patio"] * (1 - np.minimum(patio_f, 1.0)) * 0.35
        walk = np.where(snowy, 0.72, 1.0)
        dine = (sh["dine_in"] * walk + displaced * walk) * base
        patio = sh["patio"] * patio_f * base
        take = sh["takeout"] * walk * np.where(rain, 0.95, 1.0) * base
        deliv = sh["delivery"] * np.where(snowy, 1.45, np.where(rain, 1.15, 1.0)) * base
        chan[li] = np.stack([dine, patio, take, deliv], axis=1)
        expected_total[li] = chan[li].sum(axis=1) + lift[li]
        chan[li, :, 0] += lift[li] * 0.6
        chan[li, :, 2] += lift[li] * 0.4
        for d in range(n_all):
            profile[li, d] = base_profile(loc.lunch_weight, dow[d] >= 5)
        if loc.opens_days_before_end is not None:
            op = n_hist - loc.opens_days_before_end
            open_[li, :op] = False
            k = np.arange(n_all) - op
            ramp = np.where(
                k >= 0, 0.9 + 0.1 * np.minimum(1, np.maximum(k, 0) / 30) + 0.3 * np.exp(-np.maximum(k, 0) / 3.0), 0
            )
            expected_total[li] *= ramp
            chan[li] *= ramp[:, None]
    expected_total *= open_

    # Item split.
    shares = np.zeros((L, n_items, n_all))
    for k, it in enumerate(ITEMS):
        s = np.full(n_all, it.share)
        if it.cls == "drink":
            sens = {"lemonade": 0.04, "iced_tea": 0.03, "horchata": 0.012}[it.id]
            s = s * np.clip(1 + sens * (temp - 20), 0.55, 1.6)
        if it.id == "soup":
            s = s * np.clip(1 + 0.05 * (14 - temp), 0.35, 2.2)
        s = s * np.where(promo[k], 1.4, 1.0)
        shares[:, k] = s
    event_day = pressure > 0.4
    for k, it in enumerate(ITEMS):
        if it.cls == "drink" or it.id == "chips_salsa":
            shares[:, k] *= np.where(event_day, 1.15, 1.0)
    shares *= on_menu
    shares /= np.maximum(shares.sum(axis=1, keepdims=True), 1e-9)
    bowls = [k for k, it in enumerate(ITEMS) if it.cls == "bowl"]
    for k, launch in new_items.items():
        age = np.arange(n_all) - launch
        hype = np.where(age >= 0, 1 + 0.6 * np.exp(-np.maximum(age, 0) / 7.0), 0)
        class_share = shares[:, [b for b in bowls if b != k]].sum(axis=1)
        s_new = ITEMS[k].analog_share * class_share * hype * on_menu[:, k]
        for b in bowls:
            if b != k:
                shares[:, b] *= 1 - 0.5 * s_new / np.maximum(class_share, 1e-9)
        shares[:, k] = s_new  # half cannibalized, half new demand

    day_noise = rng.lognormal(-0.5 * 0.07**2, 0.07, (L, n_all))
    item_noise = rng.lognormal(-0.5 * 0.08**2, 0.08, (L, n_items, n_all))
    mean_daily = expected_total[:, None, :] * shares * day_noise[:, None, :] * item_noise

    # Intraday: base part follows the location profile, the event lift follows the event kernel.
    base_part = np.where(expected_total > 0, (expected_total - lift) / np.maximum(expected_total, 1e-9), 1.0)
    base_part = np.clip(base_part, 0, 1)
    is_truck = np.array([loc.mobile for loc in LOCATIONS])
    base_part[is_truck] = 1.0
    shape = base_part[..., None] * profile + (1 - base_part)[..., None] * kernel  # (L, n_all, slot)
    lam = mean_daily[:, :, :n_hist, None] * shape[:, None, :n_hist, :]
    units = rng.poisson(lam).astype(np.int16)

    totals = units.sum(axis=(1, 3))
    chan_hist = chan[:, :n_hist]
    chan_hist = chan_hist / np.maximum(chan_hist.sum(axis=2, keepdims=True), 1e-9) * totals[..., None]

    return Dataset(
        cfg=cfg,
        dates=dates,
        n_hist=n_hist,
        units=units,
        open_=open_,
        on_menu=on_menu,
        promo=promo,
        temp=temp,
        precip=precip,
        snow=snow,
        holiday=holiday,
        events=events,
        pressure=pressure,
        event_kernel=kernel,
        event_start=ev_start,
        site=site,
        channels=chan_hist,
        profile_truth=profile,
    )


def venues_table() -> list[dict]:
    return [{"id": v.id, "name": v.name, "kind": v.kind, "x": v.x, "y": v.y} for v in VENUES]
