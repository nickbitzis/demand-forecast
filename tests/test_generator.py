from datetime import date

import numpy as np

from demandcast.catalog import CLOSE_SLOT, ITEMS, LOC_IDS, OPEN_SLOT
from demandcast.generator import CHANNELS, SITE_KINDS, GenConfig, generate, us_holidays

LODO, CAP, HIGH, TRUCK = (LOC_IDS.index(x) for x in ("lodo", "cap-hill", "highlands", "truck"))


def hist(ds, a):
    return a[..., : ds.n_hist]


def test_shapes_and_dates(ds):
    assert ds.n_hist == 730
    assert len(ds.dates) == 730 + 28
    assert ds.dates[ds.n_hist - 1].date() == date(2026, 11, 7)
    assert ds.units.shape == (4, len(ITEMS), 730, 48)
    assert ds.units.dtype == np.int16


def test_same_seed_same_data_and_new_seed_differs():
    a, b, c = (
        generate(GenConfig(seed=3, days=120)),
        generate(GenConfig(seed=3, days=120)),
        generate(GenConfig(seed=4, days=120)),
    )
    assert np.array_equal(a.units, b.units)
    assert not np.array_equal(a.units, c.units)


def test_no_sales_outside_opening_hours(ds):
    assert ds.units[..., :OPEN_SLOT].sum() == 0
    assert ds.units[..., CLOSE_SLOT:].sum() == 0


def test_lunch_peak_on_weekdays_at_lodo(ds):
    wd = ds.dates[: ds.n_hist].day_of_week < 5
    prof = ds.units[LODO][:, wd].sum(axis=(0, 1))
    peak = int(np.argmax(prof))
    assert 12 <= 10 + peak * 0.25 <= 13


def test_closed_on_thanksgiving_and_christmas(ds):
    for d in ("2025-11-27", "2025-12-25"):
        i = ds.day_index(d)
        assert ds.units[:, :, i].sum() == 0
        assert not ds.open_[:, i].any()


def test_holiday_dates():
    h = us_holidays(range(2025, 2027))
    assert h[date(2025, 11, 27)] == "Thanksgiving"
    assert h[date(2026, 5, 25)] == "Memorial Day"
    assert h[date(2026, 9, 7)] == "Labor Day"


def _warm_dry_vs_rain(ds):
    t = hist(ds, ds.temp)
    rain = (hist(ds, ds.precip) > 1) & (hist(ds, ds.snow) == 0) & (t > 18)
    dry = (hist(ds, ds.precip) == 0) & (t > 18)
    return rain, dry


def test_rain_lowers_patio_demand(ds):
    rain, dry = _warm_dry_vs_rain(ds)
    patio = ds.channels[LODO, :, CHANNELS.index("patio")]
    assert patio[rain].mean() < 0.4 * patio[dry].mean()


def test_rain_hurts_the_patio_location_more(ds):
    rain, dry = _warm_dry_vs_rain(ds)
    calm = hist(ds, ds.pressure).max(axis=0) < 0.2
    tot = ds.daily.sum(axis=1)
    lodo = tot[LODO][rain & calm].mean() / tot[LODO][dry & calm].mean()
    cap = tot[CAP][rain & calm].mean() / tot[CAP][dry & calm].mean()
    assert lodo < cap - 0.08


def test_snow_raises_delivery_share(ds):
    snow = hist(ds, ds.snow) > 2
    ch = ds.channels[CAP]
    share = ch[:, CHANNELS.index("delivery")] / np.maximum(ch.sum(axis=1), 1)
    open_ = ch.sum(axis=1) > 0
    assert share[snow & open_].mean() > share[~snow & open_].mean() * 1.3


def test_events_raise_nearby_demand(ds):
    tot = ds.daily.sum(axis=1)[LODO]
    ball = np.zeros(ds.n_hist, bool)
    for e in ds.events:
        if e["venue"] == "ballpark" and e["day"] < ds.n_hist:
            ball[e["day"]] = True
    other = hist(ds, ds.pressure)[LODO] < 0.1
    months = ds.dates[: ds.n_hist].month
    summer = (months >= 5) & (months <= 8)
    dow = ds.dates[: ds.n_hist].day_of_week
    wk = (dow >= 1) & (dow <= 3)
    assert tot[ball & summer & wk].mean() > 1.15 * tot[other & summer & wk & (tot > 0)].mean()


def test_event_effect_decays_with_distance(ds):
    ball_days = [e["day"] for e in ds.events if e["venue"] == "ballpark"]
    p = ds.pressure[:, ball_days]
    assert p[LODO].mean() > 3 * p[HIGH].mean()


def test_truck_demand_follows_its_event(ds):
    tot = ds.daily.sum(axis=1)[TRUCK]
    site = ds.site[TRUCK, : ds.n_hist]
    ev = site >= SITE_KINDS.index("baseball")
    ev &= site != SITE_KINDS.index("closed")
    office = site == SITE_KINDS.index("office")
    assert tot[ev].mean() > 1.8 * tot[office].mean()
    amph = site == SITE_KINDS.index("amphitheater")
    assert tot[amph].mean() > 1.5 * tot[office].mean()  # far from every store, still busy for the truck


def test_truck_closed_on_quiet_mondays_and_snow_days(ds):
    site = ds.site[TRUCK, : ds.n_hist]
    closed = site == SITE_KINDS.index("closed")
    assert ds.daily.sum(axis=1)[TRUCK][closed].sum() == 0
    snowy = hist(ds, ds.snow) > 3
    assert closed[snowy].all()


def test_new_location_cold_start(ds):
    tot = ds.daily.sum(axis=1)[HIGH]
    op = ds.n_hist - 68
    assert tot[:op].sum() == 0
    assert (tot[op:] > 0).mean() > 0.95


def test_new_item_launch_and_cannibalization(ds):
    k = [i for i, it in enumerate(ITEMS) if it.id == "green_chile_bowl"][0]
    b = [i for i, it in enumerate(ITEMS) if it.id == "chicken_bowl"][0]
    launch = ds.n_hist - 53
    d = ds.daily
    assert d[:, k, :launch].sum() == 0
    assert d[:, k, launch:].sum() > 0
    share = d[:, b].sum(axis=0) / np.maximum(d.sum(axis=(0, 1)), 1)
    assert share[launch + 21 : launch + 49].mean() < share[launch - 49 : launch - 21].mean()


def test_promotions_raise_item_share(ds):
    d = ds.daily.sum(axis=0)
    share = d / np.maximum(d.sum(axis=0), 1)
    lifts = []
    for k in range(len(ITEMS)):
        p = ds.promo[k, : ds.n_hist]
        if p.any():
            lifts.append(share[k][p].mean() / share[k][~p & (share[k] > 0)].mean())
    assert np.mean(lifts) > 1.25
