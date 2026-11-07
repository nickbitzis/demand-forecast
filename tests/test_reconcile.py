import numpy as np

from demandcast.reconcile import aggregate, historical_shares, reconcile, summing_matrix

rng = np.random.default_rng(1)
BL = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # 8 bottom series in 3 locations


def kw(active=None, var_total=4.0):
    b = rng.uniform(5, 50, (8, 5))
    loc, tot = aggregate(b, BL, 3)
    loc = loc * rng.uniform(0.85, 1.15, loc.shape)
    tot = tot * 1.1
    active = np.ones((8, 5), bool) if active is None else active
    return dict(
        bottom=b,
        loc=loc,
        total=tot,
        var_bottom=np.full(8, 9.0),
        var_loc=np.full(3, 25.0),
        var_total=var_total,
        bottom_loc=BL,
        active=active,
        loc_active=np.ones((3, 5), bool),
        shares=np.full(8, 1 / 8),
    )


def coherent(r):
    loc, tot = aggregate(r["bottom"], BL, 3)
    return np.allclose(loc, r["location"]) and np.allclose(tot, r["total"])


def test_summing_matrix_shape_and_rows():
    S = summing_matrix(BL, 3)
    assert S.shape == (1 + 3 + 8, 8)
    assert S[0].sum() == 8 and S[1].sum() == 3 and S[2].sum() == 2


def test_every_method_is_coherent():
    for m in ("bottom_up", "top_down", "mint_lite"):
        assert coherent(reconcile(m, **kw()))


def test_bottom_up_keeps_item_forecasts():
    k = kw()
    r = reconcile("bottom_up", **k)
    assert np.allclose(r["bottom"], k["bottom"])


def test_top_down_splits_the_total_forecast():
    k = kw()
    r = reconcile("top_down", **k)
    assert np.allclose(r["total"], k["total"])
    assert np.allclose(r["bottom"], k["total"][None] / 8)


def test_mint_lite_leaves_coherent_input_unchanged():
    k = kw()
    k["loc"], k["total"] = aggregate(k["bottom"], BL, 3)
    r = reconcile("mint_lite", **k)
    assert np.allclose(r["bottom"], k["bottom"])


def test_mint_lite_trusts_the_low_variance_node():
    k = kw(var_total=1e-4)
    r = reconcile("mint_lite", **k)
    assert np.allclose(r["total"], k["total"], rtol=1e-3)


def test_inactive_series_are_zero_and_excluded():
    act = np.ones((8, 5), bool)
    act[2, :] = False
    for m in ("bottom_up", "top_down", "mint_lite"):
        r = reconcile(m, **kw(active=act))
        assert (r["bottom"][2] == 0).all()
        assert coherent(r)


def test_historical_shares_sum_to_one():
    Yb = rng.uniform(1, 10, (8, 40))
    Yt = Yb.sum(axis=0)
    assert np.isclose(historical_shares(Yb, Yt, 39).sum(), 1.0)


def test_live_forecast_sums_items_to_locations_to_total(art):
    tot = 0.0
    for loc in art["locations"].values():
        items = sum(v["mean"][0] for v in loc["items"].values())
        assert abs(items - loc["daily"][0]["mean"]) < 0.06 * len(loc["items"]) + 0.1
        tot += loc["daily"][0]["mean"]
    assert abs(tot - art["total"][0]["mean"]) < 0.3
