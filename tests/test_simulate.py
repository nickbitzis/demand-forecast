import numpy as np
import pytest

from demandcast import simulate


def test_accounting_identities_and_dollars():
    prep = np.array([[10.0, 5.0], [0.0, 8.0]])
    demand = np.array([[7.0, 9.0], [3.0, 8.0]])
    price, cost = np.array([12.0, 4.0]), np.array([3.0, 1.0])
    r = simulate.account(prep, demand, price, cost)
    assert r["sold_units"] + r["waste_units"] == r["prepped_units"]
    assert r["sold_units"] + r["lost_units"] == r["demand_units"]
    # waste: 3 units of item 0 at $3; lost: 4 of item 0 and 3 of item 1
    assert r["waste_cost"] == pytest.approx(9.0)
    assert r["lost_revenue"] == pytest.approx(4 * 12 + 3 * 4)
    assert r["lost_margin"] == pytest.approx(4 * 9 + 3 * 3)
    assert r["fill_rate"] == pytest.approx((7 + 5 + 0 + 8) / 27)
    assert r["stockout_days"] == 2


def test_policy_quantities():
    f = np.array([[10.4, 0.0]])
    g = np.array([[12.0, 3.0]])
    s = np.random.default_rng(0).normal(0, 1, 500)
    assert simulate.policy_quantities("gut_feel", f, g, s, np.array([0.8])).tolist() == [[12, 3]]
    assert simulate.policy_quantities("forecast_mean", f, g, s, np.array([0.8])).tolist() == [[10, 0]]
    nv = simulate.policy_quantities("newsvendor", f, g, s, np.array([0.8]))
    assert nv[0, 0] > 10.4 and nv[0, 1] == 0


def test_closed_days_have_no_prep_and_no_demand():
    Y = np.array([[5.0, np.nan], [4.0, 6.0]])
    out = simulate.run(
        Y,
        np.ones_like(Y) * 5,
        np.ones_like(Y) * 5,
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([[True, False], [True, True]]),
        np.zeros(10),
        ["a", "b"],
    )
    gut = out["policies"][0]
    assert gut["prepped_units"] == 15 and gut["demand_units"] == 15


def test_newsvendor_has_the_lowest_total_cost(art):
    p = {x["policy"]: x for x in art["simulation"]["policies"]}
    assert p["newsvendor"]["total_cost"] < p["forecast_mean"]["total_cost"] < p["gut_feel"]["total_cost"]
    assert p["newsvendor"]["fill_rate"] > p["gut_feel"]["fill_rate"]
    assert p["newsvendor"]["lost_margin"] < p["gut_feel"]["lost_margin"]


def test_simulation_demand_is_the_same_for_every_policy(art):
    d = {x["demand_units"] for x in art["simulation"]["policies"]}
    assert len(d) == 1
