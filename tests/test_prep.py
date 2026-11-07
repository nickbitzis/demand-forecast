import numpy as np
import pytest

from demandcast import prep
from demandcast.catalog import ITEM, ITEMS, food_cost, margin


def test_critical_ratio():
    assert prep.critical_ratio(9, 3) == 0.75


def test_newsvendor_discrete_hand_example():
    # Demand 80 / 100 / 120 with probabilities 0.2 / 0.5 / 0.3. Cu = 9, Co = 3, so CR = 0.75.
    # CDF: 0.2, 0.7, 1.0. The smallest q with F(q) >= 0.75 is 120.
    assert prep.newsvendor_discrete(np.array([100, 80, 120]), np.array([0.5, 0.2, 0.3]), 9, 3) == 120
    # With Cu = Co, CR = 0.5 and the answer is the median, 100.
    assert prep.newsvendor_discrete(np.array([80, 100, 120]), np.array([0.2, 0.5, 0.3]), 3, 3) == 100


def test_newsvendor_normal_hand_example():
    # mu 100, sigma 20, CR 0.75: z = 0.6745, q = 113.49
    assert prep.newsvendor_normal(100, 20, 9, 3) == pytest.approx(113.49, abs=0.01)


def test_expected_outcome_hand_example():
    out = prep.expected_outcome(100, np.array([80, 100, 120]), cu=9, co=3)
    assert out["waste_units"] == pytest.approx(20 / 3)
    assert out["short_units"] == pytest.approx(20 / 3)
    assert out["waste_cost"] == pytest.approx(20)
    assert out["lost_margin"] == pytest.approx(60)
    assert out["fill_rate"] == pytest.approx(280 / 300)


def test_recipe_explosion_uses_yield():
    raw = prep.explode({"chicken_bowl": 10})
    assert raw["chicken"] == pytest.approx(10 * 0.15 / 0.72)
    assert raw["rice"] == pytest.approx(10 * 0.20 / 2.6)
    assert prep.prepared({"chicken_bowl": 10})["chicken"] == pytest.approx(1.5)


def test_recipe_explosion_adds_across_items():
    raw = prep.explode({"chicken_bowl": 4, "chicken_wrap": 6})
    assert raw["chicken"] == pytest.approx((4 * 0.15 + 6 * 0.12) / 0.72)
    assert raw["tortillas"] == pytest.approx(6)


def test_costs_and_ratios_are_sane():
    for it in ITEMS:
        assert 0 < food_cost(it) < it.price * 0.45, it.id
        cr = prep.critical_ratio(margin(it), food_cost(it))
        assert 0.55 < cr < 0.97, it.id


def test_prep_quantity_above_forecast_when_stockouts_cost_more():
    s = np.random.default_rng(0).normal(0, 1, 1000)
    cu, co = prep.item_costs("steak_bowl")
    assert prep.prep_quantity(40.0, s, cu, co) > 40
    assert prep.prep_quantity(0.0, s, cu, co) == 0


def test_prep_list_ingredients_match_explosion():
    s = np.random.default_rng(0).normal(0, 1, 500)
    fc = {"chicken_bowl": {"mean": 50, "lo": 35, "hi": 66}, "lemonade": {"mean": 30, "lo": 20, "hi": 41}}
    pl = prep.prep_list(fc, s, {"chicken_bowl": 44, "lemonade": 25})
    portions = {r["item"]: r["prep"] for r in pl["items"]}
    raw = prep.explode(portions)
    for r in pl["ingredients"]:
        assert r["raw"] == pytest.approx(raw[r["ingredient"]], abs=0.01)
    assert ITEM["chicken_bowl"].name in [r["name"] for r in pl["items"]]
    assert pl["expected_cost"] <= pl["gut_expected_cost"]
