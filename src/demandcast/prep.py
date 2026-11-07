"""Prep planning: newsvendor quantities per menu item and recipe explosion to ingredients.

Each item is prepped once a day and what is left at close is thrown out, so it is a single-period newsvendor:

    cost of one unit too many  (overage)  Co = food cost of the portion
    cost of one unit too few   (underage) Cu = lost margin = price - food cost
    prep the smallest q with  P(demand <= q) >= Cu / (Cu + Co)

The demand distribution is the forecast plus the conformal score scenarios (`intervals.samples`), so the
quantile, expected waste, and expected lost sales all come from the same scenarios.
"""

from __future__ import annotations

import math

import numpy as np

from . import intervals
from .catalog import ING, ITEM, ITEMS, food_cost, margin, raw_per_portion


def critical_ratio(cu: float, co: float) -> float:
    return cu / (cu + co)


def newsvendor_discrete(values: np.ndarray, probs: np.ndarray, cu: float, co: float) -> float:
    """Smallest demand value whose cumulative probability reaches the critical ratio."""
    order = np.argsort(values)
    v, p = np.asarray(values, float)[order], np.asarray(probs, float)[order]
    cdf = np.cumsum(p) / p.sum()
    cr = critical_ratio(cu, co)
    return float(v[np.searchsorted(cdf, cr - 1e-12)])


def newsvendor_normal(mu: float, sigma: float, cu: float, co: float) -> float:
    from statistics import NormalDist

    return mu + sigma * NormalDist().inv_cdf(critical_ratio(cu, co))


def expected_outcome(q: float, scenarios: np.ndarray, cu: float, co: float) -> dict:
    d = np.asarray(scenarios, float)
    over = np.maximum(q - d, 0)
    under = np.maximum(d - q, 0)
    return {
        "waste_units": float(over.mean()),
        "short_units": float(under.mean()),
        "waste_cost": float(over.mean() * co),
        "lost_margin": float(under.mean() * cu),
        "fill_rate": float(np.minimum(q, d).sum() / max(d.sum(), 1e-9)),
    }


def prep_quantity(yhat: float, scores: np.ndarray, cu: float, co: float) -> int:
    """Newsvendor quantile from the forecast's scenarios, rounded up to whole portions."""
    if yhat <= 0:
        return 0
    q = float(intervals.quantile(np.array([yhat]), scores, critical_ratio(cu, co))[0])
    return int(math.ceil(q - 1e-9))


def item_costs(item_id: str) -> tuple[float, float]:
    """(Cu, Co) for an item: lost margin and food cost per portion."""
    it = ITEM[item_id]
    return margin(it), food_cost(it)


def explode(portions: dict[str, float]) -> dict[str, float]:
    """Menu item portions to purchase units of each ingredient, through recipe quantities and yields."""
    out: dict[str, float] = {}
    for item_id, n in portions.items():
        for ing, q in raw_per_portion(ITEM[item_id]).items():
            out[ing] = out.get(ing, 0.0) + q * n
    return out


def prepared(portions: dict[str, float]) -> dict[str, float]:
    """Menu item portions to prepared quantities (cooked chicken, cooked rice) before yield."""
    out: dict[str, float] = {}
    for item_id, n in portions.items():
        for ing, q in ITEM[item_id].recipe.items():
            out[ing] = out.get(ing, 0.0) + q * n
    return out


def prep_list(forecast: dict[str, dict], scores: np.ndarray, last_week: dict[str, float]) -> dict:
    """Tomorrow's prep list for one location.

    forecast: item id -> {"mean", "lo", "hi"}. scores: calibration scores for one-day-ahead item forecasts.
    """
    rows = []
    for it in ITEMS:
        f = forecast.get(it.id)
        if f is None or f["mean"] <= 0:
            continue
        cu, co = item_costs(it.id)
        q = prep_quantity(f["mean"], scores, cu, co)
        sc = intervals.samples(f["mean"], scores)
        exp = expected_outcome(q, sc, cu, co)
        gut = last_week.get(it.id, 0.0)
        gexp = expected_outcome(gut, sc, cu, co)
        rows.append(
            {
                "item": it.id,
                "name": it.name,
                "class": it.cls,
                "forecast": round(f["mean"], 1),
                "lo": round(f["lo"], 1),
                "hi": round(f["hi"], 1),
                "price": it.price,
                "food_cost": round(co, 2),
                "margin": round(cu, 2),
                "critical_ratio": round(critical_ratio(cu, co), 3),
                "prep": q,
                "gut_feel": int(round(gut)),
                "expected_waste_cost": round(exp["waste_cost"], 2),
                "expected_lost_margin": round(exp["lost_margin"], 2),
                "expected_fill_rate": round(exp["fill_rate"], 4),
                "gut_expected_cost": round(gexp["waste_cost"] + gexp["lost_margin"], 2),
            }
        )
    portions = {r["item"]: r["prep"] for r in rows}
    prep_ing = prepared(portions)
    raw = explode(portions)
    ingredients = [
        {
            "ingredient": k,
            "name": ING[k].name,
            "unit": ING[k].unit,
            "prepared": round(prep_ing[k], 2),
            "raw": round(raw[k], 2),
            "yield": ING[k].yield_,
        }
        for k in sorted(raw, key=lambda k: -raw[k] * ING[k].cost)
    ]
    return {
        "items": rows,
        "ingredients": ingredients,
        "expected_cost": round(sum(r["expected_waste_cost"] + r["expected_lost_margin"] for r in rows), 2),
        "gut_expected_cost": round(sum(r["gut_expected_cost"] for r in rows), 2),
    }
