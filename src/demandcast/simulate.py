"""A month of kitchen operations under three prep policies.

Each evening the kitchen decides tomorrow's prep for every item at every site. Then the day's real demand arrives:
sold = min(prep, demand), the rest of the prep is thrown out at close, and unmet demand is a lost sale.

Policies:
- gut_feel: prep what sold on the same weekday last week.
- forecast_mean: prep the one-day-ahead forecast, rounded.
- newsvendor: prep the forecast's critical-ratio quantile (Cu / (Cu + Co)) from the conformal scenarios.

Waste is valued at food cost. Lost sales are reported as revenue and as margin (what the newsvendor trades off).
"""

from __future__ import annotations

import numpy as np

from . import intervals
from .catalog import ITEMS, LOCATIONS, food_cost, margin
from .prep import critical_ratio

POLICIES = ["gut_feel", "forecast_mean", "newsvendor"]


def account(prep: np.ndarray, demand: np.ndarray, price: np.ndarray, cost: np.ndarray) -> dict:
    """prep and demand are (series, days); price and cost are per series. Returns totals."""
    sold = np.minimum(prep, demand)
    waste = prep - sold
    lost = demand - sold
    marg = price - cost
    return {
        "prepped_units": float(prep.sum()),
        "demand_units": float(demand.sum()),
        "sold_units": float(sold.sum()),
        "waste_units": float(waste.sum()),
        "lost_units": float(lost.sum()),
        "waste_cost": float((waste * cost[:, None]).sum()),
        "lost_revenue": float((lost * price[:, None]).sum()),
        "lost_margin": float((lost * marg[:, None]).sum()),
        "revenue": float((sold * price[:, None]).sum()),
        "fill_rate": float(sold.sum() / max(demand.sum(), 1e-9)),
        "stockout_days": int(((lost > 0) & (demand > 0)).sum()),
    }


def policy_quantities(policy: str, forecast: np.ndarray, gut: np.ndarray, scores: np.ndarray, cr: np.ndarray):
    if policy == "gut_feel":
        return np.round(gut)
    if policy == "forecast_mean":
        return np.round(forecast)
    if policy == "newsvendor":
        out = np.zeros_like(forecast)
        for i in range(forecast.shape[0]):
            out[i] = np.ceil(intervals.quantile(forecast[i], scores, float(cr[i])) - 1e-9)
        return np.where(forecast > 0, out, 0)
    raise ValueError(policy)


def run(
    demand: np.ndarray,
    forecast: np.ndarray,
    gut: np.ndarray,
    item_index: np.ndarray,
    loc_index: np.ndarray,
    avail: np.ndarray,
    scores: np.ndarray,
    dates: list[str],
) -> dict:
    """All arrays (series, days) over the simulated month; series are item x location."""
    price = np.array([ITEMS[k].price for k in item_index])
    cost = np.array([food_cost(ITEMS[k]) for k in item_index])
    cr = np.array([critical_ratio(margin(ITEMS[k]), food_cost(ITEMS[k])) for k in item_index])
    demand = np.where(avail, np.nan_to_num(demand), 0)
    out = {"dates": [dates[0], dates[-1]], "days": len(dates), "policies": [], "by_location": []}
    for pol in POLICIES:
        q = np.where(avail, policy_quantities(pol, forecast, np.nan_to_num(gut), scores, cr), 0)
        tot = account(q, demand, price, cost)
        tot["total_cost"] = tot["waste_cost"] + tot["lost_margin"]
        out["policies"].append(
            {"policy": pol, **{k: round(v, 4) if isinstance(v, float) else v for k, v in tot.items()}}
        )
        for li, loc in enumerate(LOCATIONS):
            m = loc_index == li
            r = account(q[m], demand[m], price[m], cost[m])
            out["by_location"].append(
                {
                    "policy": pol,
                    "location": loc.id,
                    "waste_cost": round(r["waste_cost"], 2),
                    "lost_revenue": round(r["lost_revenue"], 2),
                    "lost_margin": round(r["lost_margin"], 2),
                    "fill_rate": round(r["fill_rate"], 4),
                }
            )
    return out
