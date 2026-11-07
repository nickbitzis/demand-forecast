"""Supplier orders from ingredient forecasts: lead times, order days, case packs, and shelf life.

Day 0 is tomorrow. An order placed at the end of day p (p = -1 is tonight) arrives the morning of day p + lead.
For each order opportunity, the order covers usage from its arrival until the next order's arrival, plus safety
stock for the uncertainty from now until then, minus the stock projected to be on hand at arrival:

    qty = usage[a, a') + z * sqrt(sum var[p+1, a')) - projected_on_hand(a)

It is then capped at what can be used within the ingredient's shelf life after arrival, and rounded to whole
cases: up, unless rounding up would buy more than the shelf-life cap and a smaller whole number of cases exists.
A single case that is still more than the cap is ordered with a flag, since the supplier sells nothing smaller.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from .catalog import ING, SUPPLIER, SUPPLIERS

SERVICE_Z = 1.65  # about 95% cycle service level


def opportunities(today: date, weekdays: tuple[int, ...], horizon: int) -> list[int]:
    """Order placement days p in [-1, horizon) whose weekday is an order day. Day -1 is today."""
    return [p for p in range(-1, horizon) if (today + timedelta(days=p + 1)).weekday() in weekdays]


def plan_ingredient(
    usage: np.ndarray,
    sd: np.ndarray,
    on_hand: float,
    case_pack: float,
    shelf_life: int,
    lead: int,
    opps: list[int],
    max_orders: int = 3,
) -> list[dict]:
    n = len(usage)
    orders: list[dict] = []
    arrivals: list[tuple[int, float]] = []
    for i, p in enumerate(opps[:max_orders]):
        a = p + lead
        nxt = opps[i + 1] + lead if i + 1 < len(opps) else a + 7
        if a >= n:
            break
        b = min(nxt, n)
        projected = on_hand - usage[:a].sum() + sum(q for t, q in arrivals if t <= a)
        projected_pos = max(projected, 0.0)
        ss = SERVICE_Z * math.sqrt(float((sd[max(p + 1, 0) : b] ** 2).sum()))
        need = float(usage[a:b].sum()) + ss - projected_pos
        # Shelf life past the end of the forecast does not bind (dry goods keep for months).
        cap = float(usage[a : a + shelf_life].sum()) - projected_pos if a + shelf_life <= n else math.inf
        qty = min(need, cap) if cap > 0 else 0.0
        flag = "projected to run out before this delivery" if projected < 0 else ""
        if qty <= 0:
            cases = 0
        else:
            cases = math.ceil(qty / case_pack - 1e-9)
            if cases * case_pack > cap and cases > 1:
                cases -= 1
                flag = flag or "rounded down to stay within shelf life"
            elif cases * case_pack > cap:
                flag = flag or "one case is more than can be used within shelf life"
        units = cases * case_pack
        if units > 0:
            arrivals.append((a, units))
        orders.append(
            {
                "place_day": p,
                "arrive_day": a,
                "cover_days": b - a,
                "need": round(need, 2),
                "projected_on_hand": round(projected, 2),
                "safety_stock": round(ss, 2),
                "shelf_cap": round(cap, 2) if math.isfinite(cap) else None,
                "cases": cases,
                "qty": round(units, 2),
                "flag": flag,
            }
        )
    return orders


def plan(today: date, usage: dict[str, np.ndarray], sd: dict[str, np.ndarray], on_hand: dict[str, float]) -> dict:
    """Orders for one location for every supplier. Returns lines and the next order per supplier."""
    lines = []
    for ing_id, u in usage.items():
        ing = ING[ing_id]
        sup = SUPPLIER[ing.supplier]
        opps = opportunities(today, sup.order_weekdays, 8)
        for o in plan_ingredient(
            u, sd[ing_id], on_hand.get(ing_id, 0.0), ing.case_pack, ing.shelf_life_days, sup.lead_days, opps
        ):
            o.update(
                {
                    "ingredient": ing_id,
                    "name": ing.name,
                    "unit": ing.unit,
                    "supplier": sup.id,
                    "supplier_name": sup.name,
                    "case_pack": ing.case_pack,
                    "order_date": (today + timedelta(days=o["place_day"] + 1)).isoformat(),
                    "delivery_date": (today + timedelta(days=o["arrive_day"] + 1)).isoformat(),
                    "cost": round(o["qty"] * ing.cost, 2),
                    "on_hand_now": round(on_hand.get(ing_id, 0.0), 2),
                }
            )
            lines.append(o)
    nxt = []
    for sup in SUPPLIERS:
        mine = [x for x in lines if x["supplier"] == sup.id]
        if not mine:
            continue
        first = min(x["order_date"] for x in mine)
        sel = [x for x in mine if x["order_date"] == first and x["cases"] > 0]
        nxt.append(
            {
                "supplier": sup.id,
                "supplier_name": sup.name,
                "order_date": first,
                "delivery_date": sel[0]["delivery_date"] if sel else None,
                "lead_days": sup.lead_days,
                "order_days": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in sup.order_weekdays],
                "lines": sel,
                "total": round(sum(x["cost"] for x in sel), 2),
            }
        )
    return {"next": nxt, "lines": lines}
