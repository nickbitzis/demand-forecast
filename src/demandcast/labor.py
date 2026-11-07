"""Staffing from 15-minute forecasts: Erlang C per station with a service-level target, then a shift suggestion.

Each station is a queue of tickets. In a slot, the arrival rate is tickets / 900 s and the mean service time is the
station's work for those tickets divided by their count. The station needs the fewest staff c with
c > offered load A and

    P(wait <= T) = 1 - C(c, A) * exp(-(c - A) * T / S) >= target

where C is the Erlang C probability of waiting. Every station keeps at least one person while the store is open.
Shifts are 4 to 8 hours on the half hour, laid greedily over the staffing curve until every slot is covered.
"""

from __future__ import annotations

import math

import numpy as np

from .catalog import CLOSE_SLOT, ITEM, OPEN_SLOT, UNITS_PER_TICKET, slot_label

# Seconds of work per item unit at each station, by item class.
STATIONS = {
    "register": {"_ticket": 35.0},
    "grill": {"bowl": 22.0, "wrap": 28.0},
    "line": {"bowl": 38.0, "wrap": 48.0, "side": 14.0},
    "drinks": {"drink": 18.0, "side": 8.0},
}
TARGET = {"register": (0.85, 60.0), "grill": (0.85, 120.0), "line": (0.85, 90.0), "drinks": (0.85, 90.0)}
SLOT_SECONDS = 900.0
MIN_SHIFT, MAX_SHIFT = 16, 32  # slots: 4 and 8 hours


def erlang_c(c: int, a: float) -> float:
    """Probability an arrival waits, for c servers and offered load a (Erlangs)."""
    if c <= a:
        return 1.0
    term, s = 1.0, 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    top = term * a / c * (c / (c - a))
    return top / (s + top)


def service_level(c: int, a: float, service_s: float, target_wait_s: float) -> float:
    if c <= a:
        return 0.0
    return 1.0 - erlang_c(c, a) * math.exp(-(c - a) * target_wait_s / service_s)


def staff_needed(arrivals_per_slot: float, service_s: float, sl: float, wait_s: float, floor: int = 1) -> int:
    if arrivals_per_slot <= 0 or service_s <= 0:
        return floor
    a = arrivals_per_slot / SLOT_SECONDS * service_s
    c = max(floor, int(math.floor(a)) + 1)
    while service_level(c, a, service_s, wait_s) < sl:
        c += 1
    return c


def station_load(slot_units: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """item id -> (slots,) units. Returns station -> (tickets per slot, mean service seconds per ticket)."""
    n = len(next(iter(slot_units.values())))
    units = np.zeros(n)
    work = {s: np.zeros(n) for s in STATIONS}
    for item_id, u in slot_units.items():
        units += u
        cls = ITEM[item_id].cls
        for st, secs in STATIONS.items():
            work[st] += u * secs.get(cls, 0.0)
    tickets = units / UNITS_PER_TICKET
    work["register"] = tickets * STATIONS["register"]["_ticket"]
    out = {}
    for st in STATIONS:
        with np.errstate(invalid="ignore", divide="ignore"):
            svc = np.where(tickets > 0, work[st] / np.maximum(tickets, 1e-9), 0.0)
        out[st] = (tickets, svc)
    return out


def requirements(slot_units: dict[str, np.ndarray]) -> dict[str, list[int]]:
    loads = station_load(slot_units)
    out = {}
    for st, (tickets, svc) in loads.items():
        sl, wait = TARGET[st]
        req = []
        for s in range(len(tickets)):
            if s < OPEN_SLOT or s >= CLOSE_SLOT:
                req.append(0)
            else:
                req.append(staff_needed(float(tickets[s]), float(svc[s]), sl, wait))
        out[st] = req
    return out


def schedule(req: list[int]) -> list[tuple[int, int]]:
    """Greedy shifts (start slot, end slot) covering a requirement curve. Shifts start on the half hour."""
    need = np.array(req, dtype=int)
    cover = np.zeros_like(need)
    shifts = []
    last = int(np.max(np.nonzero(need)[0])) + 1 if need.any() else 0
    while True:
        gap = np.nonzero(cover < need)[0]
        if len(gap) == 0:
            break
        s = int(gap[0]) - int(gap[0]) % 2
        # Run on while this extra person is still needed, between 4 and 8 hours, not past close.
        e = s + MIN_SHIFT
        while e < min(s + MAX_SHIFT, last) and cover[e] < need[e]:
            e += 1
        e = min(e + (e - s) % 2, max(last, s + 1))
        if e - s < MIN_SHIFT:  # near close: start earlier instead of a short shift
            s = max(0, e - MIN_SHIFT)
        cover[s:e] += 1
        shifts.append((s, e))
    return shifts


TRUCK_MERGE = {"grill": "line", "drinks": "register"}  # two stations in a food truck


def plan(slot_units: dict[str, np.ndarray], mobile: bool = False) -> dict:
    req = requirements(slot_units)
    if mobile:
        # One person works the window and drinks, the rest cook and assemble: merge loads before staffing.
        loads = station_load(slot_units)
        merged = {}
        for st in ("register", "line"):
            parts = [st] + [k for k, v in TRUCK_MERGE.items() if v == st]
            tickets = loads[st][0]
            svc = sum(loads[k][1] for k in parts)
            sl, wait = TARGET[st]
            merged[st] = [
                0 if s < OPEN_SLOT or s >= CLOSE_SLOT else staff_needed(float(tickets[s]), float(svc[s]), sl, wait)
                for s in range(len(tickets))
            ]
        req = merged
    out = {"requirements": req, "shifts": [], "slots": [slot_label(s) for s in range(len(req["register"]))]}
    total = 0.0
    for st, r in req.items():
        for s, e in schedule(r):
            out["shifts"].append({"station": st, "start": slot_label(s), "end": slot_label(e), "hours": (e - s) / 4})
            total += (e - s) / 4
    out["labor_hours"] = total
    out["peak_staff"] = int(np.max(np.sum([np.array(r) for r in req.values()], axis=0)))
    return out
