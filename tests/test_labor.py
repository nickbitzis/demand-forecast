import numpy as np
import pytest

from demandcast import labor
from demandcast.catalog import N_SLOTS


def test_erlang_c_known_value():
    # A = 2 Erlangs, 3 servers: Erlang B = 0.2105, C = 0.4444
    assert labor.erlang_c(3, 2.0) == pytest.approx(4 / 9, abs=1e-4)
    assert labor.erlang_c(2, 2.0) == 1.0


def test_service_level_rises_with_staff():
    sl = [labor.service_level(c, 3.0, 40, 60) for c in range(4, 9)]
    assert sl == sorted(sl) and sl[-1] > 0.99


def test_staff_needed_floor_and_growth():
    assert labor.staff_needed(0, 40, 0.85, 60) == 1
    assert labor.staff_needed(60, 40, 0.85, 60) > labor.staff_needed(20, 40, 0.85, 60)


def test_staff_needed_meets_target():
    c = labor.staff_needed(50, 45, 0.85, 90)
    a = 50 / 900 * 45
    assert labor.service_level(c, a, 45, 90) >= 0.85
    assert c == 1 or labor.service_level(c - 1, a, 45, 90) < 0.85


def test_schedule_covers_requirements_with_valid_shifts():
    req = [0, 0] + [1] * 6 + [3] * 8 + [2] * 12 + [3] * 12 + [1] * 6 + [0, 0]
    assert len(req) == N_SLOTS
    shifts = labor.schedule(req)
    cover = np.zeros(N_SLOTS, int)
    for s, e in shifts:
        assert 16 <= e - s <= 32
        assert s % 2 == 0
        cover[s:e] += 1
    assert (cover >= np.array(req)).all()


def test_more_demand_needs_more_hours():
    shape = np.zeros(N_SLOTS)
    shape[2:46] = 1
    shape /= shape.sum()
    small = labor.plan({"chicken_bowl": 200 * shape, "lemonade": 80 * shape})
    big = labor.plan({"chicken_bowl": 900 * shape, "lemonade": 300 * shape})
    assert big["labor_hours"] > small["labor_hours"]
    assert set(small["requirements"]) == {"register", "grill", "line", "drinks"}


def test_truck_has_two_stations(art):
    t = art["locations"]["truck"]["labor"]
    if t["requirements"]:
        assert set(t["requirements"]) == {"register", "line"}
