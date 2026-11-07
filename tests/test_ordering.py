from datetime import date

import numpy as np

from demandcast import ordering

SAT = date(2026, 11, 7)


def test_order_opportunities_follow_order_days():
    # Today is Saturday. Mon/Wed/Fri: Monday is day 1 (placed at the end of Monday), then 3, then 5.
    assert ordering.opportunities(SAT, (0, 2, 4), 7)[:3] == [1, 3, 5]
    # Saturday orders allowed: tonight counts as day -1.
    assert ordering.opportunities(SAT, (5,), 8) == [-1, 6]


def test_hand_worked_order_rounds_up_to_case_pack():
    # 10 a day, 25 on hand, order tonight (arrives day 0) and on day 2 (arrives day 3): cover days 0 to 2.
    # Need 30 - 25 = 5, one case of 12.
    o = ordering.plan_ingredient(np.full(14, 10.0), np.zeros(14), 25, 12, 30, 1, [-1, 2])[0]
    assert o["arrive_day"] == 0 and o["cover_days"] == 3
    assert o["need"] == 5.0 and o["cases"] == 1 and o["qty"] == 12


def test_second_order_counts_the_first_delivery():
    orders = ordering.plan_ingredient(np.full(14, 10.0), np.zeros(14), 25, 12, 30, 1, [-1, 2, 5])
    # Before the second arrival (day 3): 25 + 12 - 30 = 7 on hand. Cover days 3 to 5: 30, need 23 -> 2 cases.
    assert orders[1]["projected_on_hand"] == 7.0
    assert orders[1]["cases"] == 2


def test_lead_time_moves_the_arrival():
    o = ordering.plan_ingredient(np.full(20, 5.0), np.zeros(20), 40, 10, 100, 2, [1, 8])[0]
    assert o["arrive_day"] == 3 and o["cover_days"] == 7
    # Stock at arrival: 40 - 3 * 5 = 25. Need 35 - 25 = 10, one case.
    assert o["cases"] == 1


def test_safety_stock_adds_to_the_order():
    a = ordering.plan_ingredient(np.full(14, 10.0), np.zeros(14), 0, 1, 30, 1, [-1, 2])[0]
    b = ordering.plan_ingredient(np.full(14, 10.0), np.full(14, 3.0), 0, 1, 30, 1, [-1, 2])[0]
    assert b["qty"] > a["qty"]
    assert b["safety_stock"] == round(1.65 * np.sqrt(3 * 9.0), 2)


def test_shelf_life_caps_and_rounds_down():
    # Need 70 over a week, but only 2 days of shelf life: cap 20. Case of 8: 3 cases (24) > 20, so 2 cases.
    o = ordering.plan_ingredient(np.full(14, 10.0), np.zeros(14), 0, 8, 2, 1, [-1, 6])[0]
    assert o["shelf_cap"] == 20 and o["cases"] == 2
    assert "shelf life" in o["flag"]


def test_single_case_over_shelf_life_is_flagged():
    o = ordering.plan_ingredient(np.full(14, 1.0), np.zeros(14), 0, 22, 3, 1, [-1, 6])[0]
    assert o["cases"] == 1 and "one case" in o["flag"]


def test_run_out_before_delivery_is_flagged():
    o = ordering.plan_ingredient(np.full(14, 10.0), np.zeros(14), 5, 10, 30, 1, [2, 5])[0]
    assert o["projected_on_hand"] < 0 and o["flag"].startswith("projected")


def test_plan_dates_and_suppliers(art):
    for loc in art["locations"].values():
        for o in loc["orders"]["next"]:
            for x in o["lines"]:
                assert x["qty"] == x["cases"] * x["case_pack"]
                d = date.fromisoformat(x["order_date"]).weekday()
                assert ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] in o["order_days"]
                lead = (date.fromisoformat(x["delivery_date"]) - date.fromisoformat(x["order_date"])).days
                assert lead == o["lead_days"]


def test_long_shelf_life_does_not_cap():
    o = ordering.plan_ingredient(np.full(14, 1.0), np.zeros(14), 3, 25, 365, 2, [1, 8])[0]
    assert o["shelf_cap"] is None and o["cases"] == 1 and o["flag"] == ""
