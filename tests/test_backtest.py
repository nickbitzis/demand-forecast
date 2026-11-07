import numpy as np
import pandas as pd
import pytest

from demandcast import backtest


def test_metrics_hand_example():
    df = pd.DataFrame({"y": [10.0, 20.0, 30.0], "yhat": [12.0, 18.0, 33.0], "scale": [2.0, 4.0, np.nan]})
    m = backtest.metrics(df)
    assert m["wape"] == pytest.approx(7 / 60)
    assert m["bias"] == pytest.approx(3 / 60)
    assert m["mase"] == pytest.approx(np.mean([2 / 2, 2 / 4]))


def test_mase_scale_is_the_mean_weekly_change():
    Y = np.tile(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]), 6)[None] + np.repeat(np.arange(6) * 7.0, 7)
    assert backtest._mase_scale(Y, 41)[0] == pytest.approx(7.0)


def test_origins_leave_room_for_the_longest_horizon(small_ds):
    cfg = backtest.BacktestConfig(origins=5, step=14)
    o = backtest.origin_days(small_ds, cfg)
    assert o == sorted(o) and o[-1] == small_ds.n_hist - 1 - 28
    assert np.diff(o).tolist() == [14] * 4


def test_intervals_use_only_earlier_origins():
    rows = []
    for k, o in enumerate([100, 114, 128]):
        for d in range(o + 1, o + 8):
            rows += [
                {
                    "origin": o,
                    "H": 7,
                    "model": "gbm_bottom_up",
                    "level": "bottom",
                    "day": d,
                    "y": 10.0 + (k == 2) * 100,
                    "yhat": 10.0,
                }
                for _ in range(10)
            ]
    df = backtest.add_intervals(pd.DataFrame(rows), [100, 114, 128], warmup=1)
    assert df[df["origin"] == 100]["lo"].isna().all()
    # The huge errors at origin 128 must not widen origin 128's own interval.
    w = df[df["origin"] == 128]
    assert (w["hi"] - w["lo"]).max() < 1e-9


def test_gate_and_improvement(art):
    bt = art["backtest"]
    assert bt["gate"]["passed"] and bt["gate"]["horizon"] == 7
    row = next(r for r in bt["main"] if r["level"] == "bottom" and r["H"] == 7 and r["model"] == bt["model"])
    assert row["improvement_vs_snaive"] > 0.2


def test_results_cover_every_horizon_model_and_level(art):
    bt = art["backtest"]
    keys = {(r["level"], r["H"], r["model"]) for r in bt["main"]}
    for lvl in ("bottom", "location", "total"):
        for H in (1, 7, 28):
            for m in ("seasonal_naive", "moving_average", "gbm_bottom_up", "gbm_top_down", "gbm_mint_lite"):
                assert (lvl, H, m) in keys
    assert {r["location"] for r in bt["by_location"]} == {"lodo", "cap-hill", "highlands", "truck"}
    assert {r["class"] for r in bt["by_class"]} == {"bowl", "wrap", "side", "drink"}
    assert {r["day_type"] for r in bt["by_event"]} == {"event day", "normal day"}


def test_gbm_beats_seasonal_naive_on_event_days(art):
    for r in art["backtest"]["by_event"]:
        if r["day_type"] == "event day":
            assert r["gbm"] < r["snaive"]


def test_cold_start_helps_the_new_item_at_28_days(art):
    r = next(x for x in art["backtest"]["cold_start"] if x["segment"].startswith("New") and x["H"] == 28)
    assert r["gbm_cold_start"] < r["gbm_no_cold_start"]
    assert r["gbm_cold_start"] < r["snaive"]
