import numpy as np
import pytest

from demandcast.features import _roll_mean, design, lag_features, masked, training_rows
from demandcast.forecast import base_forecasts, fit
from demandcast.models import GBM, seasonal_naive

from .conftest import SMALL_GBM


def test_roll_mean_skips_nan():
    Y = np.array([[1.0, np.nan, 3.0, 5.0]])
    r = _roll_mean(Y, 2)
    assert r[0, 0] == 1.0 and r[0, 1] == 1.0 and r[0, 2] == 3.0 and r[0, 3] == 4.0


def test_lags_are_at_least_shift_days_old():
    Y = np.arange(60, dtype=float)[None, :]
    days = np.array([50])
    f = lag_features(Y, days, 7)
    assert f["level"][0, 0] == np.mean(np.arange(43 - 27, 44))
    assert f["lag_recent"][0, 0] * max(f["level"][0, 0], 1) == pytest.approx(43)
    assert f["lag_week"][0, 0] * max(f["level"][0, 0], 1) == pytest.approx(43)
    f28 = lag_features(Y, days, 28)
    assert f28["lag_week"][0, 0] * max(f28["level"][0, 0], 1) == pytest.approx(22)


def test_masking_hides_everything_after_the_cutoff(small_panels):
    P = small_panels.bottom
    m = masked(P, 300)
    assert np.isnan(m[:, 301:]).all()
    assert np.array_equal(np.isnan(m[:, :301]), np.isnan(P.Y[:, :301]))


def test_training_rows_stop_at_the_cutoff(small_panels):
    P = small_panels.location
    X, y, cols, _ = training_rows(P, 7, 200)
    n_obs = (~np.isnan(P.Y[:, 36 + 7 : 201])).sum()
    assert len(y) <= n_obs


def test_leakage_guard_future_values_do_not_move_forecasts(small_ds, small_panels):
    """Perturb every value after the cutoff. Training and forecasts at the cutoff must not change."""
    P = small_panels
    c = small_ds.n_hist - 40
    days = np.arange(c + 1, c + 8)
    m1 = fit(P, c, 7, SMALL_GBM)
    f1 = base_forecasts(m1, P, days)
    saved = {k: P[k].Y.copy() for k in ("bottom", "location", "total")}
    try:
        rng = np.random.default_rng(0)
        for k in saved:
            Y = P[k].Y
            fut = ~np.isnan(Y[:, c + 1 :])
            Y[:, c + 1 :][fut] = Y[:, c + 1 :][fut] * rng.uniform(0, 5, fut.sum())
        m2 = fit(P, c, 7, SMALL_GBM)
        f2 = base_forecasts(m2, P, days)
    finally:
        for k, v in saved.items():
            P[k].Y[:] = v
    for k in ("bottom", "location", "total"):
        assert np.allclose(f1[k], f2[k])


def test_seasonal_naive_ignores_the_future(small_panels):
    P = small_panels.location
    c = 300
    days = np.arange(c + 1, c + 15)
    a = seasonal_naive(P, c, days)
    saved = P.Y.copy()
    P.Y[:, c + 1 :] = 9999
    try:
        b = seasonal_naive(P, c, days)
    finally:
        P.Y[:] = saved
    assert np.array_equal(a, b)


def test_model_refuses_targets_beyond_its_shift(small_panels):
    m = GBM(small_panels.total, 7, 300, SMALL_GBM)
    with pytest.raises(AssertionError):
        m.predict(np.arange(301, 309))


def test_closed_days_forecast_zero(small_ds, small_panels):
    m = GBM(small_panels.location, 28, small_ds.n_hist - 1, SMALL_GBM)
    days = np.arange(small_ds.n_hist, small_ds.n_hist + 28)
    f = m.predict(days)
    assert (f[~small_panels.location.avail[:, days]] == 0).all()
    assert (f[small_panels.location.avail[:, days]] > 0).all()


def test_design_columns_include_calendar_weather_event_and_lags(small_panels):
    X, cols, cat = design(small_panels.bottom, np.array([300]), 1, 299)
    for c in ("dow", "holiday", "temp", "precip", "snow", "pressure", "promo", "lag_week", "roll7", "level"):
        assert c in cols
    assert X.shape == (small_panels.bottom.Y.shape[0], len(cols))
    assert cat[cols.index("loc")] and not cat[cols.index("temp")]
