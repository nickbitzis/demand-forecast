import numpy as np

from demandcast import intervals


def test_conformal_coverage_near_nominal_on_overdispersed_counts():
    rng = np.random.default_rng(5)
    mu = rng.uniform(3, 300, 20000)

    def draw(m):
        return rng.poisson(m * rng.lognormal(-0.02, 0.2, len(m)))

    cal = intervals.scores(draw(mu[:8000]), mu[:8000])
    lo, hi = intervals.interval(mu[8000:], cal)
    cov = intervals.coverage(draw(mu[8000:]), lo, hi)
    assert 0.88 <= cov <= 0.92


def test_score_quantile_has_finite_sample_adjustment():
    s = np.arange(1, 20, dtype=float)  # 19 scores
    assert intervals.score_quantile(s, 0.95) == 19.0
    assert intervals.score_quantile(s, 0.05) == 1.0
    assert intervals.score_quantile(s, 0.5) == 10.0


def test_interval_widens_with_the_forecast():
    s = np.random.default_rng(0).normal(0, 1, 500)
    lo, hi = intervals.interval(np.array([10.0, 400.0]), s)
    assert (hi - lo)[1] > (hi - lo)[0] * 5
    assert (lo >= 0).all()


def test_quantile_is_monotonic_in_p():
    s = np.random.default_rng(0).normal(0, 1, 500)
    qs = [intervals.quantile(np.array([50.0]), s, p)[0] for p in (0.1, 0.5, 0.75, 0.9)]
    assert qs == sorted(qs)


def test_backtest_interval_coverage_near_nominal(art):
    rows = [r for r in art["backtest"]["main"] if r["model"] == art["backtest"]["model"] and r["level"] == "bottom"]
    for r in rows:
        assert 0.8 <= r["coverage"] <= 0.97, r
