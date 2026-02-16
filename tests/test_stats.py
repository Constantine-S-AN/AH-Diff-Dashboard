"""Tests for stats diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.stats import adf_test, engle_granger_test, half_life_ar1, summary_score


def test_adf_identifies_stationary_series() -> None:
    """ADF output should indicate stationarity for a stable AR(1) process."""

    rng = np.random.default_rng(7)
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + rng.normal(0.0, 0.5)

    p_value, test_stat, used_lags = adf_test(pd.Series(x), max_lag=5)

    assert np.isfinite(test_stat)
    assert np.isfinite(p_value)
    assert used_lags >= 0
    assert p_value < 0.05


def test_half_life_is_positive_for_mean_reverting_spread() -> None:
    """Estimated half-life should be finite and positive on AR(1)-like spread."""

    rng = np.random.default_rng(11)
    n = 400
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.normal(0.0, 0.3)

    half_life = half_life_ar1(pd.Series(x))

    assert np.isfinite(half_life)
    assert half_life > 0.0


def test_engle_granger_detects_cointegration() -> None:
    """Engle-Granger residual ADF p-value should be small for linked series."""

    rng = np.random.default_rng(9)
    n = 600
    base = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    a = base + rng.normal(0.0, 0.15, n)
    h_fx = base * 1.1 + rng.normal(0.0, 0.15, n)

    p_value, beta, resid = engle_granger_test(
        log_A=np.log(pd.Series(a)),
        log_H_fx=np.log(pd.Series(h_fx)),
    )

    assert np.isfinite(p_value)
    assert np.isfinite(beta)
    assert isinstance(resid, pd.Series)
    assert resid.shape[0] > 0
    assert p_value < 0.05


def test_summary_score_rewards_better_statistics() -> None:
    """Summary score should be higher for stronger stationarity/cointegration profile."""

    stable_z = pd.Series(np.random.default_rng(1).normal(0.0, 1.0, 600))
    noisy_z = pd.Series(np.random.default_rng(2).normal(1.2, 2.0, 600))

    good = summary_score(
        adf_p_value=0.01,
        eg_p_value=0.02,
        half_life_days=18.0,
        zscore_series=stable_z,
    )
    bad = summary_score(
        adf_p_value=0.70,
        eg_p_value=0.80,
        half_life_days=float("inf"),
        zscore_series=noisy_z,
    )

    assert 0.0 <= good <= 100.0
    assert 0.0 <= bad <= 100.0
    assert good > bad
