"""Tests for structural-break diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.stats import cusum_stability_test, detect_structural_breaks


def test_detect_structural_breaks_finds_mean_shift() -> None:
    """Detector should find a breakpoint near a synthetic regime shift."""

    idx = pd.bdate_range("2023-01-02", periods=240)
    rng = np.random.default_rng(42)
    first = rng.normal(0.0, 0.05, size=120)
    second = rng.normal(0.6, 0.05, size=120)
    series = pd.Series(np.concatenate([first, second]), index=idx)

    result = detect_structural_breaks(
        series,
        window=30,
        min_distance=10,
        alpha=1e-4,
        zscore_threshold=2.5,
    )

    assert not result.breakpoints.empty
    assert {"break_date", "confidence", "p_value", "shift_zscore", "mean_shift"} == set(
        result.breakpoints.columns
    )
    assert result.breakpoints["confidence"].between(0.0, 1.0).all()

    target_date = idx[120]
    distances = np.abs((result.breakpoints["break_date"] - target_date).dt.days.to_numpy())
    assert distances.min() <= 15


def test_detect_structural_breaks_returns_empty_on_stable_series() -> None:
    """Detector should return no breakpoint for a stable random series with strict thresholds."""

    idx = pd.bdate_range("2024-01-01", periods=220)
    rng = np.random.default_rng(8)
    series = pd.Series(rng.normal(0.0, 0.08, size=220), index=idx)

    result = detect_structural_breaks(
        series,
        window=40,
        min_distance=10,
        alpha=1e-6,
        zscore_threshold=4.5,
    )

    assert result.breakpoints.empty


def test_cusum_stability_test_returns_finite_values_for_long_series() -> None:
    """CUSUM helper should produce finite test outputs on valid sample size."""

    idx = pd.bdate_range("2022-01-03", periods=300)
    rng = np.random.default_rng(12)
    series = pd.Series(rng.normal(0.0, 0.1, size=300), index=idx)

    stat, p_value = cusum_stability_test(series)

    assert np.isfinite(stat)
    assert np.isfinite(p_value)
    assert 0.0 <= p_value <= 1.0
