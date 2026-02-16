"""Tests for premium metrics computed from aligned A/H/FX series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.core import compute_premium_metrics


def test_compute_premium_metrics_known_sample_hkdcny() -> None:
    """Premium percentage and log spread should match hand-calculated values."""

    a_idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    h_idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    fx_idx = pd.to_datetime(["2024-01-02", "2024-01-03"])

    a = pd.Series([10.0, 12.0, 14.0], index=a_idx)
    h = pd.Series([5.0, 6.0, 7.0], index=h_idx)
    fx = pd.Series([0.8, 0.75], index=fx_idx)

    out = compute_premium_metrics(
        a_price_cny=a,
        h_price_hkd=h,
        fx=fx,
        fx_quote="HKDCNY",
        window=2,
    )

    assert out.shape[0] == 2
    assert int(out["aligned_sample_size"].iloc[0]) == 2

    expected_pct = pd.Series([2.0, 14.0 / (6.0 * 0.75) - 1.0], index=out.index)
    expected_log = pd.Series(
        [np.log(12.0) - np.log(5.0 * 0.8), np.log(14.0) - np.log(6.0 * 0.75)],
        index=out.index,
    )

    np.testing.assert_allclose(out["premium_pct"].to_numpy(), expected_pct.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(out["log_spread"].to_numpy(), expected_log.to_numpy(), rtol=1e-12)


def test_compute_premium_metrics_cnyhkd_conversion_and_share_ratio() -> None:
    """CNYHKD quote inversion and share ratio adjustment should be applied correctly."""

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    a = pd.Series([12.0, 14.0], index=idx)
    h = pd.Series([5.0, 6.0], index=idx)
    fx_cnyhkd = pd.Series([1.25, 4.0 / 3.0], index=idx)

    out = compute_premium_metrics(
        a_price_cny=a,
        h_price_hkd=h,
        fx=fx_cnyhkd,
        fx_quote="CNYHKD",
        share_ratio=2.0,
        window=2,
    )

    expected_h_cny = h.to_numpy() * np.array([0.8, 0.75]) * 2.0
    expected_pct = a.to_numpy() / expected_h_cny - 1.0
    expected_log = np.log(a.to_numpy()) - np.log(expected_h_cny)

    np.testing.assert_allclose(out["premium_pct"].to_numpy(), expected_pct, rtol=1e-12)
    np.testing.assert_allclose(out["log_spread"].to_numpy(), expected_log, rtol=1e-12)
    assert np.isnan(float(out["rolling_zscore"].iloc[0]))
    assert float(out["rolling_percentile"].iloc[1]) == 1.0
