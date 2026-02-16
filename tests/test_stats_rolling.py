"""Tests for rolling Engle-Granger diagnostics and stability metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.stats import rolling_engle_granger, rolling_stability_metrics


def test_rolling_engle_granger_outputs_expected_columns() -> None:
    """Rolling EG should return deterministic columns and non-empty windows."""

    idx = pd.bdate_range("2022-01-03", periods=420)
    rng = np.random.default_rng(7)

    x_shock = rng.normal(0.0, 0.02, size=len(idx))
    x = np.cumsum(x_shock)
    noise = rng.normal(0.0, 0.04, size=len(idx))
    y = 1.25 * x + noise

    log_a = pd.Series(y + 12.0, index=idx)
    log_hfx = pd.Series(x + 10.0, index=idx)

    rolling = rolling_engle_granger(log_a, log_hfx, window=120, step=21)

    assert not rolling.empty
    assert list(rolling.columns) == [
        "window_start",
        "window_end",
        "p_value",
        "beta",
        "resid_std",
        "n_obs",
    ]
    assert (rolling["n_obs"] == 120).all()
    assert rolling["p_value"].notna().any()
    assert rolling["beta"].notna().any()


def test_rolling_stability_metrics_values() -> None:
    """Stability metrics should match expected pass-rate/variance/drift values."""

    frame = pd.DataFrame(
        {
            "window_start": pd.bdate_range("2024-01-01", periods=4),
            "window_end": pd.bdate_range("2024-02-01", periods=4),
            "p_value": [0.01, 0.02, 0.10, 0.03],
            "beta": [1.00, 1.02, 0.98, 1.01],
            "resid_std": [0.50, 0.55, 0.60, 0.65],
            "n_obs": [252, 252, 252, 252],
        }
    )

    metrics = rolling_stability_metrics(frame, p_value_threshold=0.05)

    assert metrics.n_windows == 4
    assert np.isclose(metrics.p_value_pass_rate, 0.75)
    assert np.isclose(metrics.beta_variance, np.var([1.00, 1.02, 0.98, 1.01], ddof=0))
    assert np.isclose(metrics.resid_std_drift, (0.65 - 0.50) / 0.50)
    assert 0.0 <= metrics.stability_score <= 100.0
