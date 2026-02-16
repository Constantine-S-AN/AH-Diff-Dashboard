"""Tests for threshold-based spread position switching strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.backtest import run_pairs_strategy


def test_pairs_strategy_switches_positions_on_thresholds() -> None:
    """Strategy should switch short/flat/long/flat when z-score crosses thresholds."""

    dates = pd.bdate_range("2024-01-01", periods=10)

    # Construct log spread directly through price mapping: A = exp(spread) * H_CNY.
    log_spread = np.array([0.0, 0.0, 0.0, 3.0, 2.0, -3.0, -2.0, -2.0, -2.0, -2.0])
    h_cny = np.full_like(log_spread, 100.0)
    a_close = np.exp(log_spread) * h_cny

    frame = pd.DataFrame(
        {
            "date": dates,
            "a_close": a_close,
            "h_cny": h_cny,
        }
    )

    result = run_pairs_strategy(
        frame,
        entry=1.0,
        exit=0.5,
        z_window=3,
        cost_bps=10.0,
    )

    daily = result.daily

    assert float(daily.loc[dates[3], "spread_position"]) == -1.0
    assert float(daily.loc[dates[4], "spread_position"]) == 0.0
    assert float(daily.loc[dates[5], "spread_position"]) == 1.0
    assert float(daily.loc[dates[6], "spread_position"]) == 0.0

    assert "a_weight" in daily.columns
    assert "h_weight" in daily.columns
    assert "pnl_gross" in daily.columns
    assert "pnl_net" in daily.columns
    assert "turnover" in daily.columns
    assert "curve_gross" in daily.columns
    assert "curve_net" in daily.columns

    assert np.isfinite(result.max_drawdown_gross)
    assert np.isfinite(result.max_drawdown_net)
    assert (daily["turnover"] >= 0.0).all()
