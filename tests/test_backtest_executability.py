"""Tests for executability constraints in backtest module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.backtest import (
    ExecutabilityConfig,
    apply_executability_constraints,
    run_pairs_strategy,
)


def test_apply_executability_constraints_blocks_short_a() -> None:
    """Disallowing short A should clip negative A targets and record misses."""

    idx = pd.bdate_range("2024-01-01", periods=5)
    target_a = pd.Series([-1.0, -1.0, 0.0, 1.0, 1.0], index=idx)
    target_h = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0], index=idx)

    constrained, metrics = apply_executability_constraints(
        target_a,
        target_h,
        config=ExecutabilityConfig(
            enforce_a_share_t1=True,
            allow_short_a=False,
            allow_short_h=True,
        ),
    )

    assert (constrained["a_weight"] >= 0.0).all()
    assert float(constrained.iloc[0]["h_weight"]) == 1.0
    assert int(metrics.missed_trades) >= 1
    assert int(metrics.constraint_violation_count) >= 1
    assert 0.0 <= metrics.executability_score < 100.0


def test_pairs_strategy_respects_executability_settings() -> None:
    """Strategy should switch to single-leg exposure when short A is forbidden."""

    dates = pd.bdate_range("2024-01-01", periods=10)
    log_spread = np.array([0.0, 0.0, 0.0, 3.0, 2.0, -3.0, -2.0, -2.0, -2.0, -2.0])
    h_cny = np.full_like(log_spread, 100.0)
    a_close = np.exp(log_spread) * h_cny

    frame = pd.DataFrame({"date": dates, "a_close": a_close, "h_cny": h_cny})
    result = run_pairs_strategy(
        frame,
        entry=1.0,
        exit=0.5,
        z_window=3,
        cost_bps=0.0,
        executability_config=ExecutabilityConfig(
            enforce_a_share_t1=True,
            allow_short_a=False,
            allow_short_h=True,
        ),
    )

    daily = result.daily
    assert float(daily.loc[dates[3], "spread_position"]) == -1.0
    assert float(daily.loc[dates[3], "target_a_weight"]) == -1.0
    assert float(daily.loc[dates[3], "a_weight"]) == 0.0
    assert float(daily.loc[dates[3], "h_weight"]) > 0.0

    assert {"missed_trade_flag", "constraint_violations", "effective_turnover"}.issubset(
        set(daily.columns)
    )
    assert result.executability.missed_trades >= 1
    assert result.executability.constraint_violation_count >= 1


def test_pairs_strategy_can_block_short_h_when_configured() -> None:
    """When H shorting is disabled, long-spread signals should drop H short leg."""

    dates = pd.bdate_range("2024-01-01", periods=8)
    log_spread = np.array([0.0, 0.0, 0.0, -2.5, -2.0, -2.0, -2.0, -2.0])
    h_cny = np.full_like(log_spread, 120.0)
    a_close = np.exp(log_spread) * h_cny

    frame = pd.DataFrame({"date": dates, "a_close": a_close, "h_cny": h_cny})
    result = run_pairs_strategy(
        frame,
        entry=1.0,
        exit=0.5,
        z_window=3,
        executability_config=ExecutabilityConfig(
            enforce_a_share_t1=True,
            allow_short_a=True,
            allow_short_h=False,
        ),
    )

    daily = result.daily
    assert float(daily.loc[dates[3], "spread_position"]) == 1.0
    assert float(daily.loc[dates[3], "target_h_weight"]) < 0.0
    assert float(daily.loc[dates[3], "h_weight"]) == 0.0
