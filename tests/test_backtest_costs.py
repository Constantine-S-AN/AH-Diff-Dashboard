"""Tests for cost model utilities and sensitivity helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ah_premium_lab.backtest import (
    CostParams,
    apply_costs_to_daily,
    estimate_breakeven_cost_level,
    generate_cost_grid,
    run_pair_cost_sensitivity,
)


def test_generate_cost_grid_size() -> None:
    """Grid size should equal Cartesian product of each level set."""

    grid = generate_cost_grid(
        commission_bps_levels=(0, 10),
        slippage_bps_levels=(0, 20),
        stamp_duty_bps_levels=(0, 15),
        commission_h_bps_levels=(0,),
        stamp_duty_h_bps_levels=(0,),
    )

    assert len(grid) == 8
    assert isinstance(grid[0], CostParams)


def test_apply_costs_to_daily_reduces_net_curve() -> None:
    """Positive costs should reduce net curve versus gross curve when turnover exists."""

    dates = pd.bdate_range("2024-01-01", periods=5)
    daily = pd.DataFrame(
        {
            "pnl_gross": [0.0, 0.01, -0.004, 0.006, 0.0],
            "a_weight": [0.0, 1.0, 1.0, 0.0, 0.0],
            "h_weight": [0.0, -1.0, -1.0, 0.0, 0.0],
        },
        index=dates,
    )

    out = apply_costs_to_daily(
        daily,
        CostParams(
            commission_a_bps=5.0,
            commission_h_bps=5.0,
            stamp_duty_a_bps=10.0,
            stamp_duty_h_bps=10.0,
            slippage_bps=10.0,
            # Split A/H buy/sell sides to validate two-sided cost model.
            commission_a_buy_bps=2.0,
            commission_a_sell_bps=8.0,
            commission_h_buy_bps=3.0,
            commission_h_sell_bps=7.0,
            stamp_duty_a_buy_bps=0.0,
            stamp_duty_a_sell_bps=10.0,
            stamp_duty_h_buy_bps=1.0,
            stamp_duty_h_sell_bps=4.0,
            slippage_a_buy_bps=5.0,
            slippage_a_sell_bps=6.0,
            slippage_h_buy_bps=4.0,
            slippage_h_sell_bps=5.0,
            borrow_bps=120.0,
        ),
    )

    assert (out["cost_ret"] >= 0.0).all()
    assert {"trade_cost_ret", "borrow_cost_ret", "turnover_a_buy", "turnover_h_sell"}.issubset(
        out.columns
    )
    assert float(out["borrow_cost_ret"].sum()) > 0.0
    assert float(out["curve_net"].iloc[-1]) <= float(out["curve_gross"].iloc[-1])


def test_pair_sensitivity_and_breakeven_estimation(tmp_path: Path) -> None:
    """Pair sensitivity should return expected metrics and a valid report-ready table."""

    dates = pd.bdate_range("2024-01-01", periods=120)
    signal = np.sin(np.linspace(0, 12.0, len(dates))) * 0.5
    h_cny = np.full(len(dates), 100.0)
    a_close = np.exp(signal) * h_cny

    frame = pd.DataFrame(
        {
            "date": dates,
            "pair_id": "P1",
            "a_close": a_close,
            "h_cny": h_cny,
        }
    )

    grid = [
        CostParams(0.0, 0.0, 0.0, 0.0, 0.0),
        CostParams(5.0, 5.0, 5.0, 5.0, 5.0),
        CostParams(10.0, 10.0, 10.0, 10.0, 10.0),
    ]

    output = run_pair_cost_sensitivity(
        pair_frame=frame,
        cost_grid=grid,
        pair_id="P1",
        entry=0.8,
        exit=0.3,
        z_window=20,
    )

    df = output.grid_results
    assert not df.empty
    assert {
        "net_cagr",
        "net_sharpe",
        "max_dd",
        "breakeven_cost_level",
        "breakeven_total_cost",
        "breakeven_slippage",
        "worst_case_net_dd",
    }.issubset(df.columns)

    direct_breakeven = estimate_breakeven_cost_level(df)
    assert np.isfinite(direct_breakeven) or np.isnan(direct_breakeven)
