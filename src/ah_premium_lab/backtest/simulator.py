"""Simple cost-aware spread mean-reversion simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ah_premium_lab.config import BacktestConfig


@dataclass(frozen=True)
class BacktestSummary:
    """Aggregate backtest performance metrics."""

    cost_bps: float
    cumulative_return: float
    annualized_sharpe: float
    max_drawdown: float
    trade_count: int


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output including equity curve and summary metrics."""

    equity_curve: pd.DataFrame
    summary: BacktestSummary


def simulate_mean_reversion(
    frame: pd.DataFrame,
    config: BacktestConfig,
    cost_bps: float,
) -> BacktestResult:
    """Simulate a z-score mean-reversion strategy with linear transaction costs.

    Args:
        frame: Prepared pair data with `premium` and `zscore` columns.
        config: Backtest parameters.
        cost_bps: One-way cost in basis points per turnover unit.

    Returns:
        BacktestResult containing equity curve and summary metrics.
    """

    _require_columns(frame, ["premium", "zscore"])

    work = frame.sort_values("date").reset_index(drop=True).copy()
    work["position"] = _build_position_path(
        zscores=work["zscore"],
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        max_holding_days=config.max_holding_days,
    )

    spread_change = work["premium"].diff().fillna(0.0)
    prev_position = work["position"].shift(1).fillna(0.0)

    gross_ret = prev_position * spread_change
    turnover = work["position"].diff().abs().fillna(work["position"].abs())
    cost_ret = turnover * (cost_bps / 10000.0)
    net_ret = gross_ret - cost_ret

    equity = config.initial_capital * (1.0 + net_ret).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    work["gross_ret"] = gross_ret
    work["cost_ret"] = cost_ret
    work["net_ret"] = net_ret
    work["equity"] = equity
    work["drawdown"] = drawdown

    std = float(net_ret.std(ddof=0))
    sharpe = 0.0
    if std > 0.0:
        sharpe = float(np.sqrt(config.annualization) * net_ret.mean() / std)

    trade_count = int(((work["position"] != 0.0) & (prev_position == 0.0)).sum())
    cumulative_return = float(equity.iloc[-1] / config.initial_capital - 1.0)

    summary = BacktestSummary(
        cost_bps=float(cost_bps),
        cumulative_return=cumulative_return,
        annualized_sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        trade_count=trade_count,
    )
    return BacktestResult(equity_curve=work, summary=summary)


def _build_position_path(
    zscores: pd.Series,
    entry_z: float,
    exit_z: float,
    max_holding_days: int,
) -> pd.Series:
    """Construct discrete spread positions from z-score thresholds."""

    positions: list[float] = []
    current_pos = 0.0
    holding_days = 0

    for raw_z in zscores.to_numpy():
        z = float(raw_z) if not np.isnan(raw_z) else 0.0

        if current_pos == 0.0:
            if z >= entry_z:
                current_pos = -1.0
                holding_days = 0
            elif z <= -entry_z:
                current_pos = 1.0
                holding_days = 0
        else:
            holding_days += 1
            if abs(z) <= exit_z or holding_days >= max_holding_days:
                current_pos = 0.0
                holding_days = 0

        positions.append(current_pos)

    return pd.Series(positions, index=zscores.index, dtype=float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Ensure required columns exist."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {', '.join(missing)}")
