"""Daily close-based research strategy for A/H spread mean reversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ah_premium_lab.backtest.executability import (
    ExecutabilityConfig,
    ExecutabilityMetrics,
    apply_executability_constraints,
)
from ah_premium_lab.core import rolling_zscore

FxQuote = Literal["HKDCNY", "CNYHKD"]


@dataclass(frozen=True)
class PairsStrategyResult:
    """Backtest result for the A/H pairs spread strategy.

    Attributes:
        daily: Daily simulation table with positions, returns, curves, and drawdowns.
        max_drawdown_gross: Max drawdown from gross curve.
        max_drawdown_net: Max drawdown from net curve.
        executability: Executability summary metrics.
    """

    daily: pd.DataFrame
    max_drawdown_gross: float
    max_drawdown_net: float
    executability: ExecutabilityMetrics


def run_pairs_strategy(
    frame: pd.DataFrame,
    *,
    entry: float = 2.0,
    exit: float = 0.5,
    z_window: int = 252,
    cost_bps: float = 0.0,
    share_ratio: float = 1.0,
    fx_quote: FxQuote = "HKDCNY",
    executability_config: ExecutabilityConfig | None = None,
) -> PairsStrategyResult:
    """Run a daily close spread strategy on one A/H pair.

    Strategy rules on `rolling zscore(log_spread)`:
    - `z > entry`: short spread (sell A, buy H)
    - `z < -entry`: long spread (buy A, sell H)
    - `|z| < exit`: flat

    Daily-frequency execution assumptions:
    - Signals are computed from day-`t` close data.
    - Target weights are applied at day `t` close.
    - Daily PnL on row `t` uses previous-close to current-close returns and previous day weights.
    - Transaction costs are linear in daily turnover and applied on rebalance days.
    - No intraday/high-frequency matching, slippage impact model, financing, or borrow constraints.
    - Optional executability constraints can be enabled via `executability_config`.

    Required columns:
    - always: `a_close`
    - for price conversion: either `h_cny` or (`h_close` and `fx_rate`)
    - optionally: `log_spread` (if absent, computed from prices)

    Returns:
        `PairsStrategyResult` containing full daily table and max drawdowns.

    Notes:
        When `executability_config` is provided, the daily table includes:
        `target_a_weight`, `target_h_weight`, `missed_trade_flag`, and
        `constraint_violations`, and realized weights may differ from signal targets.
    """

    if entry <= 0.0 or exit < 0.0:
        raise ValueError("entry must be > 0 and exit must be >= 0")
    if share_ratio <= 0.0:
        raise ValueError("share_ratio must be positive")

    work = _normalize_frame(frame)
    _require_columns(work, ["a_close"])

    work["h_cny"] = _resolve_h_cny(work, fx_quote=fx_quote)

    if "log_spread" not in work.columns:
        _require_columns(work, ["a_close", "h_cny"])
        if (work["a_close"] <= 0.0).any() or (work["h_cny"] <= 0.0).any():
            raise ValueError("a_close and h_cny must be strictly positive")
        work["log_spread"] = np.log(work["a_close"]) - np.log(work["h_cny"] * share_ratio)

    work["zscore"] = rolling_zscore(work["log_spread"], window=z_window)
    work["spread_position"] = _build_spread_position(work["zscore"], entry=entry, exit=exit)

    work["target_a_weight"] = work["spread_position"]
    work["target_h_weight"] = -work["spread_position"] * share_ratio

    if executability_config is None:
        work["a_weight"] = work["target_a_weight"]
        work["h_weight"] = work["target_h_weight"]
        work["missed_trade_flag"] = False
        work["constraint_violations"] = 0
        executability = ExecutabilityMetrics(
            missed_trades=0,
            constraint_violation_count=0,
            effective_turnover=float(
                (
                    (work["a_weight"] - work["a_weight"].shift(1).fillna(0.0)).abs()
                    + (work["h_weight"] - work["h_weight"].shift(1).fillna(0.0)).abs()
                ).sum()
            ),
            executability_score=100.0,
        )
    else:
        constrained, executability = apply_executability_constraints(
            work["target_a_weight"],
            work["target_h_weight"],
            config=executability_config,
        )
        work["a_weight"] = constrained["a_weight"]
        work["h_weight"] = constrained["h_weight"]
        work["missed_trade_flag"] = constrained["missed_trade_flag"]
        work["constraint_violations"] = constrained["constraint_violations"]

    a_ret = work["a_close"].pct_change().fillna(0.0)
    h_ret = work["h_cny"].pct_change().fillna(0.0)

    prev_a_weight = work["a_weight"].shift(1).fillna(0.0)
    prev_h_weight = work["h_weight"].shift(1).fillna(0.0)

    work["pnl_gross"] = prev_a_weight * a_ret + prev_h_weight * h_ret

    turnover_a = (work["a_weight"] - prev_a_weight).abs()
    turnover_h = (work["h_weight"] - prev_h_weight).abs()
    work["turnover_a"] = turnover_a
    work["turnover_h"] = turnover_h
    work["turnover"] = turnover_a + turnover_h
    work["effective_turnover"] = work["turnover"]

    work["cost_ret"] = work["turnover"] * (cost_bps / 10000.0)
    work["pnl_net"] = work["pnl_gross"] - work["cost_ret"]

    work["curve_gross"] = (1.0 + work["pnl_gross"]).cumprod()
    work["curve_net"] = (1.0 + work["pnl_net"]).cumprod()

    gross_peak = work["curve_gross"].cummax()
    net_peak = work["curve_net"].cummax()
    work["drawdown_gross"] = work["curve_gross"] / gross_peak - 1.0
    work["drawdown_net"] = work["curve_net"] / net_peak - 1.0

    return PairsStrategyResult(
        daily=work,
        max_drawdown_gross=float(work["drawdown_gross"].min()),
        max_drawdown_net=float(work["drawdown_net"].min()),
        executability=executability,
    )


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize date handling and sort order."""

    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _resolve_h_cny(work: pd.DataFrame, fx_quote: FxQuote) -> pd.Series:
    """Resolve H-share CNY series from input columns."""

    if "h_cny" in work.columns:
        return work["h_cny"].astype(float)

    _require_columns(work, ["h_close", "fx_rate"])
    fx_hkd_to_cny = _to_hkd_to_cny(work["fx_rate"], fx_quote=fx_quote)
    return work["h_close"].astype(float) * fx_hkd_to_cny


def _to_hkd_to_cny(fx: pd.Series, fx_quote: FxQuote) -> pd.Series:
    """Convert FX quote to HKD->CNY."""

    normalized_quote = fx_quote.upper().replace("/", "")
    fx_val = fx.astype(float)

    if normalized_quote == "HKDCNY":
        return fx_val
    if normalized_quote == "CNYHKD":
        if (fx_val == 0.0).any():
            raise ValueError("Cannot invert FX series containing zeros")
        return 1.0 / fx_val

    raise ValueError(f"Unsupported fx_quote: {fx_quote}")


def _build_spread_position(zscore: pd.Series, entry: float, exit: float) -> pd.Series:
    """Generate spread position path from z-score thresholds."""

    positions: list[float] = []
    current = 0.0

    for raw_z in zscore.to_numpy():
        z = float(raw_z)
        if np.isnan(z):
            positions.append(current)
            continue

        if z > entry:
            current = -1.0
        elif z < -entry:
            current = 1.0
        elif abs(z) < exit:
            current = 0.0

        positions.append(current)

    return pd.Series(positions, index=zscore.index, dtype=float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Validate required DataFrame columns."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {', '.join(missing)}")
