"""Executability constraints for research backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutabilityConfig:
    """Configurable execution constraints for A/H research simulation.

    Attributes:
        enforce_a_share_t1: Enforce A-share T+1 style sell constraint.
        allow_short_a: Whether short A exposure is allowed.
        allow_short_h: Whether short H exposure is allowed.
    """

    enforce_a_share_t1: bool = True
    allow_short_a: bool = False
    allow_short_h: bool = True


@dataclass(frozen=True)
class ExecutabilityMetrics:
    """Summary metrics of how much the constraints changed intended trades."""

    missed_trades: int
    constraint_violation_count: int
    effective_turnover: float
    executability_score: float


def apply_executability_constraints(
    target_a_weight: pd.Series,
    target_h_weight: pd.Series,
    *,
    config: ExecutabilityConfig,
) -> tuple[pd.DataFrame, ExecutabilityMetrics]:
    """Apply A/H execution constraints to desired daily target weights.

    Rules:
    - If `allow_short_a` is false, negative A targets are clipped to zero.
    - If `allow_short_h` is false, negative H targets are clipped to zero.
    - If `enforce_a_share_t1` is true, reducing existing long-A exposure is blocked
      until at least one full bar after the most recent A buy event.

    Args:
        target_a_weight: Intended A-leg target weights indexed by date.
        target_h_weight: Intended H-leg target weights indexed by date.
        config: Executability settings.

    Returns:
        Tuple of:
        - DataFrame with columns:
          `target_a_weight`, `target_h_weight`, `a_weight`, `h_weight`,
          `missed_trade_flag`, `constraint_violations`
        - ExecutabilityMetrics with aggregate summary.
    """

    aligned = pd.concat(
        [
            target_a_weight.rename("target_a_weight"),
            target_h_weight.rename("target_h_weight"),
        ],
        axis=1,
    ).fillna(0.0)
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()

    realized_a: list[float] = []
    realized_h: list[float] = []
    missed_flags: list[bool] = []
    violation_counts: list[int] = []

    prev_a = 0.0
    days_since_last_a_buy = 10**6

    for idx, row in enumerate(aligned.itertuples(index=False)):
        if idx > 0:
            days_since_last_a_buy += 1

        desired_a = float(row.target_a_weight)
        desired_h = float(row.target_h_weight)

        adjusted_a = desired_a
        adjusted_h = desired_h
        violations = 0

        if not config.allow_short_a and adjusted_a < 0.0:
            adjusted_a = 0.0
            violations += 1

        if not config.allow_short_h and adjusted_h < 0.0:
            adjusted_h = 0.0
            violations += 1

        reduce_long_a = prev_a > 0.0 and adjusted_a < prev_a
        if config.enforce_a_share_t1 and reduce_long_a and days_since_last_a_buy < 1:
            adjusted_a = prev_a
            violations += 1

        if adjusted_a > prev_a and adjusted_a > 0.0:
            days_since_last_a_buy = 0

        missed = not (np.isclose(adjusted_a, desired_a) and np.isclose(adjusted_h, desired_h))

        realized_a.append(float(adjusted_a))
        realized_h.append(float(adjusted_h))
        missed_flags.append(bool(missed))
        violation_counts.append(int(violations))

        prev_a = float(adjusted_a)

    out = aligned.copy()
    out["a_weight"] = pd.Series(realized_a, index=out.index, dtype=float)
    out["h_weight"] = pd.Series(realized_h, index=out.index, dtype=float)
    out["missed_trade_flag"] = pd.Series(missed_flags, index=out.index, dtype=bool)
    out["constraint_violations"] = pd.Series(violation_counts, index=out.index, dtype=int)

    raw_turnover = _turnover(out["target_a_weight"], out["target_h_weight"])
    realized_turnover = _turnover(out["a_weight"], out["h_weight"])

    signal_days = int((raw_turnover > 1e-12).sum())
    missed_trades = int(out["missed_trade_flag"].sum())
    constraint_violation_count = int(out["constraint_violations"].sum())
    effective_turnover = float(realized_turnover.sum())

    score = compute_executability_score(
        missed_trades=missed_trades,
        constraint_violation_count=constraint_violation_count,
        effective_turnover=effective_turnover,
        raw_turnover=float(raw_turnover.sum()),
        signal_days=signal_days,
    )
    metrics = ExecutabilityMetrics(
        missed_trades=missed_trades,
        constraint_violation_count=constraint_violation_count,
        effective_turnover=effective_turnover,
        executability_score=score,
    )
    return out, metrics


def compute_executability_score(
    *,
    missed_trades: int,
    constraint_violation_count: int,
    effective_turnover: float,
    raw_turnover: float,
    signal_days: int,
) -> float:
    """Compute a 0-100 executability score.

    Formula:
    - `miss_rate = missed_trades / max(signal_days, 1)`
    - `violation_rate = constraint_violation_count / max(signal_days, 1)`
    - `turnover_preservation = effective_turnover / raw_turnover` (or `1` when `raw_turnover=0`)
    - `score = 100 * (0.50*(1-miss_rate) + 0.35*(1-violation_rate) + 0.15*turnover_preservation)`
    """

    base = max(signal_days, 1)
    miss_rate = float(np.clip(missed_trades / base, 0.0, 1.0))
    violation_rate = float(np.clip(constraint_violation_count / base, 0.0, 1.0))

    if raw_turnover <= 1e-12:
        turnover_preservation = 1.0
    else:
        turnover_preservation = float(np.clip(effective_turnover / raw_turnover, 0.0, 1.0))

    score = 100.0 * (
        0.50 * (1.0 - miss_rate) + 0.35 * (1.0 - violation_rate) + 0.15 * turnover_preservation
    )
    return float(np.clip(score, 0.0, 100.0))


def _turnover(a_weight: pd.Series, h_weight: pd.Series) -> pd.Series:
    """Compute daily two-leg turnover from weight changes."""

    prev_a = a_weight.shift(1).fillna(0.0)
    prev_h = h_weight.shift(1).fillna(0.0)
    return (a_weight - prev_a).abs() + (h_weight - prev_h).abs()
