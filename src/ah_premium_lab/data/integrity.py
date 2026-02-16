"""Data integrity checks and warning emitters."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ah_premium_lab.data.models import FxSeries, PriceSeries


@dataclass(frozen=True)
class IntegrityCheckResult:
    """Summary of integrity diagnostics for one series."""

    series_id: str
    missing_days: int
    jump_count: int
    jump_dates: tuple[pd.Timestamp, ...]
    warnings: tuple[str, ...]


def check_price_integrity(
    series: PriceSeries,
    zscore_threshold: float = 8.0,
) -> IntegrityCheckResult:
    """Run integrity checks for a `PriceSeries`."""

    return _run_checks(
        series_id=series.ticker,
        close=series.data["close"],
        zscore_threshold=zscore_threshold,
    )


def check_fx_integrity(
    series: FxSeries,
    zscore_threshold: float = 8.0,
) -> IntegrityCheckResult:
    """Run integrity checks for a `FxSeries`."""

    return _run_checks(
        series_id=series.pair,
        close=series.data["close"],
        zscore_threshold=zscore_threshold,
    )


def _run_checks(
    series_id: str,
    close: pd.Series,
    zscore_threshold: float,
) -> IntegrityCheckResult:
    """Check missing business days and abnormal jump points."""

    if close.empty:
        raise ValueError("Close series cannot be empty")

    index = pd.DatetimeIndex(close.index).sort_values()
    expected_days = pd.bdate_range(index.min(), index.max())
    observed_days = pd.DatetimeIndex(index.normalize().unique())
    missing = expected_days.difference(observed_days)

    returns = close.pct_change().dropna()
    jump_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    std = float(returns.std(ddof=0)) if not returns.empty else 0.0
    if returns.shape[0] >= 3 and std > 0.0:
        zscore = (returns - returns.mean()) / std
        jump_dates = pd.DatetimeIndex(zscore[np.abs(zscore) > zscore_threshold].index)

    warning_messages: list[str] = []
    if len(missing) > 0:
        msg = f"[{series_id}] missing business days: {len(missing)}"
        warnings.warn(msg, UserWarning, stacklevel=2)
        warning_messages.append(msg)

    if len(jump_dates) > 0:
        sample_dates = ", ".join(ts.strftime("%Y-%m-%d") for ts in jump_dates[:3])
        msg = (
            f"[{series_id}] detected abnormal jumps (|z| > {zscore_threshold:.1f}): "
            f"{len(jump_dates)} points; sample={sample_dates}"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        warning_messages.append(msg)

    return IntegrityCheckResult(
        series_id=series_id,
        missing_days=int(len(missing)),
        jump_count=int(len(jump_dates)),
        jump_dates=tuple(jump_dates.to_list()),
        warnings=tuple(warning_messages),
    )
