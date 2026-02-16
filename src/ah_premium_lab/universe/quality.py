"""Data quality diagnostics for A/H/FX series coverage and gaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesQuality:
    """Quality summary for one input series."""

    coverage_pct: float
    max_gap_days: int
    outlier_count: int
    missing_rate: float
    observed_days: int
    expected_days: int


@dataclass(frozen=True)
class PairQuality:
    """Aggregated quality summary for A/H/FX tuple."""

    coverage_pct: float
    max_gap_days: int
    outlier_count: int
    quality_score: float
    quality_flag: str
    missing_threshold_breached: bool
    max_missing_rate: float


def compute_series_quality(
    series: pd.Series,
    *,
    start_date: str,
    end_date: str,
    outlier_z_threshold: float = 8.0,
) -> SeriesQuality:
    """Compute coverage/gap/outlier diagnostics for one price series."""

    expected_idx = pd.bdate_range(start=start_date, end=end_date)
    expected_days = int(len(expected_idx))

    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    observed_idx = pd.DatetimeIndex(clean.index)
    if observed_idx.tz is not None:
        observed_idx = observed_idx.tz_convert(None)
    observed_idx = observed_idx.normalize()
    observed_idx = observed_idx.intersection(expected_idx).unique().sort_values()

    observed_days = int(len(observed_idx))

    if expected_days > 0:
        coverage_pct = float(100.0 * observed_days / expected_days)
        missing_rate = float(1.0 - observed_days / expected_days)
    else:
        coverage_pct = float("nan")
        missing_rate = float("nan")

    max_gap_days = _max_gap_days(expected_idx, observed_idx)
    outlier_count = _outlier_count(clean, threshold=outlier_z_threshold)

    return SeriesQuality(
        coverage_pct=coverage_pct,
        max_gap_days=max_gap_days,
        outlier_count=outlier_count,
        missing_rate=missing_rate,
        observed_days=observed_days,
        expected_days=expected_days,
    )


def combine_pair_quality(
    a_quality: SeriesQuality,
    h_quality: SeriesQuality,
    fx_quality: SeriesQuality,
    *,
    missing_threshold: float = 0.2,
) -> PairQuality:
    """Aggregate three legs into pair-level quality score and flag."""

    coverage_pct = float(
        np.nanmin(
            [a_quality.coverage_pct, h_quality.coverage_pct, fx_quality.coverage_pct],
        )
    )
    max_gap_days = int(max(a_quality.max_gap_days, h_quality.max_gap_days, fx_quality.max_gap_days))
    outlier_count = int(
        a_quality.outlier_count + h_quality.outlier_count + fx_quality.outlier_count
    )
    max_missing_rate = float(
        np.nanmax(
            [a_quality.missing_rate, h_quality.missing_rate, fx_quality.missing_rate],
        )
    )

    missing_penalty = max_missing_rate * 70.0 if np.isfinite(max_missing_rate) else 70.0
    gap_penalty = min(20.0, max_gap_days * 1.0)
    outlier_penalty = min(10.0, outlier_count * 0.5)

    base_score = max(0.0, 100.0 - missing_penalty - gap_penalty - outlier_penalty)
    breached = bool(np.isfinite(max_missing_rate) and max_missing_rate > missing_threshold)

    if breached:
        quality_score = float(base_score * 0.45)
        quality_flag = "poor"
    elif base_score < 70.0:
        quality_score = float(base_score)
        quality_flag = "warning"
    else:
        quality_score = float(base_score)
        quality_flag = "good"

    return PairQuality(
        coverage_pct=coverage_pct,
        max_gap_days=max_gap_days,
        outlier_count=outlier_count,
        quality_score=quality_score,
        quality_flag=quality_flag,
        missing_threshold_breached=breached,
        max_missing_rate=max_missing_rate,
    )


def _max_gap_days(
    expected_idx: pd.DatetimeIndex,
    observed_idx: pd.DatetimeIndex,
) -> int:
    """Compute maximum contiguous missing business-day gap."""

    expected_count = int(len(expected_idx))
    if expected_count == 0:
        return 0
    if len(observed_idx) == 0:
        return expected_count

    positions = expected_idx.get_indexer(observed_idx)
    positions = positions[positions >= 0]
    if positions.size == 0:
        return expected_count

    positions = np.sort(positions)
    gaps: list[int] = []

    leading_gap = int(positions[0])
    trailing_gap = int(expected_count - 1 - positions[-1])
    gaps.append(max(0, leading_gap))
    gaps.append(max(0, trailing_gap))

    for idx in range(1, len(positions)):
        internal_gap = int(positions[idx] - positions[idx - 1] - 1)
        gaps.append(max(0, internal_gap))

    return int(max(gaps)) if gaps else 0


def _outlier_count(series: pd.Series, *, threshold: float) -> int:
    """Count return outliers by z-score threshold."""

    returns = series.pct_change().dropna()
    if returns.shape[0] < 3:
        return 0

    std = float(returns.std(ddof=0))
    if std <= 0.0:
        return 0

    zscore = (returns - returns.mean()) / std
    return int((np.abs(zscore) > threshold).sum())
