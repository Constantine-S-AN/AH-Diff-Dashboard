"""Structural-break diagnostics for log-spread series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.diagnostic import breaks_cusumolsresid


@dataclass(frozen=True)
class BreakDetectionResult:
    """Container for structural-break diagnostics."""

    breakpoints: pd.DataFrame
    cusum_stat: float
    cusum_p_value: float


def detect_structural_breaks(
    log_spread: pd.Series,
    *,
    window: int = 63,
    min_distance: int = 21,
    alpha: float = 0.01,
    zscore_threshold: float = 2.5,
) -> BreakDetectionResult:
    """Detect structural breaks using rolling mean-shift significance checks.

    Method:
    1. For each candidate date, split left/right windows of equal length.
    2. Compute Welch t-test p-value on means.
    3. Compute mean-shift z-score:
       `(mean_right - mean_left) / sqrt(var_left/n + var_right/n)`.
    4. Keep candidates with `p_value <= alpha` and `|z| >= zscore_threshold`.
    5. Apply non-maximum suppression with `min_distance` to avoid clustered dates.

    Confidence is mapped into `[0, 1]`:
    `0.7 * (1 - p_value) + 0.3 * clip(|z| / 8, 0, 1)`.

    Args:
        log_spread: Log-spread time series indexed by date.
        window: Lookback/lookforward window length for each side.
        min_distance: Minimal spacing (in observations) between reported breaks.
        alpha: Significance cutoff for candidate detection.
        zscore_threshold: Minimal absolute mean-shift z-score.

    Returns:
        BreakDetectionResult with:
        - `breakpoints`: DataFrame with columns
          `[break_date, confidence, p_value, shift_zscore, mean_shift]`
        - `cusum_stat`, `cusum_p_value`: CUSUM stability test outputs.
    """

    if window < 10:
        raise ValueError("window must be >= 10")
    if min_distance < 1:
        raise ValueError("min_distance must be >= 1")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if zscore_threshold <= 0.0:
        raise ValueError("zscore_threshold must be > 0")

    clean = log_spread.dropna().astype(float)
    cusum_stat, cusum_p_value = cusum_stability_test(clean)

    if clean.shape[0] < window * 2 + 1:
        empty = pd.DataFrame(
            columns=["break_date", "confidence", "p_value", "shift_zscore", "mean_shift"],
        )
        return BreakDetectionResult(
            breakpoints=empty,
            cusum_stat=cusum_stat,
            cusum_p_value=cusum_p_value,
        )

    values = clean.to_numpy(dtype=float)
    index = clean.index
    candidates: list[dict[str, float | int | pd.Timestamp]] = []

    for pos in range(window, values.shape[0] - window):
        left = values[pos - window : pos]
        right = values[pos : pos + window]

        mean_shift, shift_zscore = _mean_shift_zscore(left, right)
        if not np.isfinite(shift_zscore):
            continue

        t_stat = ttest_ind(left, right, equal_var=False, nan_policy="omit")
        p_value = float(t_stat.pvalue) if np.isfinite(t_stat.pvalue) else 1.0
        confidence = _break_confidence(p_value, shift_zscore)

        if p_value <= alpha and abs(shift_zscore) >= zscore_threshold:
            candidates.append(
                {
                    "position": int(pos),
                    "break_date": pd.Timestamp(index[pos]),
                    "confidence": float(confidence),
                    "p_value": float(p_value),
                    "shift_zscore": float(shift_zscore),
                    "abs_shift_zscore": float(abs(shift_zscore)),
                    "mean_shift": float(mean_shift),
                }
            )

    if not candidates:
        empty = pd.DataFrame(
            columns=["break_date", "confidence", "p_value", "shift_zscore", "mean_shift"],
        )
        return BreakDetectionResult(
            breakpoints=empty,
            cusum_stat=cusum_stat,
            cusum_p_value=cusum_p_value,
        )

    candidate_frame = pd.DataFrame(candidates)
    selected = _suppress_neighboring_breaks(candidate_frame, min_distance=min_distance)
    output = selected[
        ["break_date", "confidence", "p_value", "shift_zscore", "mean_shift"]
    ].reset_index(drop=True)
    return BreakDetectionResult(
        breakpoints=output,
        cusum_stat=cusum_stat,
        cusum_p_value=cusum_p_value,
    )


def cusum_stability_test(series: pd.Series) -> tuple[float, float]:
    """Run CUSUM residual-stability test on a demeaned series.

    Args:
        series: Input series.

    Returns:
        Tuple `(cusum_stat, cusum_p_value)`. Returns `(nan, nan)` for short/invalid samples.
    """

    clean = series.dropna().astype(float)
    if clean.shape[0] < 20:
        return float("nan"), float("nan")

    centered = clean - float(clean.mean())
    try:
        stat, p_value, _ = breaks_cusumolsresid(centered.to_numpy(), ddof=1)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")

    return float(stat), float(p_value)


def _mean_shift_zscore(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Compute mean shift and associated Welch-style z-score."""

    if left.size < 2 or right.size < 2:
        return float("nan"), float("nan")

    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    shift = right_mean - left_mean

    left_var = float(np.var(left, ddof=1))
    right_var = float(np.var(right, ddof=1))
    denominator = float(np.sqrt(left_var / left.size + right_var / right.size))
    if denominator <= 1e-12 or not np.isfinite(denominator):
        return shift, float("nan")
    return shift, float(shift / denominator)


def _break_confidence(p_value: float, shift_zscore: float) -> float:
    """Map p-value and shift magnitude into [0, 1] confidence."""

    p_component = 1.0 - float(np.clip(p_value, 0.0, 1.0)) if np.isfinite(p_value) else 0.0
    z_component = (
        float(np.clip(abs(shift_zscore) / 8.0, 0.0, 1.0))
        if np.isfinite(shift_zscore)
        else 0.0
    )
    confidence = 0.7 * p_component + 0.3 * z_component
    return float(np.clip(confidence, 0.0, 1.0))


def _suppress_neighboring_breaks(
    candidates: pd.DataFrame,
    *,
    min_distance: int,
) -> pd.DataFrame:
    """Keep high-confidence breakpoints while enforcing minimum spacing."""

    ranked = candidates.sort_values(
        ["confidence", "abs_shift_zscore"],
        ascending=[False, False],
    ).reset_index(drop=True)

    kept_positions: list[int] = []
    kept_rows: list[pd.Series] = []
    for _, row in ranked.iterrows():
        pos = int(row["position"])
        if all(abs(pos - prev) >= min_distance for prev in kept_positions):
            kept_positions.append(pos)
            kept_rows.append(row)

    if not kept_rows:
        return pd.DataFrame(columns=candidates.columns)

    selected = pd.DataFrame(kept_rows)
    selected = selected.sort_values("break_date").reset_index(drop=True)
    return selected.drop(columns=["position", "abs_shift_zscore"])
