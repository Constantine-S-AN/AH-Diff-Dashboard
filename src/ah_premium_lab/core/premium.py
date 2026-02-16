"""A/H premium calculations and rolling indicators."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

FxQuote = Literal["HKDCNY", "CNYHKD"]


def compute_premium_metrics(
    a_price_cny: pd.Series,
    h_price_hkd: pd.Series,
    fx: pd.Series,
    *,
    fx_quote: FxQuote = "HKDCNY",
    share_ratio: float = 1.0,
    window: int = 252,
) -> pd.DataFrame:
    """Compute A/H premium and rolling indicators on aligned dates.

    Output fields:
    - premium_pct = A / (H * fx_hkd_to_cny * share_ratio) - 1
    - log_spread = log(A) - log(H * fx_hkd_to_cny * share_ratio)
    - rolling_zscore(log_spread, window)
    - rolling_percentile(premium_pct, window)
    - aligned_sample_size

    Args:
        a_price_cny: A-share close series in CNY.
        h_price_hkd: H-share close series in HKD.
        fx: FX close series. Quote direction controlled by `fx_quote`.
        fx_quote: FX quote convention for `fx`.
        share_ratio: Share conversion ratio (default 1.0).
        window: Rolling lookback window.

    Returns:
        Date-indexed DataFrame with aligned and derived columns.
    """

    if share_ratio <= 0.0:
        raise ValueError("share_ratio must be positive")

    aligned = pd.concat(
        [
            _normalize_series(a_price_cny).rename("a_price_cny"),
            _normalize_series(h_price_hkd).rename("h_price_hkd"),
            _normalize_series(fx).rename("fx_input"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("No overlapping observations after inner-join alignment")

    aligned["fx_hkd_to_cny"] = _to_hkd_to_cny(aligned["fx_input"], fx_quote=fx_quote)
    if (aligned["fx_hkd_to_cny"] <= 0.0).any():
        raise ValueError("fx_hkd_to_cny must be strictly positive")
    if (aligned["a_price_cny"] <= 0.0).any() or (aligned["h_price_hkd"] <= 0.0).any():
        raise ValueError("A/H prices must be strictly positive for log spread")

    aligned["h_price_cny_equiv"] = aligned["h_price_hkd"] * aligned["fx_hkd_to_cny"] * share_ratio
    aligned["premium_pct"] = aligned["a_price_cny"] / aligned["h_price_cny_equiv"] - 1.0
    aligned["log_spread"] = np.log(aligned["a_price_cny"]) - np.log(aligned["h_price_cny_equiv"])
    aligned["rolling_zscore"] = rolling_zscore(aligned["log_spread"], window=window)
    aligned["rolling_percentile"] = rolling_percentile(aligned["premium_pct"], window=window)
    aligned["aligned_sample_size"] = int(aligned.shape[0])

    return aligned


def rolling_zscore(
    series: pd.Series,
    *,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling z-score for one series."""

    resolved_min_periods = window if min_periods is None else min_periods
    roll_mean = series.rolling(window=window, min_periods=resolved_min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=resolved_min_periods).std(ddof=0)
    return (series - roll_mean) / roll_std.replace(0.0, np.nan)


def rolling_percentile(
    series: pd.Series,
    *,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute rolling percentile rank for the latest point in each window."""

    resolved_min_periods = window if min_periods is None else min_periods
    return series.rolling(window=window, min_periods=resolved_min_periods).apply(
        _percentile_of_last,
        raw=True,
    )


def compute_ah_premium(
    frame: pd.DataFrame,
    method: Literal["log_ratio", "pct"] = "log_ratio",
    *,
    fx_quote: FxQuote = "CNYHKD",
    share_ratio: float = 1.0,
) -> pd.DataFrame:
    """Compute one premium column from a standardized market frame.

    This legacy helper is preserved for current dashboard/backtest wiring.
    """

    _require_columns(frame, ["a_close", "h_close", "fx_rate"])

    fx_hkd_to_cny = _to_hkd_to_cny(frame["fx_rate"], fx_quote=fx_quote)
    h_in_cny = frame["h_close"] * fx_hkd_to_cny * share_ratio
    ratio = frame["a_close"] / h_in_cny

    output = frame.copy()
    if method == "log_ratio":
        output["premium"] = np.log(ratio)
    elif method == "pct":
        output["premium"] = ratio - 1.0
    else:
        raise ValueError(f"Unsupported premium method: {method}")

    return output


def add_rolling_zscore(
    frame: pd.DataFrame,
    column: str = "premium",
    window: int = 30,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Add rolling z-score column for spread normalization."""

    _require_columns(frame, [column])

    resolved_min_periods = min_periods if min_periods is not None else max(5, window // 2)
    output = frame.copy()
    output["zscore"] = rolling_zscore(
        frame[column],
        window=window,
        min_periods=resolved_min_periods,
    )
    return output


def prepare_spread_frame(
    frame: pd.DataFrame,
    method: Literal["log_ratio", "pct"] = "log_ratio",
    zscore_window: int = 30,
    *,
    fx_quote: FxQuote = "CNYHKD",
    share_ratio: float = 1.0,
) -> pd.DataFrame:
    """Compute premium and rolling z-score per pair."""

    grouped: list[pd.DataFrame] = []
    for _, pair_frame in frame.groupby("pair_id", sort=False):
        tmp = compute_ah_premium(
            pair_frame,
            method=method,
            fx_quote=fx_quote,
            share_ratio=share_ratio,
        )
        tmp = add_rolling_zscore(tmp, column="premium", window=zscore_window)
        grouped.append(tmp)
    return pd.concat(grouped, ignore_index=True)


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalize date index and numeric dtype for one input series."""

    clean = series.copy()
    clean.index = pd.to_datetime(clean.index)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError("Input series index must be datetime-like")
    if clean.index.tz is not None:
        clean.index = clean.index.tz_convert(None)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    return pd.to_numeric(clean, errors="coerce")


def _to_hkd_to_cny(fx: pd.Series, *, fx_quote: FxQuote) -> pd.Series:
    """Convert FX series into HKD->CNY quote."""

    normalized_quote = fx_quote.upper().replace("/", "")
    if normalized_quote == "HKDCNY":
        return fx.astype(float)
    if normalized_quote == "CNYHKD":
        if (fx == 0.0).any():
            raise ValueError("Cannot invert FX series containing zeros")
        return 1.0 / fx.astype(float)
    raise ValueError(f"Unsupported fx_quote: {fx_quote}")


def _percentile_of_last(values: np.ndarray) -> float:
    """Return percentile rank of the latest value within the input window."""

    sorted_values = np.sort(values)
    last = values[-1]
    rank = np.searchsorted(sorted_values, last, side="right")
    return float(rank / values.shape[0])


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Validate DataFrame required columns."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"DataFrame missing required columns: {missing_text}")
