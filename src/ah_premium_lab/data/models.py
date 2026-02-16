"""Data models for A/H market and FX time series."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PriceSeries:
    """A price series for one ticker.

    Attributes:
        ticker: Security ticker symbol.
        data: Date-indexed DataFrame with `close` and `adj_close`.
    """

    ticker: str
    data: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate and normalize the underlying DataFrame."""

        normalized = _normalize_time_index(self.data)
        required_cols = {"close", "adj_close"}
        if not required_cols.issubset(set(normalized.columns)):
            raise ValueError("PriceSeries requires columns: close, adj_close")

        frame = normalized.loc[:, ["close", "adj_close"]].astype(float)
        object.__setattr__(self, "data", frame)


@dataclass(frozen=True)
class FxSeries:
    """An FX series for one currency pair.

    Attributes:
        pair: Currency pair name, e.g. `HKDCNY`.
        data: Date-indexed DataFrame with `close`.
    """

    pair: str
    data: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate and normalize the underlying DataFrame."""

        normalized = _normalize_time_index(self.data)
        if "close" not in normalized.columns:
            raise ValueError("FxSeries requires a `close` column")

        frame = normalized.loc[:, ["close"]].astype(float)
        object.__setattr__(self, "data", frame)


@dataclass(frozen=True)
class AhPair:
    """A/H pair metadata entry."""

    name: str
    a_ticker: str
    h_ticker: str
    share_ratio: float = 1.0
    notes: str = ""


def _normalize_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to sorted, de-duplicated `DatetimeIndex`."""

    if frame.empty:
        raise ValueError("Input frame cannot be empty")

    out = frame.copy()
    out.index = _to_datetime_index(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _to_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Convert an index-like object to a timezone-naive DatetimeIndex."""

    dt_index = pd.to_datetime(index)
    if not isinstance(dt_index, pd.DatetimeIndex):
        raise ValueError("Series index must be datetime-like")

    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert(None)

    return dt_index
