"""Provider interfaces and implementations for market data."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from ah_premium_lab.data.integrity import check_fx_integrity, check_price_integrity
from ah_premium_lab.data.models import FxSeries, PriceSeries

DateLike = str | date | datetime | pd.Timestamp


class PriceProvider(ABC):
    """Abstract data provider interface with parquet caching."""

    def __init__(self, cache_dir: str | Path = Path("data/cache")) -> None:
        """Initialize provider with cache directory."""

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_price(self, ticker: str, start: DateLike, end: DateLike) -> PriceSeries:
        """Fetch price series for one ticker."""

    @abstractmethod
    def get_fx(self, pair: str, start: DateLike, end: DateLike) -> FxSeries:
        """Fetch FX series for one currency pair."""

    def _load_or_fetch(
        self,
        namespace: str,
        symbol: str,
        start: DateLike,
        end: DateLike,
        fetcher: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Load from cache first; otherwise fetch and persist to parquet."""

        cache_path = self._cache_path(namespace, symbol, start, end)
        if cache_path.exists():
            return pd.read_parquet(cache_path).sort_index()

        frame = fetcher().sort_index()
        frame.to_parquet(cache_path)
        return frame

    def _cache_path(
        self,
        namespace: str,
        symbol: str,
        start: DateLike,
        end: DateLike,
    ) -> Path:
        """Build deterministic cache path for one request."""

        key = f"{namespace}|{symbol.upper()}|{_to_date_text(start)}|{_to_date_text(end)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        safe_symbol = _slug(symbol)
        return self.cache_dir / f"{namespace}_{safe_symbol}_{digest}.parquet"


class YahooFinanceProvider(PriceProvider):
    """Yahoo Finance provider backed by `yfinance`."""

    def get_price(self, ticker: str, start: DateLike, end: DateLike) -> PriceSeries:
        """Fetch close and adjusted close for one ticker."""

        frame = self._load_or_fetch(
            namespace="price",
            symbol=ticker,
            start=start,
            end=end,
            fetcher=lambda: self._download_price_frame(ticker=ticker, start=start, end=end),
        )
        series = PriceSeries(ticker=ticker, data=frame)
        check_price_integrity(series)
        return series

    def get_fx(
        self,
        pair: str = "HKDCNY",
        start: DateLike = "2000-01-01",
        end: DateLike = "2100-01-01",
    ) -> FxSeries:
        """Fetch FX close series for HKD/CNY or CNY/HKD."""

        normalized_pair = pair.upper().replace("/", "")
        if normalized_pair not in {"HKDCNY", "CNYHKD"}:
            raise ValueError("Only HKDCNY or CNYHKD is supported")

        yahoo_ticker = f"{normalized_pair}=X"
        frame = self._load_or_fetch(
            namespace="fx",
            symbol=normalized_pair,
            start=start,
            end=end,
            fetcher=lambda: self._download_fx_frame(ticker=yahoo_ticker, start=start, end=end),
        )
        series = FxSeries(pair=normalized_pair, data=frame)
        check_fx_integrity(series)
        return series

    def _download_price_frame(self, ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
        """Download and normalize equity price columns from Yahoo Finance."""

        raw = self._download_raw(ticker=ticker, start=start, end=end)
        close_col = "Close" if "Close" in raw.columns else "Adj Close"
        if close_col not in raw.columns:
            raise ValueError(f"Yahoo data for {ticker} does not contain Close/Adj Close")

        adj_col = "Adj Close" if "Adj Close" in raw.columns else close_col
        out = pd.DataFrame(index=raw.index)
        out["close"] = raw[close_col].astype(float)
        out["adj_close"] = raw[adj_col].astype(float)
        out = out.dropna(subset=["close", "adj_close"])
        return out

    def _download_fx_frame(self, ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
        """Download and normalize FX close column from Yahoo Finance."""

        raw = self._download_raw(ticker=ticker, start=start, end=end)
        close_col = "Close" if "Close" in raw.columns else "Adj Close"
        if close_col not in raw.columns:
            raise ValueError(f"Yahoo FX data for {ticker} does not contain Close/Adj Close")

        out = pd.DataFrame(index=raw.index)
        out["close"] = raw[close_col].astype(float)
        out = out.dropna(subset=["close"])
        return out

    def _download_raw(self, ticker: str, start: DateLike, end: DateLike) -> pd.DataFrame:
        """Download raw OHLCV DataFrame from Yahoo Finance."""

        raw = yf.download(
            tickers=ticker,
            start=_to_date_text(start),
            end=_to_date_text(end),
            interval="1d",
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )

        if raw.empty:
            raise ValueError(f"No Yahoo data returned for {ticker}")

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            if ticker in frame.columns.get_level_values(-1):
                frame = frame.xs(ticker, axis=1, level=-1)
            else:
                frame.columns = frame.columns.get_level_values(0)

        frame.columns = [str(col).strip() for col in frame.columns]
        frame.index = pd.to_datetime(frame.index)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Yahoo result index is not datetime-like")
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert(None)
        return frame.sort_index()


class CacheOnlyPriceProvider(PriceProvider):
    """Offline provider that serves data only from local parquet cache."""

    def get_price(self, ticker: str, start: DateLike, end: DateLike) -> PriceSeries:
        """Load one ticker price series from local cache only."""

        frame = self._load_cached_frame(
            namespace="price",
            symbol=ticker,
            start=start,
            end=end,
            required_columns=("close", "adj_close"),
        )
        series = PriceSeries(ticker=ticker, data=frame)
        check_price_integrity(series)
        return series

    def get_fx(
        self,
        pair: str = "HKDCNY",
        start: DateLike = "2000-01-01",
        end: DateLike = "2100-01-01",
    ) -> FxSeries:
        """Load FX series from cache without any network requests."""

        normalized_pair = pair.upper().replace("/", "")
        if normalized_pair not in {"HKDCNY", "CNYHKD"}:
            raise ValueError("Only HKDCNY or CNYHKD is supported")

        try:
            frame = self._load_cached_frame(
                namespace="fx",
                symbol=normalized_pair,
                start=start,
                end=end,
                required_columns=("close",),
            )
        except FileNotFoundError as exc:
            if normalized_pair != "CNYHKD":
                raise
            hkdcny = self._load_cached_frame(
                namespace="fx",
                symbol="HKDCNY",
                start=start,
                end=end,
                required_columns=("close",),
            )
            if (hkdcny["close"] == 0.0).any():
                raise ValueError("Cannot invert HKDCNY cache containing zeros") from exc
            frame = pd.DataFrame(index=hkdcny.index)
            frame["close"] = 1.0 / hkdcny["close"]

        series = FxSeries(pair=normalized_pair, data=frame)
        check_fx_integrity(series)
        return series

    def _load_cached_frame(
        self,
        namespace: str,
        symbol: str,
        start: DateLike,
        end: DateLike,
        required_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        """Load and slice the best local cache match for one symbol."""

        start_ts = _to_timestamp(start)
        end_ts = _to_timestamp(end)
        if start_ts > end_ts:
            raise ValueError("start must be <= end")

        candidates = self._candidate_cache_paths(
            namespace=namespace,
            symbol=symbol,
            start=start,
            end=end,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Offline cache miss for {namespace}/{symbol}. "
                "Run online fetch first or use fixtures."
            )

        best_full: pd.DataFrame | None = None
        best_sliced: pd.DataFrame | None = None
        best_overlap = -1

        for path in candidates:
            try:
                frame = pd.read_parquet(path)
                normalized = _normalize_cached_frame(frame, required_columns=required_columns)
            except Exception:  # noqa: BLE001
                continue

            if best_full is None or normalized.shape[0] > best_full.shape[0]:
                best_full = normalized

            sliced = normalized[(normalized.index >= start_ts) & (normalized.index <= end_ts)]
            overlap = int(sliced.shape[0])
            if overlap > best_overlap:
                best_overlap = overlap
                if overlap > 0:
                    best_sliced = sliced

        if best_sliced is not None:
            return best_sliced.sort_index()
        if best_full is not None:
            return best_full.sort_index()

        raise FileNotFoundError(
            f"No readable offline cache found for {namespace}/{symbol}. "
            "Run online fetch first or use fixtures."
        )

    def _candidate_cache_paths(
        self,
        namespace: str,
        symbol: str,
        start: DateLike,
        end: DateLike,
    ) -> list[Path]:
        """Collect exact and fuzzy cache candidates for one request."""

        exact = self._cache_path(namespace=namespace, symbol=symbol, start=start, end=end)
        pattern = f"{namespace}_{_slug(symbol)}_*.parquet"
        out: list[Path] = []
        if exact.exists():
            out.append(exact)

        for path in sorted(self.cache_dir.glob(pattern)):
            if path not in out:
                out.append(path)

        return out


def _to_date_text(value: DateLike) -> str:
    """Convert date-like value to ISO date string."""

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _slug(text: str) -> str:
    """Build filesystem-safe text snippet."""

    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    """Convert date-like value into timezone-naive timestamp."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _normalize_cached_frame(
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize cached frame shape and enforce required columns."""

    out = frame.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")

    out.index = pd.to_datetime(out.index)
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Cache index must be datetime-like")
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    out = out[~out.index.duplicated(keep="last")].sort_index()
    missing = [col for col in required_columns if col not in out.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Cache frame missing columns: {missing_text}")

    return out.loc[:, list(required_columns)].astype(float)
