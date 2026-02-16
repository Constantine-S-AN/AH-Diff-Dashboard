"""Tests for provider-level parquet cache behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import ah_premium_lab.data.providers as providers_module
from ah_premium_lab.data.providers import CacheOnlyPriceProvider, YahooFinanceProvider


def test_yahoo_provider_writes_and_reads_cache(tmp_path: Path, monkeypatch) -> None:
    """Yahoo provider should fetch once and reuse parquet cache for same request."""

    calls = {"count": 0}

    def fake_download(**_: object) -> pd.DataFrame:
        calls["count"] += 1
        index = pd.bdate_range("2024-01-01", periods=6)
        return pd.DataFrame(
            {
                "Open": [10, 10, 10, 10, 10, 10],
                "High": [11, 11, 11, 11, 11, 11],
                "Low": [9, 9, 9, 9, 9, 9],
                "Close": [10, 11, 12, 13, 14, 15],
                "Adj Close": [10, 11, 12, 13, 14, 15],
                "Volume": [100, 100, 100, 100, 100, 100],
            },
            index=index,
        )

    monkeypatch.setattr(providers_module.yf, "download", fake_download)

    provider = YahooFinanceProvider(cache_dir=tmp_path)
    first = provider.get_price("TEST", start="2024-01-01", end="2024-01-31")
    second = provider.get_price("TEST", start="2024-01-01", end="2024-01-31")

    assert calls["count"] == 1
    assert len(list(tmp_path.glob("price_*.parquet"))) == 1
    pd.testing.assert_frame_equal(first.data, second.data, check_freq=False)


def test_cache_only_provider_reads_local_cache_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cache-only provider should return cached prices and never call yfinance."""

    calls = {"count": 0}

    def fail_download(**_: object) -> pd.DataFrame:
        calls["count"] += 1
        raise AssertionError("yfinance.download should not be called in offline mode")

    monkeypatch.setattr(providers_module.yf, "download", fail_download)

    provider = CacheOnlyPriceProvider(cache_dir=tmp_path)
    cache_path = provider._cache_path("price", "TEST", "2024-01-01", "2024-01-31")
    frame = pd.DataFrame(
        {"close": [10.0, 10.5, 11.0], "adj_close": [10.0, 10.5, 11.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    frame.to_parquet(cache_path)

    exact = provider.get_price("TEST", "2024-01-01", "2024-01-31")
    fuzzy = provider.get_price("TEST", "2024-02-01", "2024-02-29")

    assert calls["count"] == 0
    pd.testing.assert_frame_equal(exact.data, frame, check_freq=False)
    pd.testing.assert_frame_equal(fuzzy.data, frame, check_freq=False)


def test_cache_only_provider_can_infer_cnyhkd_from_hkdcny_cache(tmp_path: Path) -> None:
    """Cache-only provider should invert HKDCNY cache when CNYHKD cache is missing."""

    provider = CacheOnlyPriceProvider(cache_dir=tmp_path)
    cache_path = provider._cache_path("fx", "HKDCNY", "2024-01-01", "2024-01-31")
    hkdcny = pd.DataFrame(
        {"close": [0.90, 0.92, 0.95]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    hkdcny.to_parquet(cache_path)

    cnyhkd = provider.get_fx("CNYHKD", "2024-01-01", "2024-01-31")
    expected = pd.DataFrame({"close": 1.0 / hkdcny["close"]}, index=hkdcny.index)
    pd.testing.assert_frame_equal(cnyhkd.data, expected, check_freq=False)
