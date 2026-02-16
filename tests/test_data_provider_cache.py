"""Tests for provider-level parquet cache behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import ah_premium_lab.data.providers as providers_module
from ah_premium_lab.data.providers import YahooFinanceProvider


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
