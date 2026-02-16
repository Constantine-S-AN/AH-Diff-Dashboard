"""Tests for premium calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ah_premium_lab.core import prepare_spread_frame


def test_prepare_spread_frame_adds_columns() -> None:
    """Premium preparation should add premium and zscore columns."""

    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    h_close = np.linspace(80.0, 100.0, len(dates))
    fx_rate = np.full(len(dates), 1.08)
    a_close = (h_close / fx_rate) * np.exp(np.sin(np.linspace(0, 5, len(dates))) * 0.02)

    frame = pd.DataFrame(
        {
            "date": dates,
            "pair_id": "x-y",
            "a_close": a_close,
            "h_close": h_close,
            "fx_rate": fx_rate,
        }
    )

    out = prepare_spread_frame(frame, method="log_ratio", zscore_window=20)

    assert "premium" in out.columns
    assert "zscore" in out.columns
    assert out["premium"].notna().all()
    assert out["zscore"].notna().sum() > 10
