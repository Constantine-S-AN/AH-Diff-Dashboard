"""Tests for missing-day and abnormal-jump integrity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ah_premium_lab.data.integrity import check_price_integrity
from ah_premium_lab.data.models import PriceSeries


def test_integrity_detects_missing_days_and_jumps() -> None:
    """Integrity checker should warn on missing business days and extreme jump points."""

    dates = pd.bdate_range("2024-01-01", periods=200)
    dates = dates.delete(40)

    close = np.linspace(100.0, 110.0, len(dates))
    close[120] = close[119] * 3.0

    frame = pd.DataFrame(
        {
            "close": close,
            "adj_close": close,
        },
        index=dates,
    )

    series = PriceSeries(ticker="TEST", data=frame)

    with pytest.warns(UserWarning) as captured:
        result = check_price_integrity(series, zscore_threshold=8.0)

    messages = [str(item.message) for item in captured]
    assert result.missing_days == 1
    assert result.jump_count >= 1
    assert any("missing business days" in msg for msg in messages)
    assert any("abnormal jumps" in msg for msg in messages)
