"""Mapping override registry for unresolved tickers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING_OVERRIDES_PATH = Path("data/mapping_overrides.csv")

MAPPING_COLUMNS = [
    "updated_at_utc",
    "ticker",
    "suggested_ticker",
    "status",
    "reason",
]


class MappingRequiredError(RuntimeError):
    """Raised when ticker mapping is missing and requires manual override."""


def load_mapping_overrides(
    path: str | Path = DEFAULT_MAPPING_OVERRIDES_PATH,
) -> pd.DataFrame:
    """Load mapping override table or return empty schema."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    frame = pd.read_csv(csv_path, dtype=str)
    for col in MAPPING_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""

    return frame[MAPPING_COLUMNS].fillna("")


def record_mapping_issue(
    *,
    ticker: str,
    reason: str,
    suggested_ticker: str = "",
    status: str = "needs_manual_fix",
    path: str | Path = DEFAULT_MAPPING_OVERRIDES_PATH,
) -> Path:
    """Upsert one ticker issue into mapping override file."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = load_mapping_overrides(csv_path)
    now_text = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    mask = table["ticker"].astype(str) == str(ticker)
    row_payload = {
        "updated_at_utc": now_text,
        "ticker": str(ticker),
        "suggested_ticker": str(suggested_ticker),
        "status": str(status),
        "reason": str(reason),
    }

    if mask.any():
        idx = table[mask].index[-1]
        for key, value in row_payload.items():
            table.loc[idx, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([row_payload])], ignore_index=True)

    table = table[MAPPING_COLUMNS]
    table.to_csv(csv_path, index=False)
    return csv_path
