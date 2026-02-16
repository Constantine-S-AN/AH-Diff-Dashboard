"""A/H pair universe loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ah_premium_lab.data.models import AhPair

DEFAULT_PAIRS_PATH = Path("data/pairs.csv")


def load_ah_pairs(path: str | Path = DEFAULT_PAIRS_PATH) -> list[AhPair]:
    """Load A/H pair metadata from CSV.

    CSV schema:
    - name
    - a_ticker
    - h_ticker
    - share_ratio (optional, default 1.0)
    - notes (optional)
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"name", "a_ticker", "h_ticker"}
    if not required.issubset(set(frame.columns)):
        raise ValueError("pairs.csv must contain columns: name, a_ticker, h_ticker")

    pairs: list[AhPair] = []
    for _, row in frame.iterrows():
        share_ratio_raw = row.get("share_ratio", 1.0)
        notes_raw = row.get("notes", "")

        share_ratio = 1.0 if pd.isna(share_ratio_raw) else float(share_ratio_raw)
        notes = "" if pd.isna(notes_raw) else str(notes_raw)

        pairs.append(
            AhPair(
                name=str(row["name"]),
                a_ticker=str(row["a_ticker"]),
                h_ticker=str(row["h_ticker"]),
                share_ratio=share_ratio,
                notes=notes,
            )
        )

    return pairs
