"""Loader helpers for selecting pair universe sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ah_premium_lab.data import load_ah_pairs
from ah_premium_lab.data.models import AhPair

DEFAULT_PAIRS_MASTER_PATH = Path("data/pairs_master.csv")
DEFAULT_PAIRS_FALLBACK_PATH = Path("data/pairs.csv")


def resolve_universe_path(
    master_path: str | Path = DEFAULT_PAIRS_MASTER_PATH,
    fallback_path: str | Path = DEFAULT_PAIRS_FALLBACK_PATH,
) -> Path:
    """Resolve preferred universe path.

    Priority:
    1) `pairs_master.csv`
    2) fallback `pairs.csv`
    """

    master = Path(master_path)
    fallback = Path(fallback_path)

    if master.exists():
        return master
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No universe file found. Checked: {master.as_posix()}, {fallback.as_posix()}"
    )


def load_universe_pairs(
    master_path: str | Path = DEFAULT_PAIRS_MASTER_PATH,
    fallback_path: str | Path = DEFAULT_PAIRS_FALLBACK_PATH,
) -> list[AhPair]:
    """Load universe A/H pairs with master/fallback behavior."""

    target = resolve_universe_path(master_path=master_path, fallback_path=fallback_path)
    return load_ah_pairs(target)


def load_universe_frame(
    master_path: str | Path = DEFAULT_PAIRS_MASTER_PATH,
    fallback_path: str | Path = DEFAULT_PAIRS_FALLBACK_PATH,
) -> pd.DataFrame:
    """Load universe into DataFrame for dashboard rendering."""

    rows: list[dict[str, object]] = []
    for pair in load_universe_pairs(master_path=master_path, fallback_path=fallback_path):
        pair_id = f"{pair.a_ticker}-{pair.h_ticker}"
        rows.append(
            {
                "pair_id": pair_id,
                "name": pair.name,
                "a_ticker": pair.a_ticker,
                "h_ticker": pair.h_ticker,
                "share_ratio": float(pair.share_ratio),
                "notes": pair.notes,
            }
        )

    return pd.DataFrame(rows)
