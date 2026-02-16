"""Data loading utilities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ah_premium_lab.config import LabConfig, PairConfig


def load_market_data(config: LabConfig, pair_id: str | None = None) -> pd.DataFrame:
    """Load market data for configured A/H pairs.

    Args:
        config: Project configuration.
        pair_id: Optional pair identifier filter.

    Returns:
        Market data with one row per date/pair.

    Raises:
        ValueError: If configured source is unsupported.
    """

    if config.data.source != "synthetic":
        raise ValueError(
            "Only synthetic source is implemented in this research skeleton. "
            "Set data.source='synthetic'."
        )

    frames: list[pd.DataFrame] = []
    for pair_cfg in config.data.pairs:
        if pair_id is not None and pair_cfg.pair_id != pair_id:
            continue
        frames.append(_generate_synthetic_pair_series(pair_cfg, config))

    if not frames:
        raise ValueError(f"No data generated for pair filter: {pair_id}")

    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["pair_id", "date"]).reset_index(drop=True)
    return output


def _generate_synthetic_pair_series(pair: PairConfig, config: LabConfig) -> pd.DataFrame:
    """Generate synthetic but coherent A/H market series for one pair."""

    dates = pd.date_range(
        start=config.data.start_date,
        end=config.data.end_date,
        freq=config.data.frequency,
    )

    n_obs = len(dates)
    seed = _stable_pair_seed(pair.pair_id, config.data.seed)
    rng = np.random.default_rng(seed)

    h_log_returns = rng.normal(loc=0.00025, scale=0.015, size=n_obs)
    fx_log_returns = rng.normal(loc=0.0, scale=0.0015, size=n_obs)

    h_base = 50.0 + rng.uniform(10.0, 80.0)
    h_close = h_base * np.exp(np.cumsum(h_log_returns))

    fx_base = 1.08
    fx_rate = fx_base * np.exp(np.cumsum(fx_log_returns))

    premium_noise = rng.normal(loc=0.0, scale=0.03, size=n_obs)
    premium_signal = pd.Series(premium_noise).rolling(5, min_periods=1).mean().to_numpy()

    h_in_cny = h_close / fx_rate
    a_close = h_in_cny * np.exp(premium_signal)

    return pd.DataFrame(
        {
            "date": dates,
            "pair_id": pair.pair_id,
            "a_symbol": pair.a_symbol,
            "h_symbol": pair.h_symbol,
            "fx_symbol": pair.fx_symbol,
            "a_close": a_close,
            "h_close": h_close,
            "fx_rate": fx_rate,
        }
    )


def _stable_pair_seed(pair_id: str, base_seed: int) -> int:
    """Build deterministic pair-specific RNG seed."""

    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    pair_seed = int.from_bytes(digest[:4], "little")
    return (pair_seed + base_seed) % (2**32 - 1)
