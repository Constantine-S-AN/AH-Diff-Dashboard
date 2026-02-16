"""Utilities for deterministic research regression fixtures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ah_premium_lab.backtest import run_pairs_strategy
from ah_premium_lab.core import compute_premium_metrics
from ah_premium_lab.data import AhPair, load_ah_pairs
from ah_premium_lab.stats import adf_test, engle_granger_test, half_life_ar1, summary_score


@dataclass(frozen=True)
class RegressionParams:
    """Fixed parameters for deterministic research regression checks."""

    start_date: str
    end_date: str
    fx_pair: str
    window: int
    entry: float
    exit: float
    cost_bps: float


FIXTURE_DIR = Path("tests/fixtures/research_regression")
PAIRS_PATH = FIXTURE_DIR / "pairs.csv"
PARAMS_PATH = FIXTURE_DIR / "params.yaml"
GOLDEN_PATH = FIXTURE_DIR / "golden_metrics.json"


def load_regression_params(path: Path = PARAMS_PATH) -> RegressionParams:
    """Load fixed regression parameters from YAML fixture."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RegressionParams(
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]),
        fx_pair=str(payload["fx_pair"]),
        window=int(payload["window"]),
        entry=float(payload["entry"]),
        exit=float(payload["exit"]),
        cost_bps=float(payload["cost_bps"]),
    )


def compute_fixture_metrics(
    pairs: list[AhPair],
    params: RegressionParams,
) -> dict[str, dict[str, float]]:
    """Compute deterministic per-pair regression metrics."""

    date_index = pd.bdate_range(start=params.start_date, end=params.end_date)
    fx_series = _generate_fx_series(date_index)

    outputs: dict[str, dict[str, float]] = {}

    for pair in pairs:
        pair_id = f"{pair.a_ticker}-{pair.h_ticker}"
        h_price_hkd, spread = _generate_pair_h_and_spread(pair_id, date_index)

        h_cny = h_price_hkd * fx_series
        a_price_cny = np.exp(spread) * h_cny * float(pair.share_ratio)

        aligned = compute_premium_metrics(
            a_price_cny=a_price_cny,
            h_price_hkd=h_price_hkd,
            fx=fx_series,
            fx_quote=params.fx_pair,
            share_ratio=float(pair.share_ratio),
            window=params.window,
        )

        adf_p, _, _ = adf_test(aligned["log_spread"])
        eg_p, _, _ = engle_granger_test(
            log_A=np.log(aligned["a_price_cny"]),
            log_H_fx=np.log(aligned["h_price_cny_equiv"]),
        )
        half_life = half_life_ar1(aligned["log_spread"])
        score = summary_score(
            adf_p_value=adf_p,
            eg_p_value=eg_p,
            half_life_days=half_life,
            zscore_series=aligned["rolling_zscore"],
        )

        strategy_frame = aligned.reset_index()
        first_column = str(strategy_frame.columns[0])
        strategy_frame = strategy_frame.rename(columns={first_column: "date"})
        strategy_frame = strategy_frame[["date", "a_price_cny", "h_price_cny_equiv"]].rename(
            columns={"a_price_cny": "a_close", "h_price_cny_equiv": "h_cny"}
        )

        strategy = run_pairs_strategy(
            strategy_frame,
            entry=params.entry,
            exit=params.exit,
            z_window=params.window,
            cost_bps=params.cost_bps,
        )

        latest = aligned.iloc[-1]
        outputs[pair_id] = {
            "latest_premium_pct": float(latest["premium_pct"] * 100.0),
            "adf_p_value": float(adf_p),
            "eg_p_value": float(eg_p),
            "half_life_days": float(half_life),
            "summary_score": float(score),
            "strategy_end_nav": float(strategy.daily["curve_net"].iloc[-1]),
            "strategy_max_dd": float(strategy.max_drawdown_net),
        }

    return dict(sorted(outputs.items()))


def _generate_fx_series(index: pd.DatetimeIndex) -> pd.Series:
    """Generate a deterministic HKD->CNY FX series."""

    t = np.arange(len(index), dtype=float)
    fx_values = 0.905 + 0.012 * np.sin(t / 37.0) + 0.004 * np.cos(t / 91.0)
    return pd.Series(fx_values, index=index, name="fx_hkd_to_cny")


def _generate_pair_h_and_spread(
    pair_id: str,
    index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Generate deterministic H price and log spread for one pair."""

    seed = _stable_seed(pair_id)
    rng = np.random.default_rng(seed)

    t = np.arange(len(index), dtype=float)
    base_h = 8.0 + (seed % 140) / 8.0
    drift = 0.0002 + (seed % 7) * 0.00002
    seasonal = 0.0012 * np.sin(t / 33.0 + (seed % 19) / 10.0)
    noise = rng.normal(loc=0.0, scale=0.0085, size=len(index))
    h_log_ret = drift + seasonal + noise
    h_price = base_h * np.exp(np.cumsum(h_log_ret))

    phi = 0.955 + (seed % 5) * 0.004
    sigma = 0.020 + (seed % 3) * 0.003
    spread_target = 0.16 + 0.06 * np.sin(t / 79.0 + (seed % 23) / 11.0)

    spread = np.zeros(len(index), dtype=float)
    for idx in range(1, len(index)):
        innovation = rng.normal(loc=0.0, scale=sigma)
        spread[idx] = phi * spread[idx - 1] + (1.0 - phi) * spread_target[idx] + innovation

    h_series = pd.Series(h_price, index=index, name="h_price_hkd")
    spread_series = pd.Series(spread, index=index, name="log_spread")
    return h_series, spread_series


def _stable_seed(text: str) -> int:
    """Build stable integer seed from text."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def load_regression_pairs(path: Path = PAIRS_PATH) -> list[AhPair]:
    """Load fixed pair universe for regression fixture."""

    return load_ah_pairs(path)
