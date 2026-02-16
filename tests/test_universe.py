"""Tests for universe loader, mapping overrides, and quality diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ah_premium_lab.universe import (
    combine_pair_quality,
    compute_series_quality,
    load_universe_pairs,
    record_mapping_issue,
    resolve_universe_path,
)


def test_resolve_universe_prefers_master(tmp_path: Path) -> None:
    """Master universe should be preferred when both files exist."""

    master = tmp_path / "pairs_master.csv"
    fallback = tmp_path / "pairs.csv"

    master.write_text(
        "name,a_ticker,h_ticker,share_ratio,notes\nMaster,A1,H1,1.0,master\n",
        encoding="utf-8",
    )
    fallback.write_text(
        "name,a_ticker,h_ticker,share_ratio,notes\nFallback,A2,H2,1.0,fallback\n",
        encoding="utf-8",
    )

    resolved = resolve_universe_path(master_path=master, fallback_path=fallback)
    pairs = load_universe_pairs(master_path=master, fallback_path=fallback)

    assert resolved == master
    assert len(pairs) == 1
    assert pairs[0].name == "Master"


def test_record_mapping_issue_upsert(tmp_path: Path) -> None:
    """Mapping override writer should upsert by ticker."""

    mapping_file = tmp_path / "mapping_overrides.csv"

    record_mapping_issue(
        ticker="600000.SS",
        reason="initial failure",
        path=mapping_file,
    )
    record_mapping_issue(
        ticker="600000.SS",
        reason="retry failure",
        suggested_ticker="600000.SH",
        path=mapping_file,
    )

    frame = pd.read_csv(mapping_file)
    assert frame.shape[0] == 1
    assert frame.loc[0, "ticker"] == "600000.SS"
    assert frame.loc[0, "reason"] == "retry failure"
    assert frame.loc[0, "suggested_ticker"] == "600000.SH"


def test_quality_metrics_and_missing_threshold_flag() -> None:
    """Quality diagnostics should expose coverage/gap and threshold breach."""

    idx = pd.bdate_range("2024-01-01", periods=10)
    values = np.array([10.0, 10.1, 10.2, 10.3, 15.0, 10.4, 10.5, 10.6, 10.7, 10.8])

    a_series = pd.Series(values, index=idx).drop(index=[idx[3], idx[4]])
    h_series = pd.Series(values * 0.9, index=idx)
    fx_series = pd.Series(np.full(len(idx), 0.91), index=idx)

    a_quality = compute_series_quality(
        a_series,
        start_date="2024-01-01",
        end_date="2024-01-12",
    )
    h_quality = compute_series_quality(
        h_series,
        start_date="2024-01-01",
        end_date="2024-01-12",
    )
    fx_quality = compute_series_quality(
        fx_series,
        start_date="2024-01-01",
        end_date="2024-01-12",
    )

    assert np.isclose(a_quality.coverage_pct, 80.0)
    assert a_quality.max_gap_days >= 2

    pair_quality = combine_pair_quality(
        a_quality,
        h_quality,
        fx_quality,
        missing_threshold=0.15,
    )

    assert pair_quality.missing_threshold_breached is True
    assert pair_quality.quality_flag == "poor"
    assert pair_quality.coverage_pct <= 80.0


def test_pairs_master_contains_large_universe() -> None:
    """Master universe fixture should contain at least 50 pairs."""

    frame = pd.read_csv("data/pairs_master.csv")
    assert len(frame) >= 50
