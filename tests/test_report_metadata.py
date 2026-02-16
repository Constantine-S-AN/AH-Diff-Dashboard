"""Tests for report metadata embedding in generated HTML."""

from __future__ import annotations

import pandas as pd

from ah_premium_lab.report.generate_report import (
    PairReportResult,
    ReportMetadata,
    _render_index_html,
    _render_single_report_html,
)


def test_report_html_includes_metadata_section() -> None:
    """Single/index report HTML should include metadata keys and values."""

    aligned = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "premium_pct": [0.1, 0.2, 0.15, 0.18, 0.12],
            "log_spread": [0.095, 0.182, 0.14, 0.165, 0.113],
            "rolling_zscore": [0.0, 0.8, -0.2, 0.5, -0.1],
        }
    )

    sensitivity = pd.DataFrame(
        {
            "commission_a_bps": [0.0],
            "commission_h_bps": [0.0],
            "stamp_duty_a_bps": [0.0],
            "stamp_duty_h_bps": [0.0],
            "slippage_bps": [0.0],
            "total_cost_level_bps": [0.0],
            "net_cagr": [0.11],
            "net_sharpe": [1.1],
            "max_dd": [-0.09],
            "breakeven_cost_level": [5.5],
        }
    )

    pair = PairReportResult(
        pair_id="600036.SS-3968.HK",
        name="China Merchants Bank",
        a_ticker="600036.SS",
        h_ticker="3968.HK",
        notes="metadata-test",
        sample_start="2024-01-01",
        sample_end="2024-01-05",
        aligned_sample_size=5,
        latest_premium_pct=12.0,
        latest_rolling_z=0.2,
        latest_premium_percentile=0.7,
        half_life_days=15.0,
        adf_p_value=0.04,
        adf_stat=-3.2,
        adf_used_lags=1,
        eg_p_value=0.03,
        eg_beta=0.95,
        summary_score=88.0,
        a_missing_days=0,
        h_missing_days=1,
        fx_missing_days=2,
        a_missing_rate=0.0,
        h_missing_rate=0.02,
        fx_missing_rate=0.04,
        integrity_warnings=("h-missing",),
        aligned_frame=aligned,
        sensitivity_df=sensitivity,
        breakeven_cost_level=5.5,
    )

    metadata = ReportMetadata(
        generated_at_utc="2026-02-16 00:00:00 UTC",
        data_fetch_time_utc="2026-02-16 00:00:00 UTC",
        git_commit="abc1234",
        start_date="2024-01-01",
        end_date="2024-12-31",
        expected_business_days=252,
        pair_count=1,
        total_aligned_samples=5,
        mean_aligned_samples=5.0,
        avg_a_missing_rate=0.0,
        avg_h_missing_rate=0.02,
        avg_fx_missing_rate=0.04,
        fx_pair="HKDCNY",
        fx_alignment="direct: HKD->CNY = FX",
        cost_grid_size=1,
        cost_parameters={"commission_min_bps": 0.0, "commission_max_bps": 10.0},
    )

    single_html = _render_single_report_html(
        [pair],
        metadata=metadata,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    index_html = _render_index_html(
        [pair],
        [(pair.pair_id, "pairs/600036.SS-3968.HK.html")],
        metadata=metadata,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert "Metadata" in single_html
    assert "abc1234" in single_html
    assert "fx_alignment" in single_html
    assert "cost_parameters" in single_html

    assert "Metadata" in index_html
    assert "abc1234" in index_html
    assert "HKDCNY" in index_html
