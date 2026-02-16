"""Tests for HTML sensitivity report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ah_premium_lab.backtest import generate_sensitivity_html_report


def test_generate_sensitivity_html_report_writes_file(tmp_path: Path) -> None:
    """Report generator should output non-empty HTML file."""

    result_df = pd.DataFrame(
        {
            "pair_id": ["P1", "P1", "P2", "P2"],
            "commission_a_bps": [0.0, 5.0, 0.0, 5.0],
            "commission_h_bps": [0.0, 5.0, 0.0, 5.0],
            "stamp_duty_a_bps": [0.0, 5.0, 0.0, 5.0],
            "stamp_duty_h_bps": [0.0, 5.0, 0.0, 5.0],
            "slippage_bps": [0.0, 10.0, 0.0, 10.0],
            "total_cost_level_bps": [0.0, 30.0, 0.0, 30.0],
            "net_cagr": [0.12, 0.02, 0.08, -0.01],
            "net_sharpe": [1.2, 0.5, 1.0, 0.2],
            "max_dd": [-0.12, -0.20, -0.10, -0.18],
            "breakeven_cost_level": [25.0, 25.0, 18.0, 18.0],
            "breakeven_total_cost": [25.0, 25.0, 18.0, 18.0],
            "breakeven_slippage": [7.5, 7.5, 5.5, 5.5],
            "worst_case_net_dd": [-0.20, -0.20, -0.18, -0.18],
            "borrow_bps": [0.0, 0.0, 0.0, 0.0],
        }
    )

    out_path = tmp_path / "sensitivity_report.html"
    path = generate_sensitivity_html_report(result_df=result_df, output_path=out_path)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Research Use Only" in text
    assert "P1" in text
    assert "P2" in text
    assert "Cost Tolerance Radar" in text
