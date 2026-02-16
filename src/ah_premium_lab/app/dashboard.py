"""Streamlit dashboard for AH premium research."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ah_premium_lab.backtest import simulate_mean_reversion
from ah_premium_lab.config import LabConfig, load_config, resolve_project_root
from ah_premium_lab.core import prepare_spread_frame
from ah_premium_lab.data import load_market_data
from ah_premium_lab.report import run_cost_sensitivity_scan
from ah_premium_lab.stats import run_pair_diagnostics


@st.cache_data(show_spinner=False)
def _load_dataset(config_path: str) -> tuple[LabConfig, pd.DataFrame]:
    """Load configuration and prepared spread data."""

    config = load_config(config_path)
    raw = load_market_data(config)
    prepared = prepare_spread_frame(
        raw,
        method=config.core.premium_method,
        zscore_window=config.core.zscore_window,
    )
    return config, prepared


def _parse_cli_config_path() -> Path:
    """Parse optional `--config` passed after Streamlit's `--`."""

    parser = argparse.ArgumentParser(add_help=False)
    default_path = resolve_project_root(Path(__file__)) / "config" / "default.yaml"
    parser.add_argument("--config", default=str(default_path))
    args, _ = parser.parse_known_args()
    return Path(args.config)


def _latest_overview_table(prepared: pd.DataFrame, config: LabConfig) -> pd.DataFrame:
    """Build pair-level overview table using latest observation and diagnostics."""

    latest = prepared.sort_values("date").groupby("pair_id", as_index=False).tail(1)

    rows: list[dict[str, float | str | bool]] = []
    for _, row in latest.iterrows():
        pair_id = str(row["pair_id"])
        pair_frame = prepared[prepared["pair_id"] == pair_id]
        diag = run_pair_diagnostics(pair_frame, config.stats)

        rows.append(
            {
                "pair_id": pair_id,
                "latest_premium": float(row["premium"]),
                "latest_zscore": float(row["zscore"]),
                "adf_pvalue": diag.adf_pvalue,
                "half_life_days": diag.half_life_days,
                "coint_pvalue": diag.coint_pvalue,
                "stationary": diag.is_stationary,
                "cointegrated": diag.is_cointegrated,
            }
        )

    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def main() -> None:
    """Render Streamlit dashboard."""

    st.set_page_config(page_title="AH Premium Lab", page_icon="📊", layout="wide")
    st.title("AH Premium Lab")
    st.caption("A/H 价差仪表盘 + 统计检验 + 成本敏感性（研究版）")

    config_path = _parse_cli_config_path()
    config, prepared = _load_dataset(str(config_path))

    pair_ids = sorted(prepared["pair_id"].unique().tolist())
    selected_pair = st.sidebar.selectbox("选择股票对", options=pair_ids)

    st.subheader("总览")
    overview = _latest_overview_table(prepared, config)
    st.dataframe(overview, use_container_width=True)

    st.subheader("单对详情")
    pair_frame = prepared[prepared["pair_id"] == selected_pair].copy()
    diag = run_pair_diagnostics(pair_frame, config.stats)

    col1, col2, col3 = st.columns(3)
    col1.metric("ADF p-value", f"{diag.adf_pvalue:.4f}")
    col2.metric("Half-life (days)", f"{diag.half_life_days:.2f}")
    col3.metric("Engle-Granger p-value", f"{diag.coint_pvalue:.4f}")

    premium_fig = px.line(
        pair_frame,
        x="date",
        y="premium",
        title=f"Premium Time Series - {selected_pair}",
    )
    st.plotly_chart(premium_fig, use_container_width=True)

    zscore_fig = px.line(
        pair_frame,
        x="date",
        y="zscore",
        title=f"Z-score - {selected_pair}",
    )
    zscore_fig.add_hline(y=config.backtest.entry_z, line_dash="dash", line_color="red")
    zscore_fig.add_hline(y=-config.backtest.entry_z, line_dash="dash", line_color="red")
    zscore_fig.add_hline(y=config.backtest.exit_z, line_dash="dot", line_color="gray")
    zscore_fig.add_hline(y=-config.backtest.exit_z, line_dash="dot", line_color="gray")
    st.plotly_chart(zscore_fig, use_container_width=True)

    default_cost = float(config.report.cost_grid_bps[0])
    bt_result = simulate_mean_reversion(pair_frame, config.backtest, cost_bps=default_cost)

    col4, col5, col6 = st.columns(3)
    col4.metric("Cumulative Return", f"{bt_result.summary.cumulative_return:.4f}")
    col5.metric("Sharpe", f"{bt_result.summary.annualized_sharpe:.3f}")
    col6.metric("Max Drawdown", f"{bt_result.summary.max_drawdown:.4f}")

    equity_fig = px.line(
        bt_result.equity_curve,
        x="date",
        y="equity",
        title="Backtest Equity Curve",
    )
    st.plotly_chart(equity_fig, use_container_width=True)

    scan_df = run_cost_sensitivity_scan(pair_frame, config)
    cost_fig = px.line(
        scan_df,
        x="cost_bps",
        y="cumulative_return",
        markers=True,
        title="成本敏感性扫描",
    )
    st.plotly_chart(cost_fig, use_container_width=True)


if __name__ == "__main__":
    main()
