"""Cost sensitivity grid scan and HTML report rendering."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

from ah_premium_lab.backtest import simulate_mean_reversion
from ah_premium_lab.config import LabConfig


def run_cost_sensitivity_scan(frame: pd.DataFrame, config: LabConfig) -> pd.DataFrame:
    """Run backtests across configured transaction cost grid."""

    rows: list[dict[str, float | int]] = []
    for cost_bps in config.report.cost_grid_bps:
        result = simulate_mean_reversion(frame=frame, config=config.backtest, cost_bps=cost_bps)
        rows.append(
            {
                "cost_bps": float(cost_bps),
                "cumulative_return": result.summary.cumulative_return,
                "annualized_sharpe": result.summary.annualized_sharpe,
                "max_drawdown": result.summary.max_drawdown,
                "trade_count": result.summary.trade_count,
            }
        )
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def generate_cost_sensitivity_report(
    pair_id: str,
    frame: pd.DataFrame,
    config: LabConfig,
    output_path: str | Path,
) -> Path:
    """Generate HTML report for cost sensitivity analysis."""

    result_df = run_cost_sensitivity_scan(frame=frame, config=config)
    best_row = result_df.sort_values("cumulative_return", ascending=False).iloc[0].to_dict()

    table_html = result_df.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    chart_html = _build_plotly_chart(result_df, include_plotly=config.report.include_plotly)

    env = Environment(
        loader=PackageLoader("ah_premium_lab", "report/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("cost_report.html.j2")

    rendered = template.render(
        pair_id=pair_id,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        best_row=best_row,
        table_html=table_html,
        chart_html=chart_html,
        config=asdict(config.report),
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


def _build_plotly_chart(result_df: pd.DataFrame, include_plotly: bool) -> str:
    """Build embedded Plotly chart HTML for cumulative return vs cost."""

    if not include_plotly:
        return ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result_df["cost_bps"],
            y=result_df["cumulative_return"],
            mode="lines+markers",
            name="Cumulative Return",
        )
    )
    fig.update_layout(
        title="Cost Sensitivity",
        xaxis_title="Cost (bps)",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
