"""Cost sensitivity backtests and HTML report generation for A/H pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ah_premium_lab.backtest.costs import CostParams, apply_costs_to_daily, compute_net_metrics
from ah_premium_lab.backtest.pairs_strategy import FxQuote, run_pairs_strategy


@dataclass(frozen=True)
class PairSensitivityOutput:
    """Pair-level sensitivity result."""

    pair_id: str
    grid_results: pd.DataFrame
    breakeven_cost_level: float
    breakeven_total_cost: float
    breakeven_slippage: float
    worst_case_net_dd: float


def run_pair_cost_sensitivity(
    pair_frame: pd.DataFrame,
    cost_grid: list[CostParams],
    *,
    pair_id: str,
    entry: float = 2.0,
    exit: float = 0.5,
    z_window: int = 252,
    share_ratio: float = 1.0,
    fx_quote: FxQuote = "HKDCNY",
    annualization: int = 252,
) -> PairSensitivityOutput:
    """Run one pair strategy under a grid of cost assumptions."""

    if not cost_grid:
        raise ValueError("cost_grid cannot be empty")

    base = run_pairs_strategy(
        pair_frame,
        entry=entry,
        exit=exit,
        z_window=z_window,
        cost_bps=0.0,
        share_ratio=share_ratio,
        fx_quote=fx_quote,
    )

    rows: list[dict[str, float | str]] = []
    for params in cost_grid:
        net_daily = apply_costs_to_daily(base.daily, params, annualization=annualization)
        metrics = compute_net_metrics(net_daily, annualization=annualization)

        row: dict[str, float | str] = {
            "pair_id": pair_id,
            **params.to_dict(),
            "net_cagr": metrics.net_cagr,
            "net_sharpe": metrics.net_sharpe,
            "max_dd": metrics.max_dd,
        }
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values(
        ["total_cost_level_bps", "commission_a_bps", "slippage_bps", "stamp_duty_a_bps"]
    )
    breakeven_total_cost = estimate_breakeven_total_cost(result_df)
    breakeven_slippage = estimate_breakeven_slippage(result_df)
    worst_case_net_dd = float(result_df["max_dd"].min()) if not result_df.empty else float("nan")
    result_df["breakeven_total_cost"] = breakeven_total_cost
    result_df["breakeven_slippage"] = breakeven_slippage
    result_df["worst_case_net_dd"] = worst_case_net_dd
    # Backward-compatible alias kept for existing report/dashboard paths.
    result_df["breakeven_cost_level"] = breakeven_total_cost

    return PairSensitivityOutput(
        pair_id=pair_id,
        grid_results=result_df.reset_index(drop=True),
        breakeven_cost_level=breakeven_total_cost,
        breakeven_total_cost=breakeven_total_cost,
        breakeven_slippage=breakeven_slippage,
        worst_case_net_dd=worst_case_net_dd,
    )


def run_universe_cost_sensitivity(
    frame: pd.DataFrame,
    cost_grid: list[CostParams],
    *,
    pair_col: str = "pair_id",
    entry: float = 2.0,
    exit: float = 0.5,
    z_window: int = 252,
    share_ratio: float = 1.0,
    fx_quote: FxQuote = "HKDCNY",
    annualization: int = 252,
) -> pd.DataFrame:
    """Run cost sensitivity across all A/H pairs in one DataFrame."""

    if pair_col not in frame.columns:
        raise KeyError(f"Missing pair column: {pair_col}")

    outputs: list[pd.DataFrame] = []
    for pair_id, pair_frame in frame.groupby(pair_col, sort=True):
        out = run_pair_cost_sensitivity(
            pair_frame=pair_frame,
            cost_grid=cost_grid,
            pair_id=str(pair_id),
            entry=entry,
            exit=exit,
            z_window=z_window,
            share_ratio=share_ratio,
            fx_quote=fx_quote,
            annualization=annualization,
        )
        outputs.append(out.grid_results)

    return pd.concat(outputs, ignore_index=True)


def estimate_breakeven_total_cost(result_df: pd.DataFrame) -> float:
    """Estimate breakeven total-cost level where net CAGR crosses zero.

    If no crossing exists in-grid:
    - all positive net_cagr: return NaN
    - all non-positive net_cagr: return minimum tested cost level
    """

    required = {"total_cost_level_bps", "net_cagr"}
    if not required.issubset(set(result_df.columns)):
        raise KeyError("result_df requires total_cost_level_bps and net_cagr")

    sorted_df = result_df[["total_cost_level_bps", "net_cagr"]].sort_values("total_cost_level_bps")
    return _estimate_zero_crossing(
        x=sorted_df["total_cost_level_bps"].to_numpy(dtype=float),
        y=sorted_df["net_cagr"].to_numpy(dtype=float),
    )


def estimate_breakeven_cost_level(result_df: pd.DataFrame) -> float:
    """Backward-compatible alias of `estimate_breakeven_total_cost`."""

    return estimate_breakeven_total_cost(result_df)


def estimate_breakeven_slippage(result_df: pd.DataFrame) -> float:
    """Estimate breakeven slippage level where average net CAGR crosses zero.

    Uses the 1D projection of `net_cagr` on `slippage_bps` by averaging
    over all other cost dimensions.
    """

    required = {"slippage_bps", "net_cagr"}
    if not required.issubset(set(result_df.columns)):
        raise KeyError("result_df requires slippage_bps and net_cagr")

    grouped = (
        result_df[["slippage_bps", "net_cagr"]]
        .groupby("slippage_bps", as_index=False)["net_cagr"]
        .mean()
        .sort_values("slippage_bps")
    )
    return _estimate_zero_crossing(
        x=grouped["slippage_bps"].to_numpy(dtype=float),
        y=grouped["net_cagr"].to_numpy(dtype=float),
    )


def generate_sensitivity_html_report(
    result_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "AH Cost Sensitivity Report",
) -> Path:
    """Generate pair-wise cost tolerance heatmaps and tables as one HTML report."""

    if result_df.empty:
        raise ValueError("result_df cannot be empty")

    required = {"pair_id", "commission_a_bps", "slippage_bps", "net_cagr", "net_sharpe", "max_dd"}
    if not required.issubset(set(result_df.columns)):
        missing = sorted(required - set(result_df.columns))
        raise KeyError(f"Missing required columns for report: {missing}")

    sections: list[str] = []
    include_plotlyjs = True

    for pair_id, group in result_df.groupby("pair_id", sort=True):
        heat_pivot = (
            group.groupby(["slippage_bps", "commission_a_bps"], as_index=True)["net_cagr"]
            .mean()
            .unstack("commission_a_bps")
            .sort_index()
        )

        fig = px.imshow(
            heat_pivot,
            labels={"x": "Commission A (bps)", "y": "Slippage (bps)", "color": "Avg Net CAGR"},
            title=f"{pair_id} Cost Tolerance Heatmap (avg over stamp duty levels)",
            aspect="auto",
            color_continuous_scale="RdYlGn",
        )
        heatmap_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if include_plotlyjs else False,
        )
        include_plotlyjs = False

        table_cols = [
            "commission_a_bps",
            "commission_h_bps",
            "stamp_duty_a_bps",
            "stamp_duty_h_bps",
            "slippage_bps",
            "total_cost_level_bps",
            "net_cagr",
            "net_sharpe",
            "max_dd",
            "borrow_bps",
            "breakeven_total_cost",
            "breakeven_slippage",
            "worst_case_net_dd",
            "breakeven_cost_level",
        ]
        table_cols = [col for col in table_cols if col in group.columns]
        table_html = (
            group[table_cols]
            .sort_values(
                ["total_cost_level_bps", "commission_a_bps", "slippage_bps", "stamp_duty_a_bps"]
            )
            .to_html(index=False, float_format=lambda x: f"{x:.4f}")
        )

        tolerance_df = _build_cost_tolerance_table(group)
        radar_html = _build_cost_tolerance_radar(
            pair_id=str(pair_id),
            tolerance_table=tolerance_df,
        )
        tolerance_table_html = tolerance_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")

        best_row = group.sort_values("net_cagr", ascending=False).iloc[0]
        breakeven_total = float(tolerance_df.loc[0, "breakeven_total_cost"])
        breakeven_slippage = float(tolerance_df.loc[0, "breakeven_slippage"])
        if pd.isna(breakeven_total):
            breakeven_total_text = "Not reached within tested grid"
        else:
            breakeven_total_text = f"{breakeven_total:.2f} bps"
        if pd.isna(breakeven_slippage):
            breakeven_slippage_text = "Not reached within tested grid"
        else:
            breakeven_slippage_text = f"{breakeven_slippage:.2f} bps"
        borrow_series = group.get("borrow_bps", pd.Series([0.0]))
        borrow_levels = sorted(float(item) for item in borrow_series.unique())
        borrow_text = ", ".join([f"{item:.2f}" for item in borrow_levels])

        conclusion = (
            f"Pair <strong>{pair_id}</strong>: best net CAGR appears at total cost level "
            f"<strong>{best_row['total_cost_level_bps']:.2f} bps</strong> with net CAGR "
            f"<strong>{best_row['net_cagr']:.4f}</strong>. Estimated breakeven total cost is "
            f"<strong>{breakeven_total_text}</strong>, breakeven slippage is "
            f"<strong>{breakeven_slippage_text}</strong>. "
            f"Borrow scenario levels (bps): <strong>{borrow_text}</strong>. "
            "This section is for research sensitivity analysis only and is not trading advice."
        )

        section_html = f"""
        <section class=\"pair-section\">
          <h2>{pair_id}</h2>
          <p class=\"conclusion\">{conclusion}</p>
          <div class=\"card\"><h3>Cost Tolerance Radar</h3>{radar_html}</div>
          <div class=\"card\"><h3>Cost Tolerance Table</h3>{tolerance_table_html}</div>
          <div class=\"card\">{heatmap_html}</div>
          <div class=\"card\">{table_html}</div>
        </section>
        """
        sections.append(section_html)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"UTF-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
      <title>{title}</title>
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
          margin: 26px;
          color: #111827;
          background: #f8fafc;
        }}
        h1 {{ margin-bottom: 4px; }}
        .meta {{ color: #4b5563; margin-bottom: 18px; }}
        .notice {{
          background: #fff7ed;
          border: 1px solid #fed7aa;
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 22px;
        }}
        .pair-section {{ margin-bottom: 30px; }}
        .card {{
          background: #ffffff;
          border-radius: 10px;
          padding: 14px;
          margin-top: 12px;
          box-shadow: 0 2px 8px rgba(15,23,42,.08);
          overflow-x: auto;
        }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 6px; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        .conclusion {{ margin-top: 6px; line-height: 1.45; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <p class=\"meta\">Generated at: {generated_at}</p>
      <div class=\"notice\">
        <strong>Research Use Only:</strong>
        This report is a cost sensitivity study for research and portfolio experimentation.
        It is <strong>not</strong> investment, execution, or trading advice.
      </div>
      {"".join(sections)}
    </body>
    </html>
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def np_all_positive(values: np.ndarray) -> bool:
    """Return whether all numeric values are strictly positive."""

    return bool((values > 0.0).all())


def _build_cost_tolerance_table(group: pd.DataFrame) -> pd.DataFrame:
    """Build one-row cost tolerance summary table for one pair."""

    breakeven_total_cost = (
        float(group["breakeven_total_cost"].iloc[0])
        if "breakeven_total_cost" in group.columns
        else estimate_breakeven_total_cost(group)
    )
    breakeven_slippage = (
        float(group["breakeven_slippage"].iloc[0])
        if "breakeven_slippage" in group.columns
        else estimate_breakeven_slippage(group)
    )
    worst_case_net_dd = (
        float(group["worst_case_net_dd"].iloc[0])
        if "worst_case_net_dd" in group.columns
        else float(group["max_dd"].min())
    )
    borrow_min = float(group["borrow_bps"].min()) if "borrow_bps" in group.columns else 0.0
    borrow_max = float(group["borrow_bps"].max()) if "borrow_bps" in group.columns else 0.0
    best_net_cagr = float(group["net_cagr"].max())
    median_net_sharpe = float(group["net_sharpe"].median())

    return pd.DataFrame(
        [
            {
                "breakeven_total_cost": breakeven_total_cost,
                "breakeven_slippage": breakeven_slippage,
                "worst_case_net_dd": worst_case_net_dd,
                "borrow_bps_min": borrow_min,
                "borrow_bps_max": borrow_max,
                "best_net_cagr": best_net_cagr,
                "median_net_sharpe": median_net_sharpe,
            }
        ]
    )


def _build_cost_tolerance_radar(
    *,
    pair_id: str,
    tolerance_table: pd.DataFrame,
) -> str:
    """Build radar chart HTML for cost tolerance profile."""

    row = tolerance_table.iloc[0]

    categories = [
        "Breakeven Total Cost (bps)",
        "Breakeven Slippage (bps)",
        "Borrow Max (bps)",
        "Best Net CAGR (%)",
        "Worst Net DD (%)",
    ]
    values = [
        _finite_or_zero(float(row["breakeven_total_cost"])),
        _finite_or_zero(float(row["breakeven_slippage"])),
        _finite_or_zero(float(row["borrow_bps_max"])),
        _finite_or_zero(float(row["best_net_cagr"]) * 100.0),
        abs(_finite_or_zero(float(row["worst_case_net_dd"]) * 100.0)),
    ]

    # Close the polar path
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=str(pair_id),
        )
    )
    fig.update_layout(
        title=f"{pair_id} Cost Tolerance Radar",
        polar={"radialaxis": {"visible": True}},
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _estimate_zero_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate x where y crosses zero using linear interpolation."""

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return float("nan")

    if np_all_positive(y):
        return float("nan")
    if (y <= 0.0).all():
        return float(np.min(x))

    for idx in range(1, len(y)):
        y0 = float(y[idx - 1])
        y1 = float(y[idx])
        if y0 > 0.0 and y1 <= 0.0:
            x0 = float(x[idx - 1])
            x1 = float(x[idx])
            if y1 == y0:
                return x1
            slope = (y1 - y0) / (x1 - x0)
            return float(x0 - y0 / slope)

    return float("nan")


def _finite_or_zero(value: float) -> float:
    """Replace non-finite numeric values with zero."""

    if np.isfinite(value):
        return float(value)
    return 0.0
