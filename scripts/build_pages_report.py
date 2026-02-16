"""Build deterministic demo report assets for GitHub Pages publishing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from make_demo_assets import generate_demo_assets

from ah_premium_lab.backtest import (
    generate_cost_grid,
    generate_sensitivity_html_report,
    run_pair_cost_sensitivity,
)


def build_pages_docs(output_dir: Path) -> Path:
    """Build docs site with deterministic demo report and image assets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    generate_demo_assets(screenshot_dir)
    sensitivity_frame = _build_demo_sensitivity_frame()
    report_path = report_dir / "cost_sensitivity_demo.html"
    generate_sensitivity_html_report(
        result_df=sensitivity_frame,
        output_path=report_path,
        title="AH Premium Lab Demo Cost Sensitivity Report",
    )

    index_path = output_dir / "index.html"
    index_path.write_text(_render_index_html(), encoding="utf-8")
    return index_path


def _build_demo_sensitivity_frame() -> pd.DataFrame:
    """Create deterministic pair-level sensitivity table for Pages demo."""

    cost_grid = generate_cost_grid(
        commission_bps_levels=(0, 2, 5, 8),
        slippage_bps_levels=(0, 5, 10, 15),
        stamp_duty_bps_levels=(0, 5, 10),
        borrow_bps_levels=(0.0, 100.0),
    )

    outputs: list[pd.DataFrame] = []
    for pair_id, phase_shift in (("600036.SS-3968.HK", 0.0), ("601318.SS-2318.HK", 0.9)):
        frame = _synthetic_pair_frame(pair_id, phase_shift=phase_shift, periods=300)
        out = run_pair_cost_sensitivity(
            pair_frame=frame,
            cost_grid=cost_grid,
            pair_id=pair_id,
            entry=1.2,
            exit=0.4,
            z_window=40,
        )
        outputs.append(out.grid_results)
    return pd.concat(outputs, ignore_index=True)


def _synthetic_pair_frame(pair_id: str, *, phase_shift: float, periods: int) -> pd.DataFrame:
    """Build deterministic synthetic pair frame for demonstration."""

    dates = pd.bdate_range("2023-01-02", periods=periods)
    t = np.arange(periods, dtype=float)
    h_cny = 95.0 + 2.5 * np.sin(t / 24.0 + phase_shift) + 0.08 * t
    spread = 0.18 * np.sin(t / 15.0 + phase_shift) + 0.05 * np.cos(t / 31.0)
    a_close = np.exp(spread) * h_cny

    return pd.DataFrame(
        {
            "date": dates,
            "pair_id": pair_id,
            "a_close": a_close,
            "h_cny": h_cny,
        }
    )


def _render_index_html() -> str:
    """Render GitHub Pages index HTML."""

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AH Premium Lab Demo</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    .card {{
      background: #ffffff;
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 16px;
      box-shadow: 0 2px 8px rgba(15,23,42,.08);
    }}
    .thumb {{
      width: 100%;
      max-width: 980px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      margin-top: 8px;
    }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>AH Premium Lab Demo</h1>
  <p>Generated at: {generated_at}</p>
  <div class="card">
    <h2>Online Demo Report</h2>
    <p><a href="reports/cost_sensitivity_demo.html">Open cost sensitivity report</a></p>
    <p>This site is for research demonstration only and is not trading advice.</p>
  </div>
  <div class="card">
    <h2>Dashboard Preview Assets</h2>
    <img class="thumb" src="screenshots/dashboard_overview_demo.svg" alt="Overview demo" />
    <img
      class="thumb"
      src="screenshots/dashboard_overview_filtered_demo.svg"
      alt="Overview filtered demo"
    />
    <img class="thumb" src="screenshots/dashboard_pair_detail_demo.svg" alt="Pair detail demo" />
  </div>
</body>
</html>
"""


def main() -> None:
    """Build docs assets and print generated index path."""

    output = build_pages_docs(Path("docs"))
    print(output.as_posix())


if __name__ == "__main__":
    main()
