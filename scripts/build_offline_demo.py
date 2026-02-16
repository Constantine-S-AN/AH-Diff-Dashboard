"""Build offline demo docs from local cache or fixtures."""

from __future__ import annotations

import argparse
import hashlib
import warnings
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from make_demo_assets import generate_demo_assets

from ah_premium_lab.backtest import (
    generate_cost_grid,
    generate_sensitivity_html_report,
    run_pair_cost_sensitivity,
)
from ah_premium_lab.backtest.costs import CostParams
from ah_premium_lab.data import CacheOnlyPriceProvider

DEFAULT_OUTPUT_DIR = Path("docs")
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_PAIRS_CSV = Path("data/pairs.csv")
DEFAULT_FIXTURE_PAIRS_CSV = Path("tests/fixtures/research_regression/pairs.csv")


def build_offline_demo(
    *,
    output_dir: Path,
    cache_dir: Path,
    pairs_csv: Path,
    fixture_pairs_csv: Path,
    max_pairs: int,
) -> Path:
    """Build docs index/report/screenshots without any network dependency."""

    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    generate_demo_assets(screenshot_dir)
    pair_table = _load_pair_table(pairs_csv)
    fixture_table = _load_pair_table(fixture_pairs_csv)
    sensitivity, source_label, pair_ids = _build_sensitivity(
        cache_dir=cache_dir,
        pair_table=pair_table,
        fixture_table=fixture_table,
        max_pairs=max_pairs,
    )

    report_path = report_dir / "cost_sensitivity_demo.html"
    generate_sensitivity_html_report(
        result_df=sensitivity,
        output_path=report_path,
        title="AH Premium Lab Offline Demo Cost Sensitivity Report",
    )

    index_path = output_dir / "index.html"
    index_path.write_text(
        _render_index_html(
            source_label=source_label,
            pair_ids=pair_ids,
            sensitivity=sensitivity,
        ),
        encoding="utf-8",
    )
    return index_path


def _build_sensitivity(
    *,
    cache_dir: Path,
    pair_table: pd.DataFrame,
    fixture_table: pd.DataFrame,
    max_pairs: int,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Build demo sensitivity frame from cache first, fixtures second."""

    from_cache, cache_pairs = _build_sensitivity_from_cache(
        cache_dir=cache_dir,
        pair_table=pair_table,
        max_pairs=max_pairs,
    )
    if not from_cache.empty and len(cache_pairs) >= max_pairs:
        return from_cache, "cache", cache_pairs

    from_fixtures, fixture_pairs = _build_sensitivity_from_fixtures(
        fixture_table=fixture_table,
        max_pairs=max_pairs,
    )
    if from_cache.empty and not from_fixtures.empty:
        return from_fixtures, "fixtures", fixture_pairs
    if from_cache.empty and from_fixtures.empty:
        raise ValueError("Unable to build offline demo from cache or fixtures")

    if from_fixtures.empty:
        return from_cache, "cache", cache_pairs

    needed = max(0, max_pairs - len(cache_pairs))
    extra_pairs = [pair for pair in fixture_pairs if pair not in set(cache_pairs)][:needed]
    if not extra_pairs:
        return from_cache, "cache", cache_pairs

    fixture_extra = from_fixtures[from_fixtures["pair_id"].isin(extra_pairs)].copy()
    merged = pd.concat([from_cache, fixture_extra], ignore_index=True)
    merged_pairs = cache_pairs + extra_pairs
    return merged, "cache+fixtures", merged_pairs


def _build_sensitivity_from_cache(
    *,
    cache_dir: Path,
    pair_table: pd.DataFrame,
    max_pairs: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Try building report input from cached market data."""

    provider = CacheOnlyPriceProvider(cache_dir=cache_dir)
    cost_grid = _demo_cost_grid()
    outputs: list[pd.DataFrame] = []
    pair_ids: list[str] = []

    for _, row in pair_table.iterrows():
        pair_id = f"{row['a_ticker']}-{row['h_ticker']}"
        share_ratio = float(row["share_ratio"])
        try:
            pair_frame = _load_pair_frame_from_cache(
                provider=provider,
                a_ticker=str(row["a_ticker"]),
                h_ticker=str(row["h_ticker"]),
            )
        except Exception:  # noqa: BLE001
            continue

        if pair_frame.shape[0] < 120:
            continue

        output = run_pair_cost_sensitivity(
            pair_frame=pair_frame,
            cost_grid=cost_grid,
            pair_id=pair_id,
            entry=1.2,
            exit=0.4,
            z_window=60,
            share_ratio=share_ratio,
        )
        outputs.append(output.grid_results)
        pair_ids.append(pair_id)
        if len(outputs) >= max_pairs:
            break

    if not outputs:
        return pd.DataFrame(), []
    return pd.concat(outputs, ignore_index=True), pair_ids


def _build_sensitivity_from_fixtures(
    *,
    fixture_table: pd.DataFrame,
    max_pairs: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Build deterministic synthetic report input based on fixture pair metadata."""

    if fixture_table.empty:
        fixture_table = pd.DataFrame(
            [
                {
                    "name": "Fixture Pair 1",
                    "a_ticker": "600036.SS",
                    "h_ticker": "3968.HK",
                    "share_ratio": 1.0,
                },
                {
                    "name": "Fixture Pair 2",
                    "a_ticker": "601318.SS",
                    "h_ticker": "2318.HK",
                    "share_ratio": 1.0,
                },
            ]
        )

    outputs: list[pd.DataFrame] = []
    pair_ids: list[str] = []
    cost_grid = _demo_cost_grid()

    for _, row in fixture_table.head(max_pairs).iterrows():
        pair_id = f"{row['a_ticker']}-{row['h_ticker']}"
        share_ratio = float(row["share_ratio"])
        frame = _synthetic_fixture_pair_frame(pair_id=pair_id, share_ratio=share_ratio, periods=300)

        output = run_pair_cost_sensitivity(
            pair_frame=frame,
            cost_grid=cost_grid,
            pair_id=pair_id,
            entry=1.2,
            exit=0.4,
            z_window=60,
            share_ratio=share_ratio,
        )
        outputs.append(output.grid_results)
        pair_ids.append(pair_id)

    if not outputs:
        return pd.DataFrame(), []
    return pd.concat(outputs, ignore_index=True), pair_ids


def _load_pair_frame_from_cache(
    *,
    provider: CacheOnlyPriceProvider,
    a_ticker: str,
    h_ticker: str,
) -> pd.DataFrame:
    """Load one pair frame from offline cache and align to common dates."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_series = provider.get_price(a_ticker, "2000-01-01", "2100-01-01").data["adj_close"]
        h_series = provider.get_price(h_ticker, "2000-01-01", "2100-01-01").data["adj_close"]
        fx_series = provider.get_fx("HKDCNY", "2000-01-01", "2100-01-01").data["close"]

    aligned = pd.concat(
        [
            a_series.rename("a_close"),
            h_series.rename("h_close"),
            fx_series.rename("fx_hkdcny"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError(f"No overlapping cache dates for pair {a_ticker}-{h_ticker}")

    out = pd.DataFrame(index=aligned.index)
    out["a_close"] = aligned["a_close"]
    out["h_cny"] = aligned["h_close"] * aligned["fx_hkdcny"]
    out = out.reset_index().rename(columns={"index": "date"})
    return out


def _load_pair_table(path: Path) -> pd.DataFrame:
    """Load pair metadata table with minimal schema normalization."""

    if not path.exists():
        return pd.DataFrame(columns=["name", "a_ticker", "h_ticker", "share_ratio", "notes"])

    frame = pd.read_csv(path)
    required = {"a_ticker", "h_ticker"}
    missing = required - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Pair table missing columns: {missing_text}")

    out = frame.copy()
    if "share_ratio" not in out.columns:
        out["share_ratio"] = 1.0
    out["share_ratio"] = pd.to_numeric(out["share_ratio"], errors="coerce").fillna(1.0)
    if "name" not in out.columns:
        out["name"] = out["a_ticker"].astype(str) + "-" + out["h_ticker"].astype(str)
    if "notes" not in out.columns:
        out["notes"] = ""
    return out


def _synthetic_fixture_pair_frame(
    pair_id: str,
    *,
    share_ratio: float,
    periods: int,
) -> pd.DataFrame:
    """Create deterministic synthetic pair data when cache is unavailable."""

    seed = int.from_bytes(hashlib.sha256(pair_id.encode("utf-8")).digest()[:4], "little")
    phase_shift = float(seed % 360) / 57.2958

    dates = pd.bdate_range("2023-01-02", periods=periods)
    t = np.arange(periods, dtype=float)
    h_cny = 90.0 + 2.0 * np.sin(t / 22.0 + phase_shift) + 0.05 * t
    spread = 0.16 * np.sin(t / 14.0 + phase_shift) + 0.04 * np.cos(t / 29.0)
    a_close = np.exp(spread) * h_cny * share_ratio

    return pd.DataFrame({"date": dates, "a_close": a_close, "h_cny": h_cny})


def _demo_cost_grid() -> list[CostParams]:
    """Return a compact deterministic demo cost grid."""

    return generate_cost_grid(
        commission_bps_levels=(0, 2, 5, 8),
        slippage_bps_levels=(0, 5, 10, 15),
        stamp_duty_bps_levels=(0, 5, 10),
        borrow_bps_levels=(0.0, 100.0),
    )


def _render_index_html(
    *,
    source_label: str,
    pair_ids: list[str],
    sensitivity: pd.DataFrame,
) -> str:
    """Render a polished, offline-friendly docs landing page."""

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_mapping = {
        "cache": "data/cache",
        "fixtures": "tests/fixtures/research_regression",
        "cache+fixtures": "data/cache + tests/fixtures/research_regression",
    }
    source_text = source_mapping.get(source_label, source_label)

    highlights = _build_demo_highlights(sensitivity=sensitivity, pair_ids=pair_ids)
    pair_chips = "".join(
        [f'<span class="chip"><code>{escape(item)}</code></span>' for item in pair_ids]
    )
    if not pair_chips:
        pair_chips = '<span class="chip"><code>N/A</code></span>'

    pair_table = _build_pair_table(sensitivity=sensitivity, pair_ids=pair_ids)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AH-Diff-Dashboard | Offline Showcase</title>
  <style>
    :root {{
      --bg: #f2f6fc;
      --bg-2: #e8f0fa;
      --surface: #ffffff;
      --surface-2: #f8fbff;
      --text: #1d2f46;
      --muted: #5e7391;
      --line: rgba(30, 68, 108, 0.16);
      --accent: #1574d4;
      --accent-2: #15997f;
      --radius: 16px;
      --shadow: 0 18px 42px rgba(28, 52, 84, 0.16);
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      min-height: 100%;
      color: var(--text);
      background:
        radial-gradient(1000px 520px at 8% -10%, rgba(21, 116, 212, 0.17), transparent 62%),
        radial-gradient(900px 460px at 95% 5%, rgba(21, 153, 127, 0.14), transparent 58%),
        linear-gradient(160deg, var(--bg) 0%, var(--bg-2) 60%, #eef4fc 100%);
      font-family:
        "Avenir Next",
        "Segoe UI Variable",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        sans-serif;
    }}
    body {{
      position: relative;
      overflow-x: hidden;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(21, 56, 94, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(21, 56, 94, 0.035) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: radial-gradient(circle at 50% 40%, black 25%, transparent 76%);
      opacity: 0.35;
    }}
    .page {{
      position: relative;
      z-index: 1;
      width: min(1180px, 100% - 40px);
      margin: 28px auto 40px;
    }}
    .panel {{
      background: linear-gradient(
        145deg,
        rgba(255, 255, 255, 0.95),
        rgba(248, 251, 255, 0.94)
      );
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
      gap: 18px;
      margin-bottom: 18px;
    }}
    .hero-copy {{
      padding: 28px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      margin-bottom: 14px;
      border-radius: 999px;
      border: 1px solid rgba(21, 153, 127, 0.35);
      color: #0f6e5b;
      font-size: 0.8rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: rgba(21, 153, 127, 0.08);
    }}
    h1 {{
      margin: 0 0 10px 0;
      line-height: 1.15;
      font-size: clamp(1.8rem, 3.2vw, 2.8rem);
      font-family:
        "Iowan Old Style",
        "Palatino Linotype",
        "Noto Serif CJK SC",
        serif;
    }}
    .lede {{
      margin: 0;
      max-width: 60ch;
      color: var(--muted);
      line-height: 1.6;
      font-size: 1rem;
    }}
    .meta {{
      margin-top: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid rgba(30, 68, 108, 0.22);
      background: rgba(255, 255, 255, 0.72);
      color: #3a4f6a;
      font-size: 0.84rem;
    }}
    code {{
      font-family: "SF Mono", "JetBrains Mono", "Fira Code", monospace;
      font-size: 0.78rem;
      color: #23486f;
    }}
    .actions {{
      margin-top: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid rgba(30, 68, 108, 0.2);
      color: #244465;
      text-decoration: none;
      font-weight: 600;
      transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
    }}
    .btn:hover {{
      transform: translateY(-1px);
      border-color: rgba(30, 68, 108, 0.34);
      background: rgba(21, 116, 212, 0.06);
    }}
    .btn.primary {{
      border-color: rgba(21, 116, 212, 0.48);
      background: linear-gradient(135deg, rgba(21, 116, 212, 0.16), rgba(21, 153, 127, 0.12));
    }}
    .hero-shot {{
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .hero-shot img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(30, 68, 108, 0.2);
      background: #f1f6fd;
    }}
    .hero-shot figcaption {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .kpi {{
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: linear-gradient(155deg, rgba(255, 255, 255, 0.97), rgba(245, 250, 255, 0.95));
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 0.8rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    .kpi .value {{
      margin-top: 6px;
      font-size: 1.22rem;
      font-weight: 700;
      color: #183a61;
    }}
    .sections {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 18px;
    }}
    .section {{
      padding: 20px;
    }}
    h2 {{
      margin: 0 0 12px 0;
      font-size: 1.14rem;
      letter-spacing: 0.01em;
    }}
    .artifact-grid {{
      display: grid;
      gap: 10px;
    }}
    .artifact {{
      display: block;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--line);
      text-decoration: none;
      color: #1c3858;
      background: rgba(255, 255, 255, 0.84);
      transition: border-color 0.2s ease, background 0.2s ease;
    }}
    .artifact:hover {{
      border-color: rgba(21, 153, 127, 0.45);
      background: rgba(242, 249, 255, 0.98);
    }}
    .artifact strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.95rem;
    }}
    .artifact span {{
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.4;
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 520px;
      font-size: 0.88rem;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(30, 68, 108, 0.11);
      white-space: nowrap;
    }}
    th {{
      color: #3f5977;
      font-weight: 600;
      background: rgba(226, 237, 249, 0.78);
    }}
    td {{
      color: #213a59;
    }}
    .muted {{
      color: var(--muted);
      line-height: 1.6;
    }}
    .gallery {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .gallery a {{
      display: block;
      border-radius: 10px;
      border: 1px solid var(--line);
      overflow: hidden;
    }}
    .gallery img {{
      display: block;
      width: 100%;
      background: #f1f6fd;
    }}
    .reveal {{
      opacity: 0;
      transform: translateY(14px);
      animation: reveal 0.58s ease forwards;
    }}
    .delay-1 {{ animation-delay: 0.08s; }}
    .delay-2 {{ animation-delay: 0.16s; }}
    .delay-3 {{ animation-delay: 0.24s; }}
    @keyframes reveal {{
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    @media (max-width: 980px) {{
      .hero,
      .sections {{
        grid-template-columns: 1fr;
      }}
      .kpis {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .page {{
        width: min(1180px, 100% - 24px);
      }}
      .hero-copy {{
        padding: 20px;
      }}
      .kpis {{
        grid-template-columns: 1fr;
      }}
      .actions {{
        flex-direction: column;
      }}
      .btn {{
        width: 100%;
        justify-content: center;
      }}
      .gallery {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <article class="panel hero-copy reveal">
        <div class="eyebrow">Offline Research Showcase</div>
        <h1>AH-Diff-Dashboard Demo</h1>
        <p class="lede">
          A reproducible A/H premium research surface combining diagnostics, execution constraints,
          and cost sensitivity under one audited workflow. This page is fully viewable offline.
        </p>
        <div class="meta">
          <span class="chip">Generated: <code>{generated_at}</code></span>
          <span class="chip">Source: <code>{escape(source_text)}</code></span>
          {pair_chips}
        </div>
        <div class="actions">
          <a class="btn primary" href="reports/cost_sensitivity_demo.html">Open Demo Report</a>
          <a class="btn" href="methodology.md">Methodology</a>
          <a class="btn" href="case_study.md">Case Study</a>
        </div>
      </article>

      <figure class="panel hero-shot reveal delay-1">
        <img src="screenshots/dashboard_showcase_demo.svg" alt="Dashboard showcase preview" />
        <figcaption>
          Preview of the dashboard used for pair screening and drill-down diagnostics.
        </figcaption>
      </figure>
    </section>

    <section class="kpis">
      <article class="kpi reveal delay-1">
        <div class="label">Pairs Covered</div>
        <div class="value">{highlights["pair_count"]}</div>
      </article>
      <article class="kpi reveal delay-1">
        <div class="label">Scenarios</div>
        <div class="value">{highlights["scenario_count"]}</div>
      </article>
      <article class="kpi reveal delay-2">
        <div class="label">Cost Grid Range</div>
        <div class="value">{highlights["cost_range"]}</div>
      </article>
      <article class="kpi reveal delay-2">
        <div class="label">Best Net CAGR</div>
        <div class="value">{highlights["best_net_cagr"]}</div>
      </article>
      <article class="kpi reveal delay-3">
        <div class="label">Worst Max Drawdown</div>
        <div class="value">{highlights["worst_max_dd"]}</div>
      </article>
    </section>

    <section class="sections">
      <article class="panel section reveal delay-2">
        <h2>Evidence Artifacts</h2>
        <p class="muted">
          The report and supporting docs below are generated from local cache or fixtures, so
          reviewers can verify results even without internet access.
        </p>
        <div class="artifact-grid">
          <a class="artifact" href="reports/cost_sensitivity_demo.html">
            <strong>Cost Sensitivity Report</strong>
            <span>Stress test over commission, slippage, stamp duty, and borrow assumptions.</span>
          </a>
          <a class="artifact" href="case_study.md">
            <strong>Case Study</strong>
            <span>Two A/H pairs with full diagnostics, break analysis, and cost thresholds.</span>
          </a>
          <a class="artifact" href="one_pager.md">
            <strong>One-Pager</strong>
            <span>Interview-ready summary of architecture and key engineering decisions.</span>
          </a>
        </div>
        <div class="gallery">
          <a href="screenshots/dashboard_overview_demo.svg">
            <img src="screenshots/dashboard_overview_demo.svg" alt="Overview demo screenshot" />
          </a>
          <a href="screenshots/dashboard_pair_detail_demo.svg">
            <img
              src="screenshots/dashboard_pair_detail_demo.svg"
              alt="Pair detail demo screenshot"
            />
          </a>
        </div>
      </article>

      <article class="panel section reveal delay-3">
        <h2>Pair-Level Snapshot</h2>
        <p class="muted">
          Summary statistics extracted from current demo scenarios. Values are for research
          evaluation only and do not represent trading advice.
        </p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Pair</th>
                <th>Scenarios</th>
                <th>Best Sharpe</th>
                <th>Best Net CAGR</th>
                <th>Worst Max DD</th>
                <th>Median Cost</th>
              </tr>
            </thead>
            <tbody>
              {pair_table}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  </main>
</body>
</html>
"""


def _build_demo_highlights(
    *,
    sensitivity: pd.DataFrame,
    pair_ids: list[str],
) -> dict[str, str]:
    """Build formatted KPI strings for the landing page."""

    pair_count = (
        len(pair_ids) if pair_ids else int(sensitivity.get("pair_id", pd.Series()).nunique())
    )
    scenario_count = int(sensitivity.shape[0])

    cost_series = _finite_series(sensitivity, "total_cost_level_bps")
    cagr_series = _finite_series(sensitivity, "net_cagr")
    dd_series = _finite_series(sensitivity, "max_dd")

    if cost_series.empty:
        cost_range = "N/A"
    else:
        cost_range = f"{cost_series.min():.1f}-{cost_series.max():.1f} bps"

    best_cagr = _format_pct(cagr_series.max(), signed=True) if not cagr_series.empty else "N/A"
    worst_dd = _format_pct(dd_series.min(), signed=True) if not dd_series.empty else "N/A"

    return {
        "pair_count": str(pair_count),
        "scenario_count": f"{scenario_count:,}",
        "cost_range": cost_range,
        "best_net_cagr": best_cagr,
        "worst_max_dd": worst_dd,
    }


def _build_pair_table(*, sensitivity: pd.DataFrame, pair_ids: list[str]) -> str:
    """Build HTML rows for pair-level summary table."""

    if sensitivity.empty or "pair_id" not in sensitivity.columns:
        if not pair_ids:
            return "<tr><td colspan='6'>N/A</td></tr>"
        return "".join(
            [
                (
                    "<tr>"
                    f"<td><code>{escape(pair_id)}</code></td>"
                    "<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>"
                    "</tr>"
                )
                for pair_id in pair_ids
            ]
        )

    frame = sensitivity.copy()
    for col in ["net_sharpe", "net_cagr", "max_dd", "total_cost_level_bps"]:
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")

    grouped = (
        frame.groupby("pair_id", dropna=True)
        .agg(
            scenarios=("pair_id", "size"),
            best_sharpe=("net_sharpe", "max"),
            best_cagr=("net_cagr", "max"),
            worst_dd=("max_dd", "min"),
            median_cost=("total_cost_level_bps", "median"),
        )
        .reset_index()
        .sort_values("pair_id")
    )

    rows: list[str] = []
    for row in grouped.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(row.pair_id))}</code></td>"
            f"<td>{int(row.scenarios)}</td>"
            f"<td>{_format_number(float(row.best_sharpe), 3)}</td>"
            f"<td>{_format_pct(float(row.best_cagr), signed=True)}</td>"
            f"<td>{_format_pct(float(row.worst_dd), signed=True)}</td>"
            f"<td>{_format_number(float(row.median_cost), 1)} bps</td>"
            "</tr>"
        )

    return "".join(rows) if rows else "<tr><td colspan='6'>N/A</td></tr>"


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values of one column."""

    if column not in frame.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(frame[column], errors="coerce")
    return series[np.isfinite(series)]


def _format_number(value: float, digits: int) -> str:
    """Format finite float with fixed digits."""

    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _format_pct(value: float, *, signed: bool = False) -> str:
    """Format decimal return/drawdown as percentage string."""

    if not np.isfinite(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value * 100:{sign}.2f}%"


def _parse_args() -> argparse.Namespace:
    """Parse command-line args for offline demo build."""

    parser = argparse.ArgumentParser(description="Build offline docs demo assets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--fixture-pairs-csv", type=Path, default=DEFAULT_FIXTURE_PAIRS_CSV)
    parser.add_argument("--max-pairs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for offline demo build script."""

    args = _parse_args()
    index_path = build_offline_demo(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        pairs_csv=args.pairs_csv,
        fixture_pairs_csv=args.fixture_pairs_csv,
        max_pairs=int(args.max_pairs),
    )
    print(index_path.as_posix())


if __name__ == "__main__":
    main()
