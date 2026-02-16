"""Build offline demo docs from local cache or fixtures."""

from __future__ import annotations

import argparse
import hashlib
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
    if not from_cache.empty:
        return from_cache, "cache", cache_pairs

    from_fixtures, fixture_pairs = _build_sensitivity_from_fixtures(
        fixture_table=fixture_table,
        max_pairs=max_pairs,
    )
    if not from_fixtures.empty:
        return from_fixtures, "fixtures", fixture_pairs

    raise ValueError("Unable to build offline demo from cache or fixtures")


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


def _render_index_html(*, source_label: str, pair_ids: list[str]) -> str:
    """Render offline-friendly docs landing page."""

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_text = "data/cache" if source_label == "cache" else "tests/fixtures/research_regression"
    pair_items = "".join([f"<li><code>{escape(item)}</code></li>" for item in pair_ids])
    if not pair_items:
        pair_items = "<li><code>N/A</code></li>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AH Premium Lab Offline Demo</title>
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
    code {{
      background: #f1f5f9;
      border-radius: 4px;
      padding: 1px 4px;
    }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>AH Premium Lab Offline Demo</h1>
  <p>Generated at: {generated_at}</p>
  <div class="card">
    <h2>Offline Demo Mode</h2>
    <p>This page works without internet. Open it directly from local files.</p>
    <p>Data source used for report build: <code>{escape(source_text)}</code></p>
    <p><a href="reports/cost_sensitivity_demo.html">Open cost sensitivity report</a></p>
  </div>
  <div class="card">
    <h2>Pairs Included</h2>
    <ul>{pair_items}</ul>
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
