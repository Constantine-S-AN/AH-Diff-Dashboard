# ruff: noqa: E501
"""Generate AH premium research report with statistics and cost sensitivity."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ah_premium_lab.backtest import CostParams, generate_cost_grid, run_pair_cost_sensitivity
from ah_premium_lab.core import compute_premium_metrics
from ah_premium_lab.data import (
    YahooFinanceProvider,
    check_fx_integrity,
    check_price_integrity,
    load_ah_pairs,
)
from ah_premium_lab.stats import (
    adf_test,
    detect_structural_breaks,
    engle_granger_test,
    half_life_ar1,
    rolling_engle_granger,
    rolling_stability_metrics,
    summary_score,
)

OutputMode = Literal["single", "multi"]


REFERENCE_LINKS: list[tuple[str, str]] = [
    (
        "Hang Seng Indexes: AH 指数系列信息迁移说明",
        "https://www.hsi.com.hk/redirect.html",
    ),
    (
        "HKEX: Shanghai Connect 报价限制（含 Turnaround Trading 不允许）",
        "https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Reference-Materials/Rules-and-Regulations/Quotation-Requirements-and-Restrictions-for-Shanghai-Connect-Orders?sc_lang=en",
    ),
    (
        "HKEX: China Connect 清算结算（股票 T 日、资金 T+1）",
        "https://www.hkex.com.hk/Services/Clearing/Securities/Overview/Clearing-Services?sc_lang=en",
    ),
    (
        "HKEX: Regulated Short Selling 规则",
        "https://www.hkex.com.hk/services/trading/securities/overview/regulated-short-selling?sc_lang=en",
    ),
    (
        "HKEX: Stock Connect 交易与结算日历",
        "https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Reference-Materials/Trading-Hour%2C-Trading-and-Settlement-Calendar?sc_lang=en",
    ),
    (
        "Journal of Empirical Finance (2025): The AH premium",
        "https://www.sciencedirect.com/science/article/pii/S0927539825000210",
    ),
]


@dataclass(frozen=True)
class CostGridConfig:
    """Cost grid range configuration in basis points."""

    commission_min_bps: float
    commission_max_bps: float
    commission_step_bps: float
    slippage_min_bps: float
    slippage_max_bps: float
    slippage_step_bps: float
    stamp_min_bps: float
    stamp_max_bps: float
    stamp_step_bps: float


@dataclass(frozen=True)
class PairReportResult:
    """Per-pair computed report payload."""

    pair_id: str
    name: str
    a_ticker: str
    h_ticker: str
    notes: str
    sample_start: str
    sample_end: str
    aligned_sample_size: int
    latest_premium_pct: float
    latest_rolling_z: float
    latest_premium_percentile: float
    half_life_days: float
    adf_p_value: float
    adf_stat: float
    adf_used_lags: int
    eg_p_value: float
    eg_beta: float
    summary_score: float
    a_missing_days: int
    h_missing_days: int
    fx_missing_days: int
    a_missing_rate: float
    h_missing_rate: float
    fx_missing_rate: float
    integrity_warnings: tuple[str, ...]
    aligned_frame: pd.DataFrame
    sensitivity_df: pd.DataFrame
    breakeven_cost_level: float
    rolling_p_value_pass_rate: float = float("nan")
    rolling_beta_variance: float = float("nan")
    rolling_resid_std_drift: float = float("nan")
    rolling_coint_stability_score: float = float("nan")
    breaks_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    break_count: int = 0
    max_break_confidence: float = 0.0
    cusum_stat: float = float("nan")
    cusum_p_value: float = float("nan")


@dataclass(frozen=True)
class ReportMetadata:
    """Metadata captured at report generation time."""

    generated_at_utc: str
    data_fetch_time_utc: str
    git_commit: str
    start_date: str
    end_date: str
    expected_business_days: int
    pair_count: int
    total_aligned_samples: int
    mean_aligned_samples: float
    avg_a_missing_rate: float
    avg_h_missing_rate: float
    avg_fx_missing_rate: float
    fx_pair: str
    fx_alignment: str
    cost_grid_size: int
    cost_parameters: dict[str, float]


def generate_report(
    *,
    start_date: str,
    end_date: str,
    pairs: list[str] | None,
    cost_grid_config: CostGridConfig,
    output_path: str | Path,
    output_mode: OutputMode = "single",
    pairs_csv: str | Path = "data/pairs.csv",
    cache_dir: str | Path = "data/cache",
    fx_pair: Literal["HKDCNY", "CNYHKD"] = "HKDCNY",
    window: int = 252,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> Path:
    """Generate report as a single HTML file or `index + pair pages`.

    Args:
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).
        pairs: Optional pair-id filter list (`A_TICKER-H_TICKER`).
        cost_grid_config: Cost range settings.
        output_path: Output file (single) or directory (multi).
        output_mode: `single` or `multi`.
        pairs_csv: Path to pair universe CSV.
        cache_dir: Data cache directory.
        fx_pair: FX quote direction passed to data/core modules.
        window: Rolling window for premium indicators.
        entry: Strategy entry z threshold.
        exit_: Strategy exit z threshold.

    Returns:
        Output file path (`single`) or `index.html` path (`multi`).
    """

    data_fetch_time_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    pair_reports, cost_grid_size = _collect_pair_reports(
        start_date=start_date,
        end_date=end_date,
        pairs=pairs,
        cost_grid_config=cost_grid_config,
        pairs_csv=pairs_csv,
        cache_dir=cache_dir,
        fx_pair=fx_pair,
        window=window,
        entry=entry,
        exit_=exit_,
    )

    if not pair_reports:
        raise ValueError("No pair results produced. Check pairs filter and data availability.")

    metadata = _build_report_metadata(
        pair_reports,
        start_date=start_date,
        end_date=end_date,
        fx_pair=fx_pair,
        cost_grid_config=cost_grid_config,
        cost_grid_size=cost_grid_size,
        data_fetch_time_utc=data_fetch_time_utc,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output_mode == "single":
        html = _render_single_report_html(
            pair_reports,
            metadata=metadata,
            start_date=start_date,
            end_date=end_date,
        )
        output.write_text(html, encoding="utf-8")
        return output

    if output_mode == "multi":
        output.mkdir(parents=True, exist_ok=True)
        pair_pages_dir = output / "pairs"
        pair_pages_dir.mkdir(parents=True, exist_ok=True)

        pair_links: list[tuple[str, str]] = []
        for pair in pair_reports:
            pair_file_name = f"{_slug(pair.pair_id)}.html"
            pair_file_path = pair_pages_dir / pair_file_name
            pair_html = _render_pair_page_html(pair, metadata=metadata)
            pair_file_path.write_text(pair_html, encoding="utf-8")
            pair_links.append((pair.pair_id, f"pairs/{pair_file_name}"))

        index_html = _render_index_html(
            pair_reports,
            pair_links,
            metadata=metadata,
            start_date=start_date,
            end_date=end_date,
        )
        index_path = output / "index.html"
        index_path.write_text(index_html, encoding="utf-8")
        return index_path

    raise ValueError(f"Unsupported output mode: {output_mode}")


def _collect_pair_reports(
    *,
    start_date: str,
    end_date: str,
    pairs: list[str] | None,
    cost_grid_config: CostGridConfig,
    pairs_csv: str | Path,
    cache_dir: str | Path,
    fx_pair: Literal["HKDCNY", "CNYHKD"],
    window: int,
    entry: float,
    exit_: float,
) -> tuple[list[PairReportResult], int]:
    """Collect per-pair report artifacts."""

    pair_universe = load_ah_pairs(pairs_csv)
    selected_ids = {_pair_id(p.a_ticker, p.h_ticker) for p in pair_universe}
    if pairs:
        requested = {item.strip() for item in pairs if item.strip()}
        selected_ids = selected_ids.intersection(requested)

    provider = YahooFinanceProvider(cache_dir=cache_dir)
    cost_grid = _build_tied_cost_grid(cost_grid_config)

    outputs: list[PairReportResult] = []
    for pair in pair_universe:
        pid = _pair_id(pair.a_ticker, pair.h_ticker)
        if pid not in selected_ids:
            continue

        a_series = provider.get_price(pair.a_ticker, start_date, end_date)
        h_series = provider.get_price(pair.h_ticker, start_date, end_date)
        fx_series = provider.get_fx(fx_pair, start_date, end_date)

        a_integrity = check_price_integrity(a_series)
        h_integrity = check_price_integrity(h_series)
        fx_integrity = check_fx_integrity(fx_series)

        aligned = compute_premium_metrics(
            a_price_cny=a_series.data["adj_close"],
            h_price_hkd=h_series.data["adj_close"],
            fx=fx_series.data["close"],
            fx_quote=fx_pair,
            share_ratio=pair.share_ratio,
            window=window,
        )

        adf_p, adf_stat, adf_lags = adf_test(aligned["log_spread"])
        eg_p, eg_beta, resid = engle_granger_test(
            log_A=np.log(aligned["a_price_cny"]),
            log_H_fx=np.log(aligned["h_price_cny_equiv"]),
        )
        _ = resid

        half_life = half_life_ar1(aligned["log_spread"])
        score = summary_score(
            adf_p_value=adf_p,
            eg_p_value=eg_p,
            half_life_days=half_life,
            zscore_series=aligned["rolling_zscore"],
        )
        rolling_coint = rolling_engle_granger(
            logA=np.log(aligned["a_price_cny"]),
            logHfx=np.log(aligned["h_price_cny_equiv"]),
            window=window,
            step=21,
        )
        coint_stability = rolling_stability_metrics(rolling_coint)
        breaks = detect_structural_breaks(
            aligned["log_spread"],
            window=max(20, window // 4),
            min_distance=max(5, window // 12),
        )
        breaks_df = breaks.breakpoints.copy()
        if not breaks_df.empty:
            breaks_df["break_date"] = pd.to_datetime(breaks_df["break_date"])

        aligned_reset = aligned.reset_index()
        first_column = str(aligned_reset.columns[0])
        aligned_reset = aligned_reset.rename(columns={first_column: "date"})

        strategy_frame = aligned_reset[["date", "a_price_cny", "h_price_cny_equiv"]].rename(
            columns={"a_price_cny": "a_close", "h_price_cny_equiv": "h_cny"}
        )
        sensitivity = run_pair_cost_sensitivity(
            pair_frame=strategy_frame,
            cost_grid=cost_grid,
            pair_id=pid,
            entry=entry,
            exit=exit_,
            z_window=window,
        )

        warnings_text: list[str] = []
        warnings_text.extend(a_integrity.warnings)
        warnings_text.extend(h_integrity.warnings)
        warnings_text.extend(fx_integrity.warnings)

        a_missing_rate = _missing_rate(a_integrity.missing_days, int(a_series.data.shape[0]))
        h_missing_rate = _missing_rate(h_integrity.missing_days, int(h_series.data.shape[0]))
        fx_missing_rate = _missing_rate(fx_integrity.missing_days, int(fx_series.data.shape[0]))

        latest = aligned.iloc[-1]
        outputs.append(
            PairReportResult(
                pair_id=pid,
                name=pair.name,
                a_ticker=pair.a_ticker,
                h_ticker=pair.h_ticker,
                notes=pair.notes,
                sample_start=str(aligned.index.min().date()),
                sample_end=str(aligned.index.max().date()),
                aligned_sample_size=int(aligned.shape[0]),
                latest_premium_pct=float(latest["premium_pct"] * 100.0),
                latest_rolling_z=float(latest["rolling_zscore"]),
                latest_premium_percentile=float(latest["rolling_percentile"]),
                half_life_days=float(half_life),
                adf_p_value=float(adf_p),
                adf_stat=float(adf_stat),
                adf_used_lags=int(adf_lags),
                eg_p_value=float(eg_p),
                eg_beta=float(eg_beta),
                summary_score=float(score),
                a_missing_days=int(a_integrity.missing_days),
                h_missing_days=int(h_integrity.missing_days),
                fx_missing_days=int(fx_integrity.missing_days),
                a_missing_rate=a_missing_rate,
                h_missing_rate=h_missing_rate,
                fx_missing_rate=fx_missing_rate,
                integrity_warnings=tuple(warnings_text),
                aligned_frame=aligned_reset,
                sensitivity_df=sensitivity.grid_results,
                breakeven_cost_level=float(sensitivity.breakeven_cost_level),
                rolling_p_value_pass_rate=float(coint_stability.p_value_pass_rate),
                rolling_beta_variance=float(coint_stability.beta_variance),
                rolling_resid_std_drift=float(coint_stability.resid_std_drift),
                rolling_coint_stability_score=float(coint_stability.stability_score),
                breaks_df=breaks_df,
                break_count=int(breaks_df.shape[0]),
                max_break_confidence=(
                    float(breaks_df["confidence"].max()) if not breaks_df.empty else 0.0
                ),
                cusum_stat=float(breaks.cusum_stat),
                cusum_p_value=float(breaks.cusum_p_value),
            )
        )

    return outputs, len(cost_grid)


def _build_tied_cost_grid(config: CostGridConfig) -> list[CostParams]:
    """Build cost grid with symmetric A/H commission and stamp levels."""

    commission_levels = _build_levels(
        config.commission_min_bps,
        config.commission_max_bps,
        config.commission_step_bps,
    )
    slippage_levels = _build_levels(
        config.slippage_min_bps,
        config.slippage_max_bps,
        config.slippage_step_bps,
    )
    stamp_levels = _build_levels(
        config.stamp_min_bps,
        config.stamp_max_bps,
        config.stamp_step_bps,
    )

    raw_grid = generate_cost_grid(
        commission_bps_levels=commission_levels,
        slippage_bps_levels=slippage_levels,
        stamp_duty_bps_levels=stamp_levels,
        commission_h_bps_levels=commission_levels,
        stamp_duty_h_bps_levels=stamp_levels,
    )

    tied: list[CostParams] = []
    for item in raw_grid:
        if (
            item.commission_a_bps == item.commission_h_bps
            and item.stamp_duty_a_bps == item.stamp_duty_h_bps
        ):
            tied.append(item)
    return tied


def _build_levels(min_value: float, max_value: float, step: float) -> tuple[float, ...]:
    """Build inclusive float level tuple from min/max/step."""

    if step <= 0.0:
        raise ValueError("step must be > 0")
    if max_value < min_value:
        raise ValueError("max must be >= min")

    values: list[float] = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(round(current, 8))
        current += step
    return tuple(values)


def _build_report_metadata(
    pair_reports: list[PairReportResult],
    *,
    start_date: str,
    end_date: str,
    fx_pair: str,
    cost_grid_config: CostGridConfig,
    cost_grid_size: int,
    data_fetch_time_utc: str,
) -> ReportMetadata:
    """Build report-level metadata summary."""

    expected_business_days = int(len(pd.bdate_range(start=start_date, end=end_date)))
    total_aligned_samples = int(sum(item.aligned_sample_size for item in pair_reports))
    mean_aligned_samples = (
        float(total_aligned_samples / len(pair_reports)) if pair_reports else float("nan")
    )

    avg_a_missing_rate = _mean_or_nan([item.a_missing_rate for item in pair_reports])
    avg_h_missing_rate = _mean_or_nan([item.h_missing_rate for item in pair_reports])
    avg_fx_missing_rate = _mean_or_nan([item.fx_missing_rate for item in pair_reports])

    return ReportMetadata(
        generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        data_fetch_time_utc=data_fetch_time_utc,
        git_commit=_resolve_git_commit(),
        start_date=start_date,
        end_date=end_date,
        expected_business_days=expected_business_days,
        pair_count=int(len(pair_reports)),
        total_aligned_samples=total_aligned_samples,
        mean_aligned_samples=mean_aligned_samples,
        avg_a_missing_rate=avg_a_missing_rate,
        avg_h_missing_rate=avg_h_missing_rate,
        avg_fx_missing_rate=avg_fx_missing_rate,
        fx_pair=fx_pair,
        fx_alignment=_fx_alignment_text(fx_pair),
        cost_grid_size=cost_grid_size,
        cost_parameters={
            "commission_min_bps": cost_grid_config.commission_min_bps,
            "commission_max_bps": cost_grid_config.commission_max_bps,
            "commission_step_bps": cost_grid_config.commission_step_bps,
            "slippage_min_bps": cost_grid_config.slippage_min_bps,
            "slippage_max_bps": cost_grid_config.slippage_max_bps,
            "slippage_step_bps": cost_grid_config.slippage_step_bps,
            "stamp_min_bps": cost_grid_config.stamp_min_bps,
            "stamp_max_bps": cost_grid_config.stamp_max_bps,
            "stamp_step_bps": cost_grid_config.stamp_step_bps,
        },
    )


def _resolve_git_commit() -> str:
    """Resolve current git commit hash when available."""

    repo_root: Path | None = None
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is None:
        return "unknown"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"

    commit = result.stdout.strip()
    if result.returncode == 0 and commit:
        return commit
    return "unknown"


def _fx_alignment_text(fx_pair: str) -> str:
    """Describe FX conversion route into HKD->CNY."""

    normalized = fx_pair.upper().replace("/", "")
    if normalized == "HKDCNY":
        return "direct: HKD->CNY = FX"
    if normalized == "CNYHKD":
        return "inverted: HKD->CNY = 1 / FX"
    return "unknown"


def _missing_rate(missing_days: int, observed_days: int) -> float:
    """Compute missing-day rate from missing and observed counts."""

    denominator = missing_days + observed_days
    if denominator <= 0:
        return float("nan")
    return float(missing_days / denominator)


def _mean_or_nan(values: list[float]) -> float:
    """Return finite mean or NaN when no finite values are available."""

    finite = [item for item in values if np.isfinite(item)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _render_single_report_html(
    pair_reports: list[PairReportResult],
    *,
    metadata: ReportMetadata,
    start_date: str,
    end_date: str,
) -> str:
    """Render complete single-file HTML report."""

    summary_df = _summary_dataframe(pair_reports)
    top_df = summary_df.sort_values("latest_premium_pct", ascending=False).head(5)
    bottom_df = summary_df.sort_values("latest_premium_pct", ascending=True).head(5)
    strongest = summary_df.sort_values("summary_score", ascending=False).head(3)
    weakest = summary_df.sort_values("summary_score", ascending=True).head(3)
    coint_stability_rank = summary_df.dropna(subset=["rolling_coint_stability_score"]).copy()
    coint_top10 = coint_stability_rank.sort_values("rolling_coint_stability_score", ascending=False).head(
        10
    )
    coint_bottom10 = coint_stability_rank.sort_values("rolling_coint_stability_score", ascending=True).head(
        10
    )
    break_rank = summary_df.sort_values(
        ["break_count", "max_break_confidence"],
        ascending=[False, False],
    ).head(10)

    pair_sections: list[str] = []
    include_plotlyjs = True
    for pair in pair_reports:
        section_html, include_plotlyjs = _render_pair_section(pair, include_plotlyjs=include_plotlyjs)
        pair_sections.append(section_html)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AH Premium Lab Research Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 26px; color: #0f172a; background: #f8fafc; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .meta {{ color: #334155; margin-bottom: 18px; }}
    .notice {{ background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 12px; margin-bottom: 20px; }}
    .section {{ background: #ffffff; border-radius: 10px; padding: 14px; margin: 14px 0; box-shadow: 0 2px 8px rgba(15,23,42,.08); }}
    .pair {{ margin-top: 24px; padding-top: 12px; border-top: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    ul {{ margin-top: 6px; }}
    a {{ color: #0f62fe; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>AH Premium Lab Research Report</h1>
  <p class="meta">Generated at: {metadata.generated_at_utc} | Sample window: {start_date} ~ {end_date}</p>
  <div class="notice"><strong>Research Use Only:</strong> 本报告用于研究敏感性分析，不构成任何交易建议或执行建议。</div>

  <section class="section">
    <h2>Metadata</h2>
    {_df_to_html(_metadata_table(metadata))}
    <h3>Per-Pair Sample & Missing Rate</h3>
    {_df_to_html(_pair_quality_table(pair_reports))}
  </section>

  <section class="section">
    <h2>1) 背景：AH 溢价概念与指数定义</h2>
    <p>
      AH 溢价用于比较同一发行人 A/H 两地股票的估值差异。常见口径是比较 A 股价格相对 H 股（经汇率与股本比例调整后的）溢价/折价。
      市场上常用恒生沪深港通 AH 指数系列中的溢价指标作为跟踪工具（一般地，指数高于 100 表示 A 相对 H 为溢价）。
    </p>
    <ul>
      {_reference_links_html()}
    </ul>
  </section>

  <section class="section">
    <h2>2) 方法</h2>
    <p><strong>premium 口径：</strong> premium_pct = A / (H × fx_hkd_to_cny × share_ratio) - 1；log_spread = log(A) - log(H × fx_hkd_to_cny × share_ratio)。</p>
    <p><strong>检验方法：</strong> ADF、Engle–Granger（残差 ADF）、AR(1) 半衰期，以及综合研究评分 summary score（0-100）。</p>
    <p><strong>结构变化检测：</strong> 对 log_spread 进行 rolling mean-shift 检测（Welch t-test + 均值位移 z-score），并用 CUSUM 作为全局稳定性补充。输出断点日期与置信度（0-1）。</p>
    <h3>结构变化为何重要</h3>
    <p>若价差生成机制发生切换（监管、流动性、估值锚迁移），历史均值回归参数会失效。结构突变用于提示“旧参数是否仍可用”，帮助避免把非平稳阶段误判成可交易的均值回归。</p>
    <p><strong>回测假设：</strong> 日频 close 执行；信号由 rolling zscore(log_spread) 触发；仅做研究级收益与成本敏感性，不含撮合细节。</p>
    <p><strong>成本模型：</strong> commission/stamp/slippage 按 A/H 分腿参数化；按每日换手线性计提，输出 net CAGR、net Sharpe、max DD 与 breakeven 成本水平。</p>
  </section>

  <section class="section">
    <h2>3) 结果</h2>
    <h3>Top Premium（latest）</h3>
    {_df_to_html(top_df)}
    <h3>Bottom Premium（latest）</h3>
    {_df_to_html(bottom_df)}
    <h3>均值回归最强（summary score Top 3）</h3>
    {_df_to_html(strongest)}
    <h3>均值回归最弱（summary score Bottom 3）</h3>
    {_df_to_html(weakest)}
    <h3>协整稳定性排名 Top 10</h3>
    {_df_to_html(coint_top10[["pair_id", "rolling_p_value_pass_rate", "rolling_beta_variance", "rolling_resid_std_drift", "rolling_coint_stability_score"]])}
    <h3>协整稳定性排名 Bottom 10</h3>
    {_df_to_html(coint_bottom10[["pair_id", "rolling_p_value_pass_rate", "rolling_beta_variance", "rolling_resid_std_drift", "rolling_coint_stability_score"]])}
    <h3>结构突变活跃度 Top 10</h3>
    {_df_to_html(break_rank[["pair_id", "break_count", "max_break_confidence", "cusum_p_value"]])}
    <h3>成本敏感性总结（每对 AH）</h3>
    {_df_to_html(summary_df[["pair_id", "summary_score", "latest_premium_pct", "half_life_days", "breakeven_cost_level"]])}
    <p>下方为每对 AH 的详细图表与成本容忍度热力图/表格。</p>
    {''.join(pair_sections)}
  </section>

  <section class="section">
    <h2>4) 限制（研究版）</h2>
    <ul>
      <li>T+1 / Turnaround trading 约束会影响真实套利可执行性与资金占用。</li>
      <li>做空存在标的、覆盖与报价限制（例如指定证券、tick rule、规则约束）。</li>
      <li>互联互通存在交易日历、结算窗口、预交易检查等制度性约束，可能导致研究与实务偏离。</li>
      <li>统计显著性不等于未来收益确定性；本报告仅用于研究，不构成投资建议。</li>
    </ul>
  </section>
</body>
</html>
"""


def _render_index_html(
    pair_reports: list[PairReportResult],
    pair_links: list[tuple[str, str]],
    *,
    metadata: ReportMetadata,
    start_date: str,
    end_date: str,
) -> str:
    """Render multi-page index HTML."""

    summary_df = _summary_dataframe(pair_reports)
    links_html = "".join([f'<li><a href="{href}">{pid}</a></li>' for pid, href in pair_links])

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AH Premium Lab Report Index</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 26px; color: #0f172a; background: #f8fafc; }}
    .section {{ background: #ffffff; border-radius: 10px; padding: 14px; margin: 14px 0; box-shadow: 0 2px 8px rgba(15,23,42,.08); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    a {{ color: #0f62fe; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>AH Premium Lab Report Index</h1>
  <p>Generated at: {metadata.generated_at_utc} | Sample: {start_date} ~ {end_date}</p>
  <div class="section">
    <h2>Research Notice</h2>
    <p>本索引与子页报告用于研究敏感性分析，不构成任何交易建议。</p>
  </div>
  <div class="section">
    <h2>Metadata</h2>
    {_df_to_html(_metadata_table(metadata))}
  </div>
  <div class="section">
    <h2>Summary</h2>
    {_df_to_html(summary_df)}
  </div>
  <div class="section">
    <h2>Pair Pages</h2>
    <ul>{links_html}</ul>
  </div>
</body>
</html>
"""


def _render_pair_page_html(pair: PairReportResult, *, metadata: ReportMetadata) -> str:
    """Render per-pair standalone HTML page (for multi-page mode)."""

    section_html, _ = _render_pair_section(pair, include_plotlyjs=True)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{pair.pair_id} Pair Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 26px; color: #0f172a; background: #f8fafc; }}
    .section {{ background: #ffffff; border-radius: 10px; padding: 14px; margin: 14px 0; box-shadow: 0 2px 8px rgba(15,23,42,.08); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>{pair.pair_id}</h1>
  <div class="section">
    <p><strong>Research Use Only:</strong> 本页仅用于研究敏感性分析，不构成交易建议。</p>
    {_df_to_html(_pair_page_metadata_table(pair, metadata))}
  </div>
  {section_html}
</body>
</html>
"""


def _render_pair_section(
    pair: PairReportResult,
    *,
    include_plotlyjs: bool,
) -> tuple[str, bool]:
    """Render per-pair detailed section with plots and sensitivity table."""

    aligned = pair.aligned_frame
    breaks = pair.breaks_df.copy()
    if not breaks.empty:
        breaks["break_date"] = pd.to_datetime(breaks["break_date"])

    premium_fig = px.line(aligned, x="date", y="premium_pct", title=f"{pair.pair_id} premium_pct")
    _add_break_lines(premium_fig, breaks)
    premium_html = premium_fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
    )
    include_plotlyjs = False

    spread_fig = px.line(aligned, x="date", y="log_spread", title=f"{pair.pair_id} log_spread")
    _add_break_lines(spread_fig, breaks)
    spread_html = spread_fig.to_html(full_html=False, include_plotlyjs=False)

    z_fig = px.line(aligned, x="date", y="rolling_zscore", title=f"{pair.pair_id} rolling_zscore")
    _add_break_lines(z_fig, breaks)
    z_html = z_fig.to_html(full_html=False, include_plotlyjs=False)

    heat_pivot = (
        pair.sensitivity_df.groupby(["slippage_bps", "commission_a_bps"], as_index=True)["net_cagr"]
        .mean()
        .unstack("commission_a_bps")
        .sort_index()
    )
    heat_fig = px.imshow(
        heat_pivot,
        labels={"x": "Commission A (bps)", "y": "Slippage (bps)", "color": "Avg Net CAGR"},
        title=f"{pair.pair_id} cost tolerance heatmap",
        aspect="auto",
        color_continuous_scale="RdYlGn",
    )
    heat_html = heat_fig.to_html(full_html=False, include_plotlyjs=False)
    tolerance_df = _build_cost_tolerance_table(pair.sensitivity_df)
    radar_html = _build_cost_tolerance_radar(pair.pair_id, tolerance_df)
    tolerance_table_html = _df_to_html(tolerance_df)

    warnings_html = "".join([f"<li>{w}</li>" for w in pair.integrity_warnings])
    warnings_block = (
        f"<ul>{warnings_html}</ul>" if pair.integrity_warnings else "<p>No data-integrity warning detected.</p>"
    )

    metrics_df = pd.DataFrame(
        [
            {
                "pair_id": pair.pair_id,
                "latest_premium_pct": pair.latest_premium_pct,
                "latest_rolling_z": pair.latest_rolling_z,
                "latest_premium_percentile": pair.latest_premium_percentile,
                "half_life_days": pair.half_life_days,
                "adf_p_value": pair.adf_p_value,
                "eg_p_value": pair.eg_p_value,
                "eg_beta": pair.eg_beta,
                "summary_score": pair.summary_score,
                "rolling_p_value_pass_rate": pair.rolling_p_value_pass_rate,
                "rolling_beta_variance": pair.rolling_beta_variance,
                "rolling_resid_std_drift": pair.rolling_resid_std_drift,
                "rolling_coint_stability_score": pair.rolling_coint_stability_score,
                "break_count": pair.break_count,
                "max_break_confidence": pair.max_break_confidence,
                "cusum_p_value": pair.cusum_p_value,
                "breakeven_cost_level": pair.breakeven_cost_level,
                "breakeven_total_cost": _safe_first_value(
                    pair.sensitivity_df,
                    "breakeven_total_cost",
                ),
                "breakeven_slippage": _safe_first_value(
                    pair.sensitivity_df,
                    "breakeven_slippage",
                ),
                "worst_case_net_dd": _safe_first_value(
                    pair.sensitivity_df,
                    "worst_case_net_dd",
                ),
                "sample_start": pair.sample_start,
                "sample_end": pair.sample_end,
                "aligned_sample_size": pair.aligned_sample_size,
            }
        ]
    )

    sensitivity_table_cols = [
        "commission_a_bps",
        "commission_h_bps",
        "stamp_duty_a_bps",
        "stamp_duty_h_bps",
        "slippage_bps",
        "borrow_bps",
        "total_cost_level_bps",
        "net_cagr",
        "net_sharpe",
        "max_dd",
        "breakeven_cost_level",
        "breakeven_total_cost",
        "breakeven_slippage",
        "worst_case_net_dd",
    ]
    sensitivity_table_cols = [col for col in sensitivity_table_cols if col in pair.sensitivity_df.columns]
    break_table_html = (
        _df_to_html(breaks[["break_date", "confidence", "p_value", "shift_zscore", "mean_shift"]])
        if not breaks.empty
        else "<p>No significant structural break detected in this sample.</p>"
    )

    section_html = f"""
    <div class="pair">
      <h3>{pair.pair_id} | {pair.name}</h3>
      <p>{pair.notes}</p>
      <h4>Pair Metrics</h4>
      {_df_to_html(metrics_df)}
      <h4>Integrity Warnings</h4>
      {warnings_block}
      <h4>Structural Breakpoints</h4>
      {break_table_html}
      <h4>Premium / Spread / Zscore</h4>
      {premium_html}
      {spread_html}
      {z_html}
      <h4>Cost Sensitivity</h4>
      <h5>Cost Tolerance Radar</h5>
      {radar_html}
      <h5>Cost Tolerance Table</h5>
      {tolerance_table_html}
      {heat_html}
      {_df_to_html(pair.sensitivity_df[sensitivity_table_cols].sort_values('total_cost_level_bps'))}
    </div>
    """

    return section_html, include_plotlyjs


def _summary_dataframe(pair_reports: list[PairReportResult]) -> pd.DataFrame:
    """Build compact summary table for all pairs."""

    rows = [
        {
            "pair_id": p.pair_id,
            "latest_premium_pct": p.latest_premium_pct,
            "latest_rolling_z": p.latest_rolling_z,
            "half_life_days": p.half_life_days,
            "adf_p_value": p.adf_p_value,
            "eg_p_value": p.eg_p_value,
            "summary_score": p.summary_score,
            "breakeven_cost_level": p.breakeven_cost_level,
            "rolling_p_value_pass_rate": p.rolling_p_value_pass_rate,
            "rolling_beta_variance": p.rolling_beta_variance,
            "rolling_resid_std_drift": p.rolling_resid_std_drift,
            "rolling_coint_stability_score": p.rolling_coint_stability_score,
            "break_count": p.break_count,
            "max_break_confidence": p.max_break_confidence,
            "cusum_p_value": p.cusum_p_value,
            "aligned_sample_size": p.aligned_sample_size,
        }
        for p in pair_reports
    ]
    return pd.DataFrame(rows).sort_values("summary_score", ascending=False).reset_index(drop=True)


def _metadata_table(metadata: ReportMetadata) -> pd.DataFrame:
    """Render report-level metadata into table rows."""

    rows = [
        {"key": "git_commit", "value": metadata.git_commit},
        {"key": "generated_at_utc", "value": metadata.generated_at_utc},
        {"key": "data_fetch_time_utc", "value": metadata.data_fetch_time_utc},
        {"key": "start_date", "value": metadata.start_date},
        {"key": "end_date", "value": metadata.end_date},
        {"key": "expected_business_days", "value": metadata.expected_business_days},
        {"key": "pair_count", "value": metadata.pair_count},
        {"key": "total_aligned_samples", "value": metadata.total_aligned_samples},
        {"key": "mean_aligned_samples", "value": metadata.mean_aligned_samples},
        {"key": "avg_a_missing_rate", "value": metadata.avg_a_missing_rate},
        {"key": "avg_h_missing_rate", "value": metadata.avg_h_missing_rate},
        {"key": "avg_fx_missing_rate", "value": metadata.avg_fx_missing_rate},
        {"key": "fx_pair", "value": metadata.fx_pair},
        {"key": "fx_alignment", "value": metadata.fx_alignment},
        {"key": "cost_grid_size", "value": metadata.cost_grid_size},
        {
            "key": "cost_parameters",
            "value": json.dumps(metadata.cost_parameters, ensure_ascii=False, sort_keys=True),
        },
    ]
    return pd.DataFrame(rows)


def _pair_quality_table(pair_reports: list[PairReportResult]) -> pd.DataFrame:
    """Build per-pair sample and missing-rate table."""

    rows = [
        {
            "pair_id": item.pair_id,
            "aligned_sample_size": item.aligned_sample_size,
            "a_missing_days": item.a_missing_days,
            "h_missing_days": item.h_missing_days,
            "fx_missing_days": item.fx_missing_days,
            "a_missing_rate": item.a_missing_rate,
            "h_missing_rate": item.h_missing_rate,
            "fx_missing_rate": item.fx_missing_rate,
            "fx_alignment": "HKD->CNY",
        }
        for item in pair_reports
    ]
    return pd.DataFrame(rows)


def _pair_page_metadata_table(pair: PairReportResult, metadata: ReportMetadata) -> pd.DataFrame:
    """Build compact metadata table for one pair sub-page."""

    rows = [
        {"key": "git_commit", "value": metadata.git_commit},
        {"key": "data_fetch_time_utc", "value": metadata.data_fetch_time_utc},
        {"key": "sample_range", "value": f"{pair.sample_start} ~ {pair.sample_end}"},
        {"key": "aligned_sample_size", "value": pair.aligned_sample_size},
        {"key": "a_missing_rate", "value": pair.a_missing_rate},
        {"key": "h_missing_rate", "value": pair.h_missing_rate},
        {"key": "fx_missing_rate", "value": pair.fx_missing_rate},
        {"key": "fx_pair", "value": metadata.fx_pair},
        {"key": "fx_alignment", "value": metadata.fx_alignment},
        {
            "key": "cost_parameters",
            "value": json.dumps(metadata.cost_parameters, ensure_ascii=False, sort_keys=True),
        },
    ]
    return pd.DataFrame(rows)


def _build_cost_tolerance_table(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row cost tolerance summary table."""

    if sensitivity_df.empty:
        return pd.DataFrame(
            [
                {
                    "breakeven_total_cost": float("nan"),
                    "breakeven_slippage": float("nan"),
                    "worst_case_net_dd": float("nan"),
                    "borrow_bps_min": float("nan"),
                    "borrow_bps_max": float("nan"),
                    "best_net_cagr": float("nan"),
                    "median_net_sharpe": float("nan"),
                }
            ]
        )

    breakeven_total_cost = _safe_first_value(sensitivity_df, "breakeven_total_cost")
    if not np.isfinite(breakeven_total_cost):
        breakeven_total_cost = _safe_first_value(sensitivity_df, "breakeven_cost_level")

    breakeven_slippage = _safe_first_value(sensitivity_df, "breakeven_slippage")
    worst_case_net_dd = _safe_first_value(sensitivity_df, "worst_case_net_dd")
    if not np.isfinite(worst_case_net_dd) and "max_dd" in sensitivity_df.columns:
        worst_case_net_dd = float(sensitivity_df["max_dd"].min())

    borrow_min = (
        float(sensitivity_df["borrow_bps"].min()) if "borrow_bps" in sensitivity_df.columns else 0.0
    )
    borrow_max = (
        float(sensitivity_df["borrow_bps"].max()) if "borrow_bps" in sensitivity_df.columns else 0.0
    )

    return pd.DataFrame(
        [
            {
                "breakeven_total_cost": breakeven_total_cost,
                "breakeven_slippage": breakeven_slippage,
                "worst_case_net_dd": worst_case_net_dd,
                "borrow_bps_min": borrow_min,
                "borrow_bps_max": borrow_max,
                "best_net_cagr": float(sensitivity_df["net_cagr"].max()),
                "median_net_sharpe": float(sensitivity_df["net_sharpe"].median()),
            }
        ]
    )


def _build_cost_tolerance_radar(pair_id: str, tolerance_df: pd.DataFrame) -> str:
    """Build radar chart HTML for one pair cost tolerance profile."""

    row = tolerance_df.iloc[0]
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
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=pair_id,
        )
    )
    fig.update_layout(
        title=f"{pair_id} Cost Tolerance Radar",
        polar={"radialaxis": {"visible": True}},
        showlegend=False,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _safe_first_value(frame: pd.DataFrame, column: str) -> float:
    """Return first value for a column when available, else NaN."""

    if column not in frame.columns or frame.empty:
        return float("nan")
    return float(frame[column].iloc[0])


def _finite_or_zero(value: float) -> float:
    """Replace non-finite scalar with zero."""

    if np.isfinite(value):
        return float(value)
    return 0.0


def _reference_links_html() -> str:
    """Render references as HTML list items."""

    return "".join([f'<li><a href="{url}" target="_blank">{title}</a></li>' for title, url in REFERENCE_LINKS])


def _add_break_lines(fig: go.Figure, breaks: pd.DataFrame) -> None:
    """Add structural-break vertical lines to Plotly figures."""

    if breaks.empty or "break_date" not in breaks.columns:
        return

    for _, row in breaks.iterrows():
        break_date = pd.to_datetime(row["break_date"])
        confidence = float(row.get("confidence", float("nan")))
        line_color = "#f59e0b" if confidence >= 0.7 else "#fbbf24"
        fig.add_vline(
            x=break_date,
            line_dash="dot",
            line_color=line_color,
            line_width=1,
            opacity=0.75,
        )


def _df_to_html(frame: pd.DataFrame) -> str:
    """Render DataFrame to compact HTML table."""

    return frame.to_html(index=False, float_format=lambda x: f"{x:.4f}")


def _pair_id(a_ticker: str, h_ticker: str) -> str:
    """Build canonical pair identifier."""

    return f"{a_ticker}-{h_ticker}"


def _slug(text: str) -> str:
    """Convert text to filesystem-safe slug."""

    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _build_parser() -> argparse.ArgumentParser:
    """Build command line parser for report generation."""

    parser = argparse.ArgumentParser(description="Generate AH premium research report")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2024-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2025-12-31")
    parser.add_argument(
        "--pairs",
        default="",
        help="Comma-separated pair IDs (A_TICKER-H_TICKER). Empty means all in pairs.csv",
    )
    parser.add_argument("--pairs-csv", default="data/pairs.csv", help="Path to pairs.csv")
    parser.add_argument("--cache-dir", default="data/cache", help="Data cache directory")

    parser.add_argument("--window", type=int, default=252, help="Rolling window for metrics/strategy")
    parser.add_argument("--entry", type=float, default=2.0, help="Strategy entry z")
    parser.add_argument("--exit", type=float, default=0.5, help="Strategy exit z")
    parser.add_argument("--fx-pair", choices=["HKDCNY", "CNYHKD"], default="HKDCNY")

    parser.add_argument("--commission-min", type=float, default=0.0)
    parser.add_argument("--commission-max", type=float, default=10.0)
    parser.add_argument("--commission-step", type=float, default=2.0)
    parser.add_argument("--slippage-min", type=float, default=0.0)
    parser.add_argument("--slippage-max", type=float, default=20.0)
    parser.add_argument("--slippage-step", type=float, default=5.0)
    parser.add_argument("--stamp-min", type=float, default=0.0)
    parser.add_argument("--stamp-max", type=float, default=15.0)
    parser.add_argument("--stamp-step", type=float, default=5.0)

    parser.add_argument("--output", required=True, help="Output file (single) or directory (multi)")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")

    return parser


def main() -> None:
    """CLI entrypoint for report generation."""

    parser = _build_parser()
    args = parser.parse_args()

    pair_list = [item.strip() for item in args.pairs.split(",") if item.strip()]
    config = CostGridConfig(
        commission_min_bps=float(args.commission_min),
        commission_max_bps=float(args.commission_max),
        commission_step_bps=float(args.commission_step),
        slippage_min_bps=float(args.slippage_min),
        slippage_max_bps=float(args.slippage_max),
        slippage_step_bps=float(args.slippage_step),
        stamp_min_bps=float(args.stamp_min),
        stamp_max_bps=float(args.stamp_max),
        stamp_step_bps=float(args.stamp_step),
    )

    out = generate_report(
        start_date=args.start,
        end_date=args.end,
        pairs=pair_list if pair_list else None,
        cost_grid_config=config,
        output_path=args.output,
        output_mode=args.mode,
        pairs_csv=args.pairs_csv,
        cache_dir=args.cache_dir,
        fx_pair=args.fx_pair,
        window=int(args.window),
        entry=float(args.entry),
        exit_=float(args.exit),
    )
    print(f"Report generated: {out}")


if __name__ == "__main__":
    main()
