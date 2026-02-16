"""Build reproducible A/H case study markdown and supporting report assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ah_premium_lab.backtest import (
    ExecutabilityConfig,
    generate_cost_grid,
    generate_sensitivity_html_report,
    run_pair_cost_sensitivity,
    run_pairs_strategy,
)
from ah_premium_lab.core import compute_premium_metrics
from ah_premium_lab.data import CacheOnlyPriceProvider
from ah_premium_lab.stats import (
    adf_test,
    detect_structural_breaks,
    engle_granger_test,
    half_life_ar1,
    rolling_engle_granger,
    rolling_stability_metrics,
    summary_score,
)
from ah_premium_lab.universe import load_universe_frame

DEFAULT_PAIRS = ("600036.SS-3968.HK", "601318.SS-2318.HK")


@dataclass(frozen=True)
class CaseStudyResult:
    """Case study output for one AH pair."""

    pair_id: str
    name: str
    a_ticker: str
    h_ticker: str
    share_ratio: float
    sample_start: str
    sample_end: str
    n_obs: int
    latest_premium_pct: float
    latest_zscore: float
    latest_percentile: float
    adf_p: float
    adf_stat: float
    adf_lags: int
    eg_p: float
    eg_stat: float
    eg_lags: int
    eg_beta: float
    half_life_days: float
    summary_score_value: float
    rolling_window_count: int
    rolling_pass_rate: float
    rolling_beta_var: float
    rolling_resid_std_drift: float
    rolling_stability_score: float
    break_count: int
    max_break_confidence: float
    cusum_stat: float
    cusum_p: float
    gross_return: float
    net_return: float
    max_dd_gross: float
    max_dd_net: float
    missed_trades: int
    constraint_violations: int
    effective_turnover: float
    executability_score: float
    breakeven_total_cost: float
    breakeven_slippage: float
    worst_case_net_dd: float
    best_cost_level_bps: float
    best_net_cagr: float
    best_net_sharpe: float
    rolling_csv_path: Path
    breaks_csv_path: Path
    cost_csv_path: Path
    report_html_path: Path


def build_case_study(
    *,
    output_path: Path,
    reports_dir: Path,
    cache_dir: Path,
    pair_ids: list[str],
    start_date: str,
    end_date: str,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
) -> Path:
    """Build case study markdown and linked report artifacts."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    universe = load_universe_frame()
    provider = CacheOnlyPriceProvider(cache_dir=cache_dir)
    selected = _select_pairs(universe=universe, pair_ids=pair_ids)

    results: list[CaseStudyResult] = []
    for _, pair in selected.iterrows():
        result = _analyze_one_pair(
            provider=provider,
            pair=pair,
            reports_dir=reports_dir,
            start_date=start_date,
            end_date=end_date,
            window=window,
            entry=entry,
            exit_=exit_,
            fx_pair=fx_pair,
        )
        results.append(result)

    content = _render_case_study_markdown(
        output_path=output_path,
        reports_dir=reports_dir,
        cache_dir=cache_dir,
        pair_ids=pair_ids,
        start_date=start_date,
        end_date=end_date,
        window=window,
        entry=entry,
        exit_=exit_,
        fx_pair=fx_pair,
        results=results,
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _analyze_one_pair(
    *,
    provider: CacheOnlyPriceProvider,
    pair: pd.Series,
    reports_dir: Path,
    start_date: str,
    end_date: str,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
) -> CaseStudyResult:
    """Run full diagnostics and cost sensitivity for one pair."""

    pair_id = str(pair["pair_id"])
    name = str(pair["name"])
    a_ticker = str(pair["a_ticker"])
    h_ticker = str(pair["h_ticker"])
    share_ratio = float(pair["share_ratio"])
    slug = _slug(pair_id)

    a_series = provider.get_price(a_ticker, start_date, end_date).data["adj_close"]
    h_series = provider.get_price(h_ticker, start_date, end_date).data["adj_close"]
    fx_series = provider.get_fx(fx_pair, start_date, end_date).data["close"]

    aligned = compute_premium_metrics(
        a_price_cny=a_series,
        h_price_hkd=h_series,
        fx=fx_series,
        fx_quote=fx_pair,
        share_ratio=share_ratio,
        window=window,
    )
    latest = aligned.iloc[-1]

    adf_p, adf_stat, adf_lags = adf_test(aligned["log_spread"])
    eg_p, eg_beta, eg_resid = engle_granger_test(
        log_A=np.log(aligned["a_price_cny"]),
        log_H_fx=np.log(aligned["h_price_cny_equiv"]),
    )
    eg_resid_p, eg_stat, eg_lags = adf_test(eg_resid)
    if np.isfinite(eg_resid_p):
        eg_p = eg_resid_p

    half_life_days = half_life_ar1(aligned["log_spread"])
    score = summary_score(
        adf_p_value=adf_p,
        eg_p_value=eg_p,
        half_life_days=half_life_days,
        zscore_series=aligned["rolling_zscore"],
    )

    rolling = rolling_engle_granger(
        logA=np.log(aligned["a_price_cny"]),
        logHfx=np.log(aligned["h_price_cny_equiv"]),
        window=window,
        step=21,
    )
    rolling_metrics = rolling_stability_metrics(rolling)

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
    pair_frame = (
        aligned_reset.rename(columns={first_column: "date"})
        .loc[:, ["date", "a_price_cny", "h_price_cny_equiv"]]
        .rename(columns={"a_price_cny": "a_close", "h_price_cny_equiv": "h_cny"})
    )
    strategy = run_pairs_strategy(
        pair_frame,
        entry=entry,
        exit=exit_,
        z_window=window,
        cost_bps=0.0,
        share_ratio=share_ratio,
        fx_quote=fx_pair,
        executability_config=ExecutabilityConfig(
            enforce_a_share_t1=True,
            allow_short_a=False,
            allow_short_h=True,
        ),
    )

    cost_grid = generate_cost_grid(
        commission_bps_levels=(0, 2, 4, 6, 8, 10),
        slippage_bps_levels=(0, 5, 10, 15, 20),
        stamp_duty_bps_levels=(0, 5, 10, 15),
        borrow_bps_levels=(0.0, 100.0),
    )
    sensitivity = run_pair_cost_sensitivity(
        pair_frame=pair_frame,
        cost_grid=cost_grid,
        pair_id=pair_id,
        entry=entry,
        exit=exit_,
        z_window=window,
        share_ratio=share_ratio,
        fx_quote=fx_pair,
    )
    cost_df = sensitivity.grid_results.copy()
    best = cost_df.sort_values("net_cagr", ascending=False).iloc[0]

    rolling_csv = reports_dir / f"case_study_{slug}_rolling_coint.csv"
    breaks_csv = reports_dir / f"case_study_{slug}_breaks.csv"
    cost_csv = reports_dir / f"case_study_{slug}_cost_grid.csv"
    report_html = reports_dir / f"case_study_{slug}_cost_report.html"

    rolling.to_csv(rolling_csv, index=False)
    breaks_df.to_csv(breaks_csv, index=False)
    cost_df.to_csv(cost_csv, index=False)
    generate_sensitivity_html_report(
        result_df=cost_df,
        output_path=report_html,
        title=f"AH Case Study Cost Sensitivity - {pair_id}",
    )

    return CaseStudyResult(
        pair_id=pair_id,
        name=name,
        a_ticker=a_ticker,
        h_ticker=h_ticker,
        share_ratio=share_ratio,
        sample_start=str(aligned.index.min().date()),
        sample_end=str(aligned.index.max().date()),
        n_obs=int(aligned.shape[0]),
        latest_premium_pct=float(latest["premium_pct"] * 100.0),
        latest_zscore=float(latest["rolling_zscore"]),
        latest_percentile=float(latest["rolling_percentile"]),
        adf_p=float(adf_p),
        adf_stat=float(adf_stat),
        adf_lags=int(adf_lags),
        eg_p=float(eg_p),
        eg_stat=float(eg_stat),
        eg_lags=int(eg_lags),
        eg_beta=float(eg_beta),
        half_life_days=float(half_life_days),
        summary_score_value=float(score),
        rolling_window_count=int(rolling_metrics.n_windows),
        rolling_pass_rate=float(rolling_metrics.p_value_pass_rate),
        rolling_beta_var=float(rolling_metrics.beta_variance),
        rolling_resid_std_drift=float(rolling_metrics.resid_std_drift),
        rolling_stability_score=float(rolling_metrics.stability_score),
        break_count=int(breaks_df.shape[0]),
        max_break_confidence=(
            float(breaks_df["confidence"].max()) if not breaks_df.empty else float("nan")
        ),
        cusum_stat=float(breaks.cusum_stat),
        cusum_p=float(breaks.cusum_p_value),
        gross_return=float(strategy.daily["curve_gross"].iloc[-1] - 1.0),
        net_return=float(strategy.daily["curve_net"].iloc[-1] - 1.0),
        max_dd_gross=float(strategy.max_drawdown_gross),
        max_dd_net=float(strategy.max_drawdown_net),
        missed_trades=int(strategy.executability.missed_trades),
        constraint_violations=int(strategy.executability.constraint_violation_count),
        effective_turnover=float(strategy.executability.effective_turnover),
        executability_score=float(strategy.executability.executability_score),
        breakeven_total_cost=float(sensitivity.breakeven_total_cost),
        breakeven_slippage=float(sensitivity.breakeven_slippage),
        worst_case_net_dd=float(sensitivity.worst_case_net_dd),
        best_cost_level_bps=float(best["total_cost_level_bps"]),
        best_net_cagr=float(best["net_cagr"]),
        best_net_sharpe=float(best["net_sharpe"]),
        rolling_csv_path=rolling_csv,
        breaks_csv_path=breaks_csv,
        cost_csv_path=cost_csv,
        report_html_path=report_html,
    )


def _render_case_study_markdown(
    *,
    output_path: Path,
    reports_dir: Path,
    cache_dir: Path,
    pair_ids: list[str],
    start_date: str,
    end_date: str,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
    results: list[CaseStudyResult],
) -> str:
    """Render case study markdown from computed pair results."""

    rel_reports = reports_dir.as_posix()
    rel_cache = cache_dir.as_posix()
    pair_text = ",".join(pair_ids)
    cmd = (
        "PYTHONPATH=src python scripts/build_case_study.py "
        f'--pairs "{pair_text}" --start {start_date} --end {end_date} '
        f"--window {window} --entry {entry:.2f} --exit {exit_:.2f} --fx-pair {fx_pair}"
    )

    sections: list[str] = []
    doc_dir = output_path.parent
    for result in results:
        sections.append(_render_pair_section(result=result, doc_dir=doc_dir))
    section_text = "\n\n---\n\n".join(sections)

    return f"""# Case Study: AH 双对完整诊断（可复现）

本文档由仓库脚本自动生成，所有表格/阈值结论均来自仓库现有模块计算结果。

## 复现方式

```bash
{cmd}
```

输入数据：
- 仅使用离线缓存：`{rel_cache}`
- pair 来源：`data/pairs_master.csv`（若不存在则自动回退 `data/pairs.csv`）

输出文件：
- 本文档：`{output_path.as_posix()}`
- 报告与表格目录：`{rel_reports}`

参数：
- 日期区间：`{start_date}` ~ `{end_date}`
- rolling window：`{window}`
- 策略阈值：`entry={entry:.2f}`，`exit={exit_:.2f}`
- FX quote：`{fx_pair}`

---

{section_text}
"""


def _render_pair_section(*, result: CaseStudyResult, doc_dir: Path) -> str:
    """Render one pair section with full diagnostics and conclusions."""

    sample_rows = [
        ("Pair", f"`{result.pair_id}`"),
        ("Name", result.name),
        ("A/H", f"`{result.a_ticker}` / `{result.h_ticker}`"),
        ("Share Ratio", _fmt(result.share_ratio, 2)),
        ("Sample", f"{result.sample_start} ~ {result.sample_end}"),
        ("Aligned Obs", str(result.n_obs)),
        ("Latest Premium %", _fmt(result.latest_premium_pct, 3)),
        ("Latest Rolling Z", _fmt(result.latest_zscore, 3)),
        ("Latest Premium Percentile", _fmt(result.latest_percentile, 3)),
    ]
    diag_rows = [
        ("ADF p-value", _fmt(result.adf_p, 6)),
        ("ADF stat", _fmt(result.adf_stat, 4)),
        ("ADF lags", str(result.adf_lags)),
        ("EG p-value", _fmt(result.eg_p, 6)),
        ("EG stat", _fmt(result.eg_stat, 4)),
        ("EG lags", str(result.eg_lags)),
        ("EG beta", _fmt(result.eg_beta, 4)),
        ("Half-life (days)", _fmt(result.half_life_days, 2)),
        ("Summary Score (0-100)", _fmt(result.summary_score_value, 2)),
    ]
    rolling_rows = [
        ("Rolling windows", str(result.rolling_window_count)),
        ("p-value pass rate (<0.05)", _fmt(result.rolling_pass_rate, 4)),
        ("beta variance", _fmt(result.rolling_beta_var, 6)),
        ("resid std drift", _fmt(result.rolling_resid_std_drift, 4)),
        ("stability score (0-100)", _fmt(result.rolling_stability_score, 2)),
    ]
    break_rows = [
        ("Detected breakpoints", str(result.break_count)),
        ("Max breakpoint confidence", _fmt(result.max_break_confidence, 4)),
        ("CUSUM stat", _fmt(result.cusum_stat, 4)),
        ("CUSUM p-value", _fmt(result.cusum_p, 6)),
    ]
    exec_rows = [
        ("Gross return", _fmt(result.gross_return, 4)),
        ("Net return", _fmt(result.net_return, 4)),
        ("Max DD gross", _fmt(result.max_dd_gross, 4)),
        ("Max DD net", _fmt(result.max_dd_net, 4)),
        ("Missed trades", str(result.missed_trades)),
        ("Constraint violations", str(result.constraint_violations)),
        ("Effective turnover", _fmt(result.effective_turnover, 4)),
        ("Executability score (0-100)", _fmt(result.executability_score, 2)),
    ]
    cost_rows = [
        ("Best total cost level (bps)", _fmt(result.best_cost_level_bps, 2)),
        ("Best net CAGR", _fmt(result.best_net_cagr, 4)),
        ("Best net Sharpe", _fmt(result.best_net_sharpe, 4)),
        ("Breakeven total cost (bps)", _fmt(result.breakeven_total_cost, 2)),
        ("Breakeven slippage (bps)", _fmt(result.breakeven_slippage, 2)),
        ("Worst-case net drawdown", _fmt(result.worst_case_net_dd, 4)),
    ]

    conclusion = _build_conclusion(result)
    links = [
        (
            "Rolling coint table",
            _relative_path(result.rolling_csv_path, base_dir=doc_dir),
        ),
        (
            "Breakpoints table",
            _relative_path(result.breaks_csv_path, base_dir=doc_dir),
        ),
        (
            "Cost grid table",
            _relative_path(result.cost_csv_path, base_dir=doc_dir),
        ),
        (
            "Cost sensitivity report (含热力图/雷达图)",
            _relative_path(result.report_html_path, base_dir=doc_dir),
        ),
    ]
    links_text = "\n".join([f"- [{label}]({path})" for label, path in links])

    return f"""## {result.pair_id} - {result.name}

### 1) 样本与口径结果
{_render_table(sample_rows)}

### 2) 统计检验
{_render_table(diag_rows)}

### 3) Rolling 稳定性
{_render_table(rolling_rows)}

### 4) 结构突变
{_render_table(break_rows)}

### 5) 可执行性约束（研究版回测）
{_render_table(exec_rows)}

### 6) 成本敏感性阈值
{_render_table(cost_rows)}

结论：
- {conclusion}

图/表产物（全部由仓库模块生成）：
{links_text}
"""


def _build_conclusion(result: CaseStudyResult) -> str:
    """Build one-line decision summary for cost threshold and diagnostics."""

    stationarity_hint = (
        "统计检验偏弱" if result.adf_p > 0.05 or result.eg_p > 0.05 else "统计检验通过"
    )
    stability_hint = (
        "rolling 稳定性一般"
        if result.rolling_stability_score < 50.0
        else "rolling 稳定性较好"
    )
    break_hint = "存在结构突变风险" if result.break_count > 0 else "未检测到显著结构突变"
    total_cost_text = _fmt(result.breakeven_total_cost, 2)
    slippage_text = _fmt(result.breakeven_slippage, 2)
    if total_cost_text == "N/A" or slippage_text == "N/A":
        cost_hint = "在当前成本网格内未观察到净收益穿越 0 的阈值点。"
    else:
        cost_hint = (
            f"breakeven total cost 约 {total_cost_text} bps，"
            f"breakeven slippage 约 {slippage_text} bps。"
        )
    return f"{stationarity_hint}，{stability_hint}，{break_hint}；在成本阈值上，{cost_hint}"


def _render_table(rows: list[tuple[str, str]]) -> str:
    """Render two-column markdown table."""

    body = "\n".join([f"| {key} | {value} |" for key, value in rows])
    return f"| Metric | Value |\n|---|---|\n{body}"


def _fmt(value: float, digits: int) -> str:
    """Format numeric values with finite checks."""

    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _slug(text: str) -> str:
    """Create filesystem-safe slug."""

    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _relative_path(path: Path, *, base_dir: Path) -> str:
    """Return markdown-friendly relative path."""

    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _select_pairs(universe: pd.DataFrame, pair_ids: list[str]) -> pd.DataFrame:
    """Validate and select pair rows from universe table."""

    if universe.empty:
        raise ValueError("Universe table is empty")

    available = set(universe["pair_id"].astype(str).tolist())
    missing = [pair_id for pair_id in pair_ids if pair_id not in available]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Pairs not found in universe: {missing_text}")

    selected = universe[universe["pair_id"].isin(pair_ids)].copy()
    selected["pair_order"] = selected["pair_id"].map({pid: i for i, pid in enumerate(pair_ids)})
    selected = (
        selected.sort_values("pair_order").drop(columns=["pair_order"]).reset_index(drop=True)
    )
    return selected


def _parse_args() -> argparse.Namespace:
    """Parse command-line args."""

    parser = argparse.ArgumentParser(description="Build reproducible AH case study markdown")
    parser.add_argument("--output", type=Path, default=Path("docs/case_study.md"))
    parser.add_argument("--reports-dir", type=Path, default=Path("docs/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--entry", type=float, default=2.0)
    parser.add_argument("--exit", dest="exit_", type=float, default=0.5)
    parser.add_argument("--fx-pair", default="HKDCNY")
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""

    args = _parse_args()
    pair_ids = [item.strip() for item in str(args.pairs).split(",") if item.strip()]
    if len(pair_ids) != 2:
        raise ValueError("Case study requires exactly 2 pairs")

    output = build_case_study(
        output_path=args.output,
        reports_dir=args.reports_dir,
        cache_dir=args.cache_dir,
        pair_ids=pair_ids,
        start_date=str(args.start),
        end_date=str(args.end),
        window=int(args.window),
        entry=float(args.entry),
        exit_=float(args.exit_),
        fx_pair=str(args.fx_pair),
    )
    print(output.as_posix())


if __name__ == "__main__":
    main()
