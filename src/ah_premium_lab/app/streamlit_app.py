"""Streamlit dashboard for AH premium research workflows."""

from __future__ import annotations

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ah_premium_lab.backtest import (
    ExecutabilityConfig,
    generate_cost_grid,
    run_pair_cost_sensitivity,
    run_pairs_strategy,
)
from ah_premium_lab.core import compute_premium_metrics
from ah_premium_lab.data import (
    CacheOnlyPriceProvider,
    PriceProvider,
    PriceSeries,
    YahooFinanceProvider,
    check_fx_integrity,
    check_price_integrity,
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
from ah_premium_lab.universe import (
    DEFAULT_MAPPING_OVERRIDES_PATH,
    DEFAULT_PAIRS_FALLBACK_PATH,
    DEFAULT_PAIRS_MASTER_PATH,
    MappingRequiredError,
    combine_pair_quality,
    compute_series_quality,
    load_mapping_overrides,
    load_universe_frame,
    record_mapping_issue,
)

PAIRS_MASTER_CSV_PATH = DEFAULT_PAIRS_MASTER_PATH
PAIRS_FALLBACK_CSV_PATH = DEFAULT_PAIRS_FALLBACK_PATH
MAPPING_OVERRIDES_PATH = DEFAULT_MAPPING_OVERRIDES_PATH
CACHE_DIR = Path("data/cache")


def main() -> None:
    """Render the two-page Streamlit research dashboard."""

    st.set_page_config(page_title="AH Premium Lab", page_icon="📈", layout="wide")
    st.title("AH Premium Lab")
    st.caption("研究用途：AH 价差仪表盘、统计检验与成本敏感性分析（非交易建议）")
    offline_mode = _is_offline_mode()
    if offline_mode:
        st.info("OFFLINE=1：仅从 data/cache 读取数据，不会调用 yfinance。")

    universe = _load_universe(
        master_csv_path=str(PAIRS_MASTER_CSV_PATH),
        fallback_csv_path=str(PAIRS_FALLBACK_CSV_PATH),
    )
    if universe.empty:
        st.error("Universe 为空，无法渲染页面。请检查 pairs_master.csv / pairs.csv。")
        return

    controls = _sidebar_controls()
    start_date = controls["start_date"]
    end_date = controls["end_date"]
    window = controls["window"]
    entry = controls["entry"]
    exit_ = controls["exit"]
    fx_pair = controls["fx_pair"]
    missing_threshold = controls["missing_threshold"]
    enforce_a_share_t1 = controls["enforce_a_share_t1"]
    allow_short_a = controls["allow_short_a"]
    allow_short_h = controls["allow_short_h"]

    overview = _build_overview(
        universe=universe,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        window=window,
        entry=entry,
        exit_=exit_,
        fx_pair=fx_pair,
        missing_threshold=missing_threshold,
        enforce_a_share_t1=enforce_a_share_t1,
        allow_short_a=allow_short_a,
        allow_short_h=allow_short_h,
        offline_mode=offline_mode,
    )

    if controls["page"] == "Overview":
        _render_overview(
            overview=overview,
            keyword=controls["keyword"],
            score_threshold=controls["score_threshold"],
            premium_percentile_threshold=controls["premium_percentile_threshold"],
            min_coverage_pct=controls["min_coverage_pct"],
            max_gap_days_filter=controls["max_gap_days_filter"],
            min_data_quality_score=controls["min_data_quality_score"],
            min_executability_score=controls["min_executability_score"],
            only_low_quality=controls["only_low_quality"],
        )
        return

    _render_pair_detail(
        universe=universe,
        overview=overview,
        start_date=start_date,
        end_date=end_date,
        window=window,
        entry=entry,
        exit_=exit_,
        fx_pair=fx_pair,
        missing_threshold=missing_threshold,
        enforce_a_share_t1=enforce_a_share_t1,
        allow_short_a=allow_short_a,
        allow_short_h=allow_short_h,
        offline_mode=offline_mode,
    )


@st.cache_data(show_spinner=False)
def _load_universe(master_csv_path: str, fallback_csv_path: str) -> pd.DataFrame:
    """Load pair universe from `pairs_master.csv` with fallback to `pairs.csv`."""

    return load_universe_frame(
        master_path=master_csv_path,
        fallback_path=fallback_csv_path,
    )


@st.cache_data(show_spinner=False)
def _compute_pair_analytics(
    *,
    pair_id: str,
    name: str,
    a_ticker: str,
    h_ticker: str,
    share_ratio: float,
    notes: str,
    start_date: str,
    end_date: str,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
    missing_threshold: float,
    enforce_a_share_t1: bool,
    allow_short_a: bool,
    allow_short_h: bool,
    offline_mode: bool,
) -> dict[str, object]:
    """Fetch market data and compute all pair-level metrics for one pair."""

    provider = _build_provider(offline_mode=offline_mode)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_series = _get_price_with_mapping(
            provider=provider,
            ticker=a_ticker,
            leg="A",
            pair_id=pair_id,
            start_date=start_date,
            end_date=end_date,
            offline_mode=offline_mode,
        )
        h_series = _get_price_with_mapping(
            provider=provider,
            ticker=h_ticker,
            leg="H",
            pair_id=pair_id,
            start_date=start_date,
            end_date=end_date,
            offline_mode=offline_mode,
        )
        try:
            fx_series = provider.get_fx(fx_pair, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            if offline_mode:
                raise MappingRequiredError(
                    f"FX {fx_pair} 离线缓存缺失或不可读（OFFLINE=1），"
                    "请先联网运行一次以填充 data/cache，或切换 OFFLINE=0。"
                ) from exc
            raise

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a_integrity = check_price_integrity(a_series)
        h_integrity = check_price_integrity(h_series)
        fx_integrity = check_fx_integrity(fx_series)

    aligned = compute_premium_metrics(
        a_price_cny=a_series.data["adj_close"],
        h_price_hkd=h_series.data["adj_close"],
        fx=fx_series.data["close"],
        fx_quote=fx_pair,
        share_ratio=share_ratio,
        window=window,
    )

    adf_p, adf_stat, adf_lags = adf_test(aligned["log_spread"])
    eg_p, eg_beta, eg_resid = engle_granger_test(
        log_A=np.log(aligned["a_price_cny"]),
        log_H_fx=np.log(aligned["h_price_cny_equiv"]),
    )
    _, eg_stat, eg_lags = adf_test(eg_resid)

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
    strategy_frame = aligned.reset_index()
    first_column = str(strategy_frame.columns[0])
    strategy_frame = strategy_frame.rename(columns={first_column: "date"})
    strategy_frame = strategy_frame[["date", "a_price_cny", "h_price_cny_equiv"]].rename(
        columns={"a_price_cny": "a_close", "h_price_cny_equiv": "h_cny"}
    )
    strategy = run_pairs_strategy(
        strategy_frame,
        entry=entry,
        exit=exit_,
        z_window=window,
        cost_bps=0.0,
        share_ratio=share_ratio,
        executability_config=ExecutabilityConfig(
            enforce_a_share_t1=bool(enforce_a_share_t1),
            allow_short_a=bool(allow_short_a),
            allow_short_h=bool(allow_short_h),
        ),
    )

    a_quality = compute_series_quality(
        a_series.data["adj_close"],
        start_date=start_date,
        end_date=end_date,
    )
    h_quality = compute_series_quality(
        h_series.data["adj_close"],
        start_date=start_date,
        end_date=end_date,
    )
    fx_quality = compute_series_quality(
        fx_series.data["close"],
        start_date=start_date,
        end_date=end_date,
    )
    pair_quality = combine_pair_quality(
        a_quality,
        h_quality,
        fx_quality,
        missing_threshold=missing_threshold,
    )

    latest = aligned.iloc[-1]
    aligned_df = aligned.reset_index().rename(columns={"index": "date"})
    rolling_coint_df = rolling_coint.copy()
    if not rolling_coint_df.empty:
        rolling_coint_df["window_start"] = pd.to_datetime(rolling_coint_df["window_start"])
        rolling_coint_df["window_end"] = pd.to_datetime(rolling_coint_df["window_end"])
    breaks_df = breaks.breakpoints.copy()
    if not breaks_df.empty:
        breaks_df["break_date"] = pd.to_datetime(breaks_df["break_date"])
    quality_table = pd.DataFrame(
        [
            {
                "leg": "A",
                "coverage_pct": a_quality.coverage_pct,
                "max_gap_days": a_quality.max_gap_days,
                "outlier_count": a_quality.outlier_count,
                "missing_rate": a_quality.missing_rate,
            },
            {
                "leg": "H",
                "coverage_pct": h_quality.coverage_pct,
                "max_gap_days": h_quality.max_gap_days,
                "outlier_count": h_quality.outlier_count,
                "missing_rate": h_quality.missing_rate,
            },
            {
                "leg": "FX",
                "coverage_pct": fx_quality.coverage_pct,
                "max_gap_days": fx_quality.max_gap_days,
                "outlier_count": fx_quality.outlier_count,
                "missing_rate": fx_quality.missing_rate,
            },
        ]
    )

    return {
        "pair_id": pair_id,
        "name": name,
        "a_ticker": a_ticker,
        "h_ticker": h_ticker,
        "share_ratio": share_ratio,
        "notes": notes,
        "aligned_frame": aligned_df,
        "sample_start": str(aligned.index.min().date()),
        "sample_end": str(aligned.index.max().date()),
        "aligned_sample_size": int(aligned.shape[0]),
        "latest_premium_pct": float(latest["premium_pct"] * 100.0),
        "latest_rolling_z": float(latest["rolling_zscore"]),
        "latest_premium_percentile": float(latest["rolling_percentile"]),
        "half_life_days": float(half_life),
        "adf_p_value": float(adf_p),
        "adf_stat": float(adf_stat),
        "adf_lags": int(adf_lags),
        "eg_p_value": float(eg_p),
        "eg_beta": float(eg_beta),
        "eg_stat": float(eg_stat),
        "eg_lags": int(eg_lags),
        "summary_score": float(score),
        "rolling_coint_frame": rolling_coint_df,
        "rolling_p_value_pass_rate": float(coint_stability.p_value_pass_rate),
        "rolling_beta_variance": float(coint_stability.beta_variance),
        "rolling_resid_std_drift": float(coint_stability.resid_std_drift),
        "rolling_coint_stability_score": float(coint_stability.stability_score),
        "rolling_window_count": int(coint_stability.n_windows),
        "breaks_frame": breaks_df,
        "break_count": int(breaks_df.shape[0]),
        "max_break_confidence": (
            float(breaks_df["confidence"].max()) if not breaks_df.empty else 0.0
        ),
        "cusum_stat": float(breaks.cusum_stat),
        "cusum_p_value": float(breaks.cusum_p_value),
        "missed_trades": int(strategy.executability.missed_trades),
        "constraint_violation_count": int(strategy.executability.constraint_violation_count),
        "effective_turnover": float(strategy.executability.effective_turnover),
        "executability_score": float(strategy.executability.executability_score),
        "coverage_pct": float(pair_quality.coverage_pct),
        "max_gap_days": int(pair_quality.max_gap_days),
        "outlier_count": int(pair_quality.outlier_count),
        "data_quality_score": float(pair_quality.quality_score),
        "data_quality_flag": pair_quality.quality_flag,
        "missing_threshold_breached": bool(pair_quality.missing_threshold_breached),
        "max_missing_rate": float(pair_quality.max_missing_rate),
        "data_quality_table": quality_table,
        "integrity": {
            "A": {
                "missing_days": a_integrity.missing_days,
                "jump_count": a_integrity.jump_count,
                "warnings": list(a_integrity.warnings),
            },
            "H": {
                "missing_days": h_integrity.missing_days,
                "jump_count": h_integrity.jump_count,
                "warnings": list(h_integrity.warnings),
            },
            "FX": {
                "missing_days": fx_integrity.missing_days,
                "jump_count": fx_integrity.jump_count,
                "warnings": list(fx_integrity.warnings),
            },
        },
    }


@st.cache_data(show_spinner=False)
def _build_overview(
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
    missing_threshold: float,
    enforce_a_share_t1: bool,
    allow_short_a: bool,
    allow_short_h: bool,
    offline_mode: bool,
) -> pd.DataFrame:
    """Compute overview table for all pairs in the universe."""

    rows: list[dict[str, object]] = []

    for _, row in universe.iterrows():
        pair_id = str(row["pair_id"])
        try:
            result = _compute_pair_analytics(
                pair_id=pair_id,
                name=str(row["name"]),
                a_ticker=str(row["a_ticker"]),
                h_ticker=str(row["h_ticker"]),
                share_ratio=float(row["share_ratio"]),
                notes=str(row["notes"]),
                start_date=start_date,
                end_date=end_date,
                window=window,
                entry=entry,
                exit_=exit_,
                fx_pair=fx_pair,
                missing_threshold=float(missing_threshold),
                enforce_a_share_t1=bool(enforce_a_share_t1),
                allow_short_a=bool(allow_short_a),
                allow_short_h=bool(allow_short_h),
                offline_mode=bool(offline_mode),
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "name": result["name"],
                    "a_ticker": result["a_ticker"],
                    "h_ticker": result["h_ticker"],
                    "notes": result["notes"],
                    "aligned_sample_size": result["aligned_sample_size"],
                    "sample_start": result["sample_start"],
                    "sample_end": result["sample_end"],
                    "latest_premium_pct": result["latest_premium_pct"],
                    "latest_rolling_z": result["latest_rolling_z"],
                    "latest_premium_percentile": result["latest_premium_percentile"],
                    "half_life_days": result["half_life_days"],
                    "adf_p_value": result["adf_p_value"],
                    "eg_p_value": result["eg_p_value"],
                    "summary_score": result["summary_score"],
                    "coverage_pct": result["coverage_pct"],
                    "max_gap_days": result["max_gap_days"],
                    "outlier_count": result["outlier_count"],
                    "data_quality_score": result["data_quality_score"],
                    "data_quality_flag": result["data_quality_flag"],
                    "missing_threshold_breached": result["missing_threshold_breached"],
                    "missed_trades": result["missed_trades"],
                    "constraint_violation_count": result["constraint_violation_count"],
                    "effective_turnover": result["effective_turnover"],
                    "executability_score": result["executability_score"],
                    "mapping_required": False,
                    "status": "ok",
                    "error": "",
                }
            )
        except MappingRequiredError as exc:
            rows.append(
                {
                    "pair_id": pair_id,
                    "name": row["name"],
                    "a_ticker": row["a_ticker"],
                    "h_ticker": row["h_ticker"],
                    "notes": row["notes"],
                    "aligned_sample_size": 0,
                    "sample_start": "",
                    "sample_end": "",
                    "latest_premium_pct": np.nan,
                    "latest_rolling_z": np.nan,
                    "latest_premium_percentile": np.nan,
                    "half_life_days": np.nan,
                    "adf_p_value": np.nan,
                    "eg_p_value": np.nan,
                    "summary_score": np.nan,
                    "coverage_pct": np.nan,
                    "max_gap_days": np.nan,
                    "outlier_count": np.nan,
                    "data_quality_score": np.nan,
                    "data_quality_flag": "unknown",
                    "missing_threshold_breached": False,
                    "missed_trades": np.nan,
                    "constraint_violation_count": np.nan,
                    "effective_turnover": np.nan,
                    "executability_score": np.nan,
                    "mapping_required": True,
                    "status": "mapping_required",
                    "error": str(exc),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "pair_id": pair_id,
                    "name": row["name"],
                    "a_ticker": row["a_ticker"],
                    "h_ticker": row["h_ticker"],
                    "notes": row["notes"],
                    "aligned_sample_size": 0,
                    "sample_start": "",
                    "sample_end": "",
                    "latest_premium_pct": np.nan,
                    "latest_rolling_z": np.nan,
                    "latest_premium_percentile": np.nan,
                    "half_life_days": np.nan,
                    "adf_p_value": np.nan,
                    "eg_p_value": np.nan,
                    "summary_score": np.nan,
                    "coverage_pct": np.nan,
                    "max_gap_days": np.nan,
                    "outlier_count": np.nan,
                    "data_quality_score": np.nan,
                    "data_quality_flag": "unknown",
                    "missing_threshold_breached": False,
                    "missed_trades": np.nan,
                    "constraint_violation_count": np.nan,
                    "effective_turnover": np.nan,
                    "executability_score": np.nan,
                    "mapping_required": False,
                    "status": "error",
                    "error": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    status_order = {"ok": 0, "mapping_required": 1, "error": 2}
    out["status_rank"] = out["status"].map(status_order).fillna(99)
    out = (
        out.sort_values(["status_rank", "summary_score"], ascending=[True, False])
        .drop(columns=["status_rank"])
        .reset_index(drop=True)
    )
    return out


def _render_overview(
    *,
    overview: pd.DataFrame,
    keyword: str,
    score_threshold: float,
    premium_percentile_threshold: float,
    min_coverage_pct: float,
    max_gap_days_filter: int,
    min_data_quality_score: float,
    min_executability_score: float,
    only_low_quality: bool,
) -> None:
    """Render Overview page."""

    st.subheader("Overview")

    filtered = overview[overview["status"] == "ok"].copy()
    filtered = filtered[filtered["summary_score"] >= score_threshold]
    filtered = filtered[filtered["latest_premium_percentile"] >= premium_percentile_threshold]
    filtered = filtered[filtered["coverage_pct"] >= min_coverage_pct]
    filtered = filtered[filtered["max_gap_days"] <= max_gap_days_filter]
    filtered = filtered[filtered["data_quality_score"] >= min_data_quality_score]
    filtered = filtered[filtered["executability_score"] >= min_executability_score]

    if only_low_quality:
        filtered = filtered[filtered["data_quality_flag"] == "poor"]

    if keyword.strip():
        key = keyword.strip().lower()
        mask = filtered["name"].astype(str).str.lower().str.contains(key, na=False) | filtered[
            "notes"
        ].astype(str).str.lower().str.contains(key, na=False)
        filtered = filtered[mask]

    st.caption(f"筛选后 {len(filtered)} / {len(overview[overview['status'] == 'ok'])} 个股票对")

    display_cols = [
        "pair_id",
        "name",
        "latest_premium_pct",
        "latest_rolling_z",
        "half_life_days",
        "adf_p_value",
        "eg_p_value",
        "summary_score",
        "coverage_pct",
        "max_gap_days",
        "outlier_count",
        "data_quality_score",
        "data_quality_flag",
        "missing_threshold_breached",
        "executability_score",
        "missed_trades",
        "constraint_violation_count",
        "effective_turnover",
        "latest_premium_percentile",
        "sample_start",
        "sample_end",
        "aligned_sample_size",
        "notes",
    ]
    styled = filtered[display_cols].style.apply(_style_quality_rows, axis=1)
    st.dataframe(styled, use_container_width=True)

    with st.expander("Data Quality Table", expanded=False):
        quality_cols = [
            "pair_id",
            "coverage_pct",
            "max_gap_days",
            "outlier_count",
            "data_quality_score",
            "data_quality_flag",
            "missing_threshold_breached",
            "executability_score",
            "missed_trades",
            "constraint_violation_count",
            "effective_turnover",
        ]
        st.dataframe(filtered[quality_cols], use_container_width=True)

    mapping_rows = overview[overview["status"] == "mapping_required"]
    if not mapping_rows.empty:
        st.error("检测到 ticker 拉取失败，已记录到 data/mapping_overrides.csv，需要人工修正。")
        st.dataframe(
            mapping_rows[["pair_id", "name", "a_ticker", "h_ticker", "error"]],
            use_container_width=True,
        )
        mapping_table = _load_mapping_overrides(str(MAPPING_OVERRIDES_PATH))
        if not mapping_table.empty:
            with st.expander("mapping_overrides.csv", expanded=False):
                st.dataframe(mapping_table, use_container_width=True)

    error_rows = overview[overview["status"] == "error"]
    if not error_rows.empty:
        with st.expander("加载失败的股票对", expanded=False):
            st.dataframe(error_rows[["pair_id", "name", "error"]], use_container_width=True)


def _render_pair_detail(
    *,
    universe: pd.DataFrame,
    overview: pd.DataFrame,
    start_date: date,
    end_date: date,
    window: int,
    entry: float,
    exit_: float,
    fx_pair: str,
    missing_threshold: float,
    enforce_a_share_t1: bool,
    allow_short_a: bool,
    allow_short_h: bool,
    offline_mode: bool,
) -> None:
    """Render Pair Detail page."""

    st.subheader("Pair Detail")

    valid_pairs = overview[overview["status"] == "ok"]["pair_id"].tolist()
    if not valid_pairs:
        st.warning("当前参数下没有可展示的股票对。")
        return

    default_pair = valid_pairs[0]
    selected_pair = st.selectbox(
        "选择股票对",
        options=valid_pairs,
        index=valid_pairs.index(default_pair),
    )

    pair_meta = universe[universe["pair_id"] == selected_pair].iloc[0]
    result = _compute_pair_analytics(
        pair_id=str(pair_meta["pair_id"]),
        name=str(pair_meta["name"]),
        a_ticker=str(pair_meta["a_ticker"]),
        h_ticker=str(pair_meta["h_ticker"]),
        share_ratio=float(pair_meta["share_ratio"]),
        notes=str(pair_meta["notes"]),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        window=window,
        entry=entry,
        exit_=exit_,
        fx_pair=fx_pair,
        missing_threshold=missing_threshold,
        enforce_a_share_t1=enforce_a_share_t1,
        allow_short_a=allow_short_a,
        allow_short_h=allow_short_h,
        offline_mode=offline_mode,
    )

    aligned = pd.DataFrame(result["aligned_frame"])

    st.markdown(
        f"**{result['name']}** (`{result['a_ticker']}` / `{result['h_ticker']}`) | "
        f"样本期: {result['sample_start']} ~ {result['sample_end']} | "
        f"对齐样本量: {result['aligned_sample_size']}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("latest premium%", f"{float(result['latest_premium_pct']):.3f}")
    m2.metric("rolling z", f"{float(result['latest_rolling_z']):.3f}")
    m3.metric("half-life (days)", f"{float(result['half_life_days']):.2f}")
    m4.metric("summary score", f"{float(result['summary_score']):.1f}")

    m5, m6, m7 = st.columns(3)
    m5.metric("ADF p", f"{float(result['adf_p_value']):.4f}")
    m6.metric("EG p", f"{float(result['eg_p_value']):.4f}")
    m7.metric("EG beta", f"{float(result['eg_beta']):.4f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rolling EG pass rate", f"{float(result['rolling_p_value_pass_rate']):.3f}")
    c2.metric("rolling beta var", f"{float(result['rolling_beta_variance']):.6f}")
    c3.metric("rolling resid_std drift", f"{float(result['rolling_resid_std_drift']):.3f}")
    c4.metric(
        "rolling coint stability",
        f"{float(result['rolling_coint_stability_score']):.2f}",
    )
    b1, b2, b3 = st.columns(3)
    b1.metric("break count", f"{int(result['break_count'])}")
    b2.metric("max break confidence", f"{float(result['max_break_confidence']):.2f}")
    b3.metric("CUSUM p", f"{float(result['cusum_p_value']):.4f}")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("executability score", f"{float(result['executability_score']):.2f}")
    e2.metric("missed trades", f"{int(result['missed_trades'])}")
    e3.metric(
        "constraint violations",
        f"{int(result['constraint_violation_count'])}",
    )
    e4.metric("effective turnover", f"{float(result['effective_turnover']):.3f}")

    breaks = pd.DataFrame(result["breaks_frame"])
    if not breaks.empty:
        breaks["break_date"] = pd.to_datetime(breaks["break_date"])

    st.markdown("**Data Quality**")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("coverage%", f"{float(result['coverage_pct']):.2f}")
    q2.metric("max_gap_days", f"{int(result['max_gap_days'])}")
    q3.metric("outlier_count", f"{int(result['outlier_count'])}")
    q4.metric("quality_score", f"{float(result['data_quality_score']):.2f}")

    if bool(result["missing_threshold_breached"]):
        st.error("该股票对缺失率超过阈值，已标记为低质量数据。")
    elif str(result["data_quality_flag"]) == "warning":
        st.warning("该股票对数据质量一般，请谨慎解读统计结果。")
    else:
        st.success("该股票对数据质量正常。")

    with st.expander("Per-leg Data Quality Table", expanded=False):
        st.dataframe(pd.DataFrame(result["data_quality_table"]), use_container_width=True)

    rolling_coint = pd.DataFrame(result["rolling_coint_frame"])
    if not rolling_coint.empty:
        p_fig = px.line(
            rolling_coint,
            x="window_end",
            y="p_value",
            title="Rolling Engle-Granger p-value",
        )
        p_fig.add_hline(y=0.05, line_dash="dash", line_color="red")
        st.plotly_chart(p_fig, use_container_width=True)

        beta_fig = px.line(
            rolling_coint,
            x="window_end",
            y="beta",
            title="Rolling Engle-Granger beta",
        )
        st.plotly_chart(beta_fig, use_container_width=True)
    else:
        st.info("rolling 协整窗口不足，未生成 rolling p-value / beta 图。")

    st.markdown("**结构突变检测**")
    if breaks.empty:
        st.info("未检测到显著结构突变。")
    else:
        st.dataframe(
            breaks[["break_date", "confidence", "p_value", "shift_zscore", "mean_shift"]],
            use_container_width=True,
        )

    premium_fig = px.line(aligned, x="date", y="premium_pct", title="Premium %")
    _add_breakpoint_vlines(premium_fig, breaks)
    st.plotly_chart(premium_fig, use_container_width=True)

    spread_fig = px.line(aligned, x="date", y="log_spread", title="Log Spread")
    _add_breakpoint_vlines(spread_fig, breaks)
    st.plotly_chart(spread_fig, use_container_width=True)

    z_fig = px.line(aligned, x="date", y="rolling_zscore", title="Rolling Z-Score")
    _add_breakpoint_vlines(z_fig, breaks)
    z_fig.add_hline(y=entry, line_dash="dash", line_color="red")
    z_fig.add_hline(y=-entry, line_dash="dash", line_color="red")
    z_fig.add_hline(y=exit_, line_dash="dot", line_color="gray")
    z_fig.add_hline(y=-exit_, line_dash="dot", line_color="gray")
    st.plotly_chart(z_fig, use_container_width=True)

    st.markdown("**缺失数据警告**")
    integrity = result["integrity"]
    warning_messages = _format_integrity_warnings(integrity)
    if warning_messages:
        for msg in warning_messages:
            st.warning(msg)
    else:
        st.success("未发现缺失交易日或异常跳点。")

    st.markdown("**成本敏感性（可选，若已生成缓存）**")
    st.caption("研究敏感性分析，不构成交易建议。")

    cached_sensitivity = _load_sensitivity_cache(selected_pair)
    if not cached_sensitivity.empty:
        _render_sensitivity_block(selected_pair, cached_sensitivity)
    else:
        st.info("未发现成本敏感性缓存。")

    if st.button("生成/刷新成本敏感性缓存", use_container_width=False):
        sensitivity = _compute_and_cache_sensitivity(
            pair_id=selected_pair,
            aligned_frame=aligned,
            entry=entry,
            exit_=exit_,
            window=window,
        )
        st.success("成本敏感性缓存已生成。")
        _render_sensitivity_block(selected_pair, sensitivity)


@st.cache_data(show_spinner=False)
def _compute_and_cache_sensitivity(
    *,
    pair_id: str,
    aligned_frame: pd.DataFrame,
    entry: float,
    exit_: float,
    window: int,
) -> pd.DataFrame:
    """Compute pair cost sensitivity and persist to parquet cache."""

    pair_frame = aligned_frame[["date", "a_price_cny", "h_price_cny_equiv"]].copy()
    pair_frame = pair_frame.rename(columns={"a_price_cny": "a_close", "h_price_cny_equiv": "h_cny"})

    grid = generate_cost_grid(
        commission_bps_levels=(0, 2, 4, 6, 8, 10),
        slippage_bps_levels=(0, 5, 10, 15, 20),
        stamp_duty_bps_levels=(0, 5, 10, 15),
    )

    output = run_pair_cost_sensitivity(
        pair_frame=pair_frame,
        cost_grid=grid,
        pair_id=pair_id,
        entry=entry,
        exit=exit_,
        z_window=window,
    )

    cache_path = _sensitivity_cache_path(pair_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output.grid_results.to_parquet(cache_path, index=False)
    return output.grid_results


def _load_sensitivity_cache(pair_id: str) -> pd.DataFrame:
    """Load cached pair sensitivity table if present."""

    cache_path = _sensitivity_cache_path(pair_id)
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(cache_path)


def _render_sensitivity_block(pair_id: str, sensitivity: pd.DataFrame) -> None:
    """Render cached or newly computed sensitivity analysis block."""

    if sensitivity.empty:
        st.info("暂无成本敏感性结果。")
        return

    pivot = (
        sensitivity.groupby(["slippage_bps", "commission_a_bps"], as_index=True)["net_cagr"]
        .mean()
        .unstack("commission_a_bps")
        .sort_index()
    )

    fig = px.imshow(
        pivot,
        labels={"x": "Commission A (bps)", "y": "Slippage (bps)", "color": "Avg Net CAGR"},
        title=f"{pair_id} 成本容忍度热力图（对 stamp duty 取均值）",
        aspect="auto",
        color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig, use_container_width=True)

    table_cols = [
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
    table_cols = [col for col in table_cols if col in sensitivity.columns]
    st.dataframe(
        sensitivity[table_cols].sort_values(
            ["total_cost_level_bps", "commission_a_bps", "slippage_bps", "stamp_duty_a_bps"]
        ),
        use_container_width=True,
    )


def _is_offline_mode() -> bool:
    """Return whether offline mode is enabled via `OFFLINE` env."""

    normalized = os.getenv("OFFLINE", "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _build_provider(*, offline_mode: bool) -> PriceProvider:
    """Build market data provider based on offline switch."""

    if offline_mode:
        return CacheOnlyPriceProvider(cache_dir=CACHE_DIR)
    return YahooFinanceProvider(cache_dir=CACHE_DIR)


def _get_price_with_mapping(
    *,
    provider: PriceProvider,
    ticker: str,
    leg: str,
    pair_id: str,
    start_date: str,
    end_date: str,
    offline_mode: bool,
) -> PriceSeries:
    """Fetch one ticker and record manual-mapping requirement on failure."""

    try:
        return provider.get_price(ticker, start_date, end_date)
    except Exception as exc:  # noqa: BLE001
        if offline_mode:
            raise MappingRequiredError(
                f"{leg} ticker {ticker} 离线缓存缺失或不可读（OFFLINE=1），"
                "请先联网运行一次以填充 data/cache，或切换 OFFLINE=0。"
            ) from exc
        mapping_path = record_mapping_issue(
            ticker=ticker,
            reason=f"{leg} leg fetch failed for {pair_id}: {exc}",
            path=MAPPING_OVERRIDES_PATH,
        )
        raise MappingRequiredError(
            f"{leg} ticker {ticker} 拉取失败，已记录到 {mapping_path.as_posix()}，需要人工修正。"
        ) from exc


@st.cache_data(show_spinner=False)
def _load_mapping_overrides(path: str) -> pd.DataFrame:
    """Load mapping override table for UI diagnostics."""

    return load_mapping_overrides(path)


def _style_quality_rows(row: pd.Series) -> list[str]:
    """Apply red highlight style for low-quality rows."""

    is_poor = str(row.get("data_quality_flag", "")) == "poor"
    breached = bool(row.get("missing_threshold_breached", False))
    style = "background-color: #fee2e2; color: #991b1b;" if is_poor or breached else ""
    return [style for _ in row.index]


def _format_integrity_warnings(integrity: object) -> list[str]:
    """Format integrity summary into user-facing warning messages."""

    if not isinstance(integrity, dict):
        return []

    messages: list[str] = []
    for leg in ["A", "H", "FX"]:
        leg_payload = integrity.get(leg, {})
        if not isinstance(leg_payload, dict):
            continue

        missing_days = int(leg_payload.get("missing_days", 0))
        jump_count = int(leg_payload.get("jump_count", 0))

        if missing_days > 0:
            messages.append(f"{leg} leg 存在缺失交易日: {missing_days}")
        if jump_count > 0:
            messages.append(f"{leg} leg 检测到异常跳点: {jump_count}")

    return messages


def _add_breakpoint_vlines(fig: go.Figure, breaks: pd.DataFrame) -> None:
    """Annotate a Plotly figure with structural-break vertical lines."""

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


def _pair_id(a_ticker: str, h_ticker: str) -> str:
    """Build canonical pair id."""

    return f"{a_ticker}-{h_ticker}"


def _slug(text: str) -> str:
    """Build filesystem-safe slug."""

    return "".join(ch if ch.isalnum() else "_" for ch in text)


def _sensitivity_cache_path(pair_id: str) -> Path:
    """Return sensitivity cache path for one pair."""

    return CACHE_DIR / f"sensitivity_{_slug(pair_id)}.parquet"


def _sidebar_controls() -> dict[str, object]:
    """Render and return sidebar controls."""

    today = date.today()
    default_start = today - timedelta(days=365 * 2)

    with st.sidebar:
        page = st.radio("页面", options=["Overview", "Pair Detail"], index=0)
        start_date = st.date_input("开始日期", value=default_start)
        end_date = st.date_input("结束日期", value=today)

        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            st.stop()

        window = int(st.slider("rolling window", min_value=60, max_value=504, value=252, step=12))
        entry = float(st.number_input("entry z", min_value=0.1, max_value=5.0, value=2.0, step=0.1))
        exit_ = float(st.number_input("exit z", min_value=0.0, max_value=3.0, value=0.5, step=0.1))
        fx_pair = st.selectbox("FX quote", options=["HKDCNY", "CNYHKD"], index=0)
        missing_threshold = float(
            st.slider("缺失率阈值", min_value=0.05, max_value=0.60, value=0.20, step=0.01)
        )
        st.markdown("**Executability Constraints**")
        enforce_a_share_t1 = bool(st.checkbox("enforce A-share T+1", value=True))
        allow_short_a = bool(st.checkbox("allow short A", value=False))
        allow_short_h = bool(st.checkbox("allow short H", value=True))

        st.markdown("---")
        score_threshold = float(
            st.slider("score threshold", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        )
        premium_percentile_threshold = float(
            st.slider(
                "premium percentile threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )
        )
        min_coverage_pct = float(
            st.slider("min coverage%", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        )
        max_gap_days_filter = int(
            st.number_input(
                "max gap days (filter)",
                min_value=0,
                max_value=260,
                value=260,
                step=1,
            )
        )
        min_data_quality_score = float(
            st.slider("min data quality score", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        )
        min_executability_score = float(
            st.slider(
                "min executability score",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
            )
        )
        only_low_quality = bool(st.checkbox("仅显示低质量（标红）", value=False))
        keyword = st.text_input("行业关键词（name/notes）", value="")

    return {
        "page": page,
        "start_date": start_date,
        "end_date": end_date,
        "window": window,
        "entry": entry,
        "exit": exit_,
        "fx_pair": fx_pair,
        "missing_threshold": missing_threshold,
        "enforce_a_share_t1": enforce_a_share_t1,
        "allow_short_a": allow_short_a,
        "allow_short_h": allow_short_h,
        "score_threshold": score_threshold,
        "premium_percentile_threshold": premium_percentile_threshold,
        "min_coverage_pct": min_coverage_pct,
        "max_gap_days_filter": max_gap_days_filter,
        "min_data_quality_score": min_data_quality_score,
        "min_executability_score": min_executability_score,
        "only_low_quality": only_low_quality,
        "keyword": keyword,
    }


if __name__ == "__main__":
    main()
