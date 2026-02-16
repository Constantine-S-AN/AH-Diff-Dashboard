"""Statistical diagnostics for A/H premium research."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ah_premium_lab.config import StatsConfig


@dataclass(frozen=True)
class PairDiagnostics:
    """Summary of pair-level statistical diagnostics."""

    adf_pvalue: float
    adf_stat: float
    adf_used_lags: int
    half_life_days: float
    coint_pvalue: float
    coint_stat: float
    coint_used_lags: int
    coint_beta: float
    zscore_stability: float
    research_score: float
    is_stationary: bool
    is_cointegrated: bool

    def to_dict(self) -> dict[str, float | bool | int]:
        """Convert diagnostics to plain dictionary."""

        return asdict(self)


def adf_test(
    series: pd.Series,
    max_lag: int | None = None,
) -> tuple[float, float, int]:
    """Run Augmented Dickey-Fuller test.

    Args:
        series: Input time series.
        max_lag: Optional maximum lag for ADF.

    Returns:
        Tuple of `(p_value, test_stat, used_lags)`.
    """

    clean = series.dropna()
    if clean.shape[0] < 20:
        return float("nan"), float("nan"), 0

    test_stat, p_value, used_lags, *_ = adfuller(clean, maxlag=max_lag, autolag="AIC")
    return float(p_value), float(test_stat), int(used_lags)


def engle_granger_test(
    log_A: pd.Series,
    log_H_fx: pd.Series,
    max_lag: int | None = None,
) -> tuple[float, float, pd.Series]:
    """Run Engle-Granger test via OLS hedge regression + ADF on residual.

    Args:
        log_A: Log A-share series.
        log_H_fx: Log H-share (FX-adjusted) series.
        max_lag: Optional maximum lag for residual ADF.

    Returns:
        Tuple `(p_value, beta, resid_series)` where:
        - `beta` is hedge ratio from `log_A = alpha + beta * log_H_fx + resid`
        - `p_value` is ADF p-value on residual series.
    """

    aligned = pd.concat([log_A.rename("log_A"), log_H_fx.rename("log_H_fx")], axis=1).dropna()
    if aligned.shape[0] < 20:
        return float("nan"), float("nan"), pd.Series(dtype=float)

    x = aligned["log_H_fx"].to_numpy()
    y = aligned["log_A"].to_numpy()
    beta, intercept = np.polyfit(x, y, 1)

    resid_values = y - (intercept + beta * x)
    resid_series = pd.Series(resid_values, index=aligned.index, name="eg_residual")

    p_value, _, _ = adf_test(resid_series, max_lag=max_lag)
    return float(p_value), float(beta), resid_series


def half_life_ar1(series: pd.Series) -> float:
    """Estimate mean-reversion half-life in days with AR(1) approximation.

    We fit: `Δx_t = k * x_{t-1} + ε_t`, and use `half_life = -ln(2)/k`.
    """

    clean = series.dropna()
    if clean.shape[0] < 3:
        return float("inf")

    lagged = clean.shift(1)
    delta = clean - lagged
    aligned = pd.concat([lagged.rename("x"), delta.rename("y")], axis=1).dropna()

    if aligned.shape[0] < 3:
        return float("inf")

    k, _ = np.polyfit(aligned["x"].to_numpy(), aligned["y"].to_numpy(), 1)
    if k >= 0.0:
        return float("inf")

    return float(-np.log(2.0) / k)


def summary_score(
    adf_p_value: float,
    eg_p_value: float,
    half_life_days: float,
    zscore_series: pd.Series,
) -> float:
    """Compute a 0-100 research score from stationarity/co-integration signals.

    Formula:
    - `adf_component = 100 * (1 - clamp(adf_p_value, 0, 1))`
    - `eg_component = 100 * (1 - clamp(eg_p_value, 0, 1))`
    - `half_life_component = 100 * exp(-abs(log(half_life_days / 20)))`
      (targeting ~20-day medium-speed mean reversion; invalid/inf -> 0)
    - `zscore_stability_component`:
      `100 * (0.4*exp(-|mean(z)|) + 0.4*exp(-|std(z)-1|) + 0.2*exp(-8*tail_rate))`
      where `tail_rate = mean(|z| > 3)`

    Final weighted score:
    - `score = 0.35*adf_component + 0.35*eg_component`
      `+ 0.20*half_life_component + 0.10*zscore_stability_component`

    Returns:
        Research score clipped to `[0, 100]`.
    """

    adf_component = 100.0 * (1.0 - _clip01(adf_p_value))
    eg_component = 100.0 * (1.0 - _clip01(eg_p_value))
    half_life_component = _half_life_component(half_life_days)
    z_component = _zscore_stability_component(zscore_series)

    score = (
        0.35 * adf_component + 0.35 * eg_component + 0.20 * half_life_component + 0.10 * z_component
    )
    return float(np.clip(score, 0.0, 100.0))


def estimate_half_life(series: pd.Series) -> float:
    """Backward-compatible alias of `half_life_ar1`."""

    return half_life_ar1(series)


def run_pair_diagnostics(frame: pd.DataFrame, config: StatsConfig) -> PairDiagnostics:
    """Run diagnostics on one prepared pair frame (compatibility wrapper)."""

    _require_columns(frame, ["premium", "a_close", "h_close"])

    adf_pvalue, adf_stat, adf_used_lags = adf_test(frame["premium"], max_lag=config.adf_max_lag)
    half_life_days = half_life_ar1(frame["premium"])

    log_A = np.log(frame["a_close"].astype(float))
    if "fx_rate" in frame.columns:
        log_H_fx = np.log(frame["h_close"].astype(float) / frame["fx_rate"].astype(float))
    else:
        log_H_fx = np.log(frame["h_close"].astype(float))

    coint_pvalue, coint_beta, resid = engle_granger_test(
        log_A=log_A,
        log_H_fx=log_H_fx,
        max_lag=config.adf_max_lag,
    )
    coint_pvalue_for_stat, coint_stat, coint_used_lags = adf_test(resid, max_lag=config.adf_max_lag)

    # Prefer the p-value recomputed alongside coint_stat to keep tuple internally consistent.
    if np.isfinite(coint_pvalue_for_stat):
        resolved_coint_pvalue = coint_pvalue_for_stat
    else:
        resolved_coint_pvalue = coint_pvalue

    if "zscore" in frame.columns:
        z_series = frame["zscore"]
    else:
        prem = frame["premium"].astype(float)
        prem_std = float(prem.std(ddof=0))
        if prem_std > 0.0:
            z_series = (prem - prem.mean()) / prem_std
        else:
            z_series = pd.Series(index=frame.index, dtype=float)

    zscore_stability = _zscore_stability_component(z_series)
    research_score = summary_score(
        adf_p_value=adf_pvalue,
        eg_p_value=resolved_coint_pvalue,
        half_life_days=half_life_days,
        zscore_series=z_series,
    )

    alpha = config.adf_alpha
    return PairDiagnostics(
        adf_pvalue=adf_pvalue,
        adf_stat=adf_stat,
        adf_used_lags=adf_used_lags,
        half_life_days=half_life_days,
        coint_pvalue=resolved_coint_pvalue,
        coint_stat=coint_stat,
        coint_used_lags=coint_used_lags,
        coint_beta=coint_beta,
        zscore_stability=zscore_stability,
        research_score=research_score,
        is_stationary=bool(np.isfinite(adf_pvalue) and adf_pvalue < alpha),
        is_cointegrated=bool(np.isfinite(resolved_coint_pvalue) and resolved_coint_pvalue < alpha),
    )


def _half_life_component(half_life_days: float) -> float:
    """Convert half-life into a [0, 100] component score."""

    if not np.isfinite(half_life_days) or half_life_days <= 0.0:
        return 0.0

    component = 100.0 * np.exp(-abs(np.log(half_life_days / 20.0)))
    return float(np.clip(component, 0.0, 100.0))


def _zscore_stability_component(zscore_series: pd.Series) -> float:
    """Compute z-score stability component in [0, 100]."""

    z = zscore_series.dropna().astype(float)
    if z.shape[0] < 30:
        return 50.0

    mean_component = float(np.exp(-abs(float(z.mean()))))
    std_component = float(np.exp(-abs(float(z.std(ddof=0)) - 1.0)))
    tail_rate = float((np.abs(z) > 3.0).mean())
    tail_component = float(np.exp(-8.0 * tail_rate))

    score = 100.0 * (0.4 * mean_component + 0.4 * std_component + 0.2 * tail_component)
    return float(np.clip(score, 0.0, 100.0))


def _clip01(value: float) -> float:
    """Clip scalar into [0, 1], handling non-finite inputs."""

    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Validate DataFrame required columns."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {', '.join(missing)}")
