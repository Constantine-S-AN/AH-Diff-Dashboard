"""Rolling cointegration diagnostics for stability analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ah_premium_lab.stats.diagnostics import engle_granger_test


@dataclass(frozen=True)
class RollingStabilityMetrics:
    """Stability summary built from rolling Engle-Granger windows."""

    n_windows: int
    p_value_pass_rate: float
    beta_variance: float
    resid_std_drift: float
    stability_score: float


def rolling_engle_granger(
    logA: pd.Series,
    logHfx: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """Run rolling Engle-Granger diagnostics.

    Args:
        logA: Log A series.
        logHfx: Log H-in-CNY series.
        window: Window length.
        step: Step size between adjacent windows.

    Returns:
        DataFrame columns:
        - window_start
        - window_end
        - p_value
        - beta
        - resid_std
        - n_obs
    """

    if window < 20:
        raise ValueError("window must be >= 20")
    if step <= 0:
        raise ValueError("step must be > 0")

    aligned = pd.concat([logA.rename("logA"), logHfx.rename("logHfx")], axis=1).dropna()
    if aligned.empty or aligned.shape[0] < window:
        return pd.DataFrame(
            columns=["window_start", "window_end", "p_value", "beta", "resid_std", "n_obs"],
        )

    rows: list[dict[str, object]] = []
    end_positions = list(range(window, aligned.shape[0] + 1, step))
    if end_positions[-1] != aligned.shape[0]:
        end_positions.append(aligned.shape[0])

    for end_pos in end_positions:
        sub = aligned.iloc[end_pos - window : end_pos]
        p_value, beta, resid = engle_granger_test(sub["logA"], sub["logHfx"])
        resid_std = float(resid.std(ddof=0)) if not resid.empty else float("nan")

        rows.append(
            {
                "window_start": sub.index[0],
                "window_end": sub.index[-1],
                "p_value": float(p_value),
                "beta": float(beta),
                "resid_std": float(resid_std),
                "n_obs": int(sub.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def rolling_stability_metrics(
    rolling_result: pd.DataFrame,
    *,
    p_value_threshold: float = 0.05,
) -> RollingStabilityMetrics:
    """Compute stability metrics from rolling Engle-Granger outputs.

    Metrics:
    - p_value_pass_rate: proportion of windows with p_value < threshold
    - beta_variance: variance of rolling beta
    - resid_std_drift: relative drift from first to last resid_std
    """

    if rolling_result.empty:
        return RollingStabilityMetrics(
            n_windows=0,
            p_value_pass_rate=float("nan"),
            beta_variance=float("nan"),
            resid_std_drift=float("nan"),
            stability_score=0.0,
        )

    frame = rolling_result.copy()

    p_values = frame["p_value"].astype(float)
    beta = frame["beta"].astype(float)
    resid_std = frame["resid_std"].astype(float)

    p_valid = p_values[np.isfinite(p_values)]
    beta_valid = beta[np.isfinite(beta)]
    resid_valid = resid_std[np.isfinite(resid_std)]

    p_value_pass_rate = (
        float((p_valid < p_value_threshold).mean()) if not p_valid.empty else float("nan")
    )
    beta_variance = float(beta_valid.var(ddof=0)) if not beta_valid.empty else float("nan")

    if resid_valid.shape[0] >= 2 and abs(float(resid_valid.iloc[0])) > 1e-12:
        first_std = float(resid_valid.iloc[0])
        last_std = float(resid_valid.iloc[-1])
        resid_std_drift = float(
            (last_std - first_std) / abs(first_std),
        )
    else:
        resid_std_drift = float("nan")

    pass_component = p_value_pass_rate if np.isfinite(p_value_pass_rate) else 0.0
    beta_component = float(np.exp(-beta_variance)) if np.isfinite(beta_variance) else 0.0
    drift_component = float(np.exp(-abs(resid_std_drift))) if np.isfinite(resid_std_drift) else 0.0
    stability_score = float(
        100.0 * (0.5 * pass_component + 0.3 * beta_component + 0.2 * drift_component)
    )

    return RollingStabilityMetrics(
        n_windows=int(frame.shape[0]),
        p_value_pass_rate=p_value_pass_rate,
        beta_variance=beta_variance,
        resid_std_drift=resid_std_drift,
        stability_score=stability_score,
    )
