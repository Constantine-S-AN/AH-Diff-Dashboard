"""Statistical diagnostics for A/H premium series."""

from .breaks import BreakDetectionResult, cusum_stability_test, detect_structural_breaks
from .diagnostics import (
    PairDiagnostics,
    adf_test,
    engle_granger_test,
    estimate_half_life,
    half_life_ar1,
    run_pair_diagnostics,
    summary_score,
)
from .rolling import RollingStabilityMetrics, rolling_engle_granger, rolling_stability_metrics

__all__ = [
    "BreakDetectionResult",
    "PairDiagnostics",
    "RollingStabilityMetrics",
    "adf_test",
    "cusum_stability_test",
    "detect_structural_breaks",
    "estimate_half_life",
    "half_life_ar1",
    "engle_granger_test",
    "rolling_engle_granger",
    "rolling_stability_metrics",
    "run_pair_diagnostics",
    "summary_score",
]
