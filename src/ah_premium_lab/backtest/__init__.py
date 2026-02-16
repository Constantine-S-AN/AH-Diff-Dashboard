"""Backtesting module for cost-aware mean reversion strategy."""

from .costs import (
    CostParams,
    NetMetrics,
    apply_costs_to_daily,
    compute_net_metrics,
    generate_cost_grid,
)
from .executability import (
    ExecutabilityConfig,
    ExecutabilityMetrics,
    apply_executability_constraints,
    compute_executability_score,
)
from .pairs_strategy import PairsStrategyResult, run_pairs_strategy
from .sensitivity import (
    PairSensitivityOutput,
    estimate_breakeven_cost_level,
    estimate_breakeven_slippage,
    estimate_breakeven_total_cost,
    generate_sensitivity_html_report,
    run_pair_cost_sensitivity,
    run_universe_cost_sensitivity,
)
from .simulator import BacktestResult, BacktestSummary, simulate_mean_reversion

__all__ = [
    "BacktestResult",
    "BacktestSummary",
    "CostParams",
    "ExecutabilityConfig",
    "ExecutabilityMetrics",
    "NetMetrics",
    "PairSensitivityOutput",
    "PairsStrategyResult",
    "apply_costs_to_daily",
    "apply_executability_constraints",
    "compute_executability_score",
    "compute_net_metrics",
    "estimate_breakeven_slippage",
    "estimate_breakeven_total_cost",
    "estimate_breakeven_cost_level",
    "generate_cost_grid",
    "generate_sensitivity_html_report",
    "run_pair_cost_sensitivity",
    "run_pairs_strategy",
    "run_universe_cost_sensitivity",
    "simulate_mean_reversion",
]
