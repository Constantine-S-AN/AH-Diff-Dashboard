"""Transaction cost models and utility functions for sensitivity analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostParams:
    """Cost parameters in basis points (bps).

    Backward-compatible fields (`commission_*_bps`, `stamp_duty_*_bps`, `slippage_bps`)
    represent one-way coarse levels and are still accepted by existing code.

    Optional split fields allow explicit buy/sell-side modeling:
    - commission_{a,h}_{buy,sell}_bps
    - stamp_duty_{a,h}_{buy,sell}_bps
    - slippage_{a,h}_{buy,sell}_bps

    `borrow_bps` is an annualized placeholder for short-leg borrow/financing cost.
    """

    commission_a_bps: float
    commission_h_bps: float
    stamp_duty_a_bps: float
    stamp_duty_h_bps: float
    slippage_bps: float
    commission_a_buy_bps: float | None = None
    commission_a_sell_bps: float | None = None
    commission_h_buy_bps: float | None = None
    commission_h_sell_bps: float | None = None
    stamp_duty_a_buy_bps: float | None = None
    stamp_duty_a_sell_bps: float | None = None
    stamp_duty_h_buy_bps: float | None = None
    stamp_duty_h_sell_bps: float | None = None
    slippage_a_buy_bps: float | None = None
    slippage_a_sell_bps: float | None = None
    slippage_h_buy_bps: float | None = None
    slippage_h_sell_bps: float | None = None
    borrow_bps: float = 0.0

    @property
    def total_cost_level_bps(self) -> float:
        """Aggregate two-leg round-trip proxy in bps (includes borrow scenario level)."""

        return float(
            self.commission_total_bps
            + self.stamp_total_bps
            + self.slippage_total_bps
            + self.borrow_bps
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to plain dictionary."""

        data = asdict(self)
        data["commission_total_bps"] = self.commission_total_bps
        data["stamp_total_bps"] = self.stamp_total_bps
        data["slippage_total_bps"] = self.slippage_total_bps
        data["total_cost_level_bps"] = self.total_cost_level_bps
        return data

    @property
    def commission_a_buy_effective_bps(self) -> float:
        """Effective A buy-side commission bps."""

        if self.commission_a_buy_bps is None:
            return float(self.commission_a_bps)
        return float(self.commission_a_buy_bps)

    @property
    def commission_a_sell_effective_bps(self) -> float:
        """Effective A sell-side commission bps."""

        if self.commission_a_sell_bps is None:
            return float(self.commission_a_bps)
        return float(self.commission_a_sell_bps)

    @property
    def commission_h_buy_effective_bps(self) -> float:
        """Effective H buy-side commission bps."""

        if self.commission_h_buy_bps is None:
            return float(self.commission_h_bps)
        return float(self.commission_h_buy_bps)

    @property
    def commission_h_sell_effective_bps(self) -> float:
        """Effective H sell-side commission bps."""

        if self.commission_h_sell_bps is None:
            return float(self.commission_h_bps)
        return float(self.commission_h_sell_bps)

    @property
    def stamp_a_buy_effective_bps(self) -> float:
        """Effective A buy-side stamp duty bps."""

        if self.stamp_duty_a_buy_bps is None:
            return float(self.stamp_duty_a_bps)
        return float(self.stamp_duty_a_buy_bps)

    @property
    def stamp_a_sell_effective_bps(self) -> float:
        """Effective A sell-side stamp duty bps."""

        if self.stamp_duty_a_sell_bps is None:
            return float(self.stamp_duty_a_bps)
        return float(self.stamp_duty_a_sell_bps)

    @property
    def stamp_h_buy_effective_bps(self) -> float:
        """Effective H buy-side stamp duty bps."""

        if self.stamp_duty_h_buy_bps is None:
            return float(self.stamp_duty_h_bps)
        return float(self.stamp_duty_h_buy_bps)

    @property
    def stamp_h_sell_effective_bps(self) -> float:
        """Effective H sell-side stamp duty bps."""

        if self.stamp_duty_h_sell_bps is None:
            return float(self.stamp_duty_h_bps)
        return float(self.stamp_duty_h_sell_bps)

    @property
    def slippage_a_buy_effective_bps(self) -> float:
        """Effective A buy-side slippage bps."""

        if self.slippage_a_buy_bps is None:
            return float(self.slippage_bps)
        return float(self.slippage_a_buy_bps)

    @property
    def slippage_a_sell_effective_bps(self) -> float:
        """Effective A sell-side slippage bps."""

        if self.slippage_a_sell_bps is None:
            return float(self.slippage_bps)
        return float(self.slippage_a_sell_bps)

    @property
    def slippage_h_buy_effective_bps(self) -> float:
        """Effective H buy-side slippage bps."""

        if self.slippage_h_buy_bps is None:
            return float(self.slippage_bps)
        return float(self.slippage_h_buy_bps)

    @property
    def slippage_h_sell_effective_bps(self) -> float:
        """Effective H sell-side slippage bps."""

        if self.slippage_h_sell_bps is None:
            return float(self.slippage_bps)
        return float(self.slippage_h_sell_bps)

    @property
    def commission_total_bps(self) -> float:
        """Round-trip commission bps across A/H."""

        return float(
            self.commission_a_buy_effective_bps
            + self.commission_a_sell_effective_bps
            + self.commission_h_buy_effective_bps
            + self.commission_h_sell_effective_bps
        )

    @property
    def stamp_total_bps(self) -> float:
        """Round-trip stamp-duty bps across A/H."""

        return float(
            self.stamp_a_buy_effective_bps
            + self.stamp_a_sell_effective_bps
            + self.stamp_h_buy_effective_bps
            + self.stamp_h_sell_effective_bps
        )

    @property
    def slippage_total_bps(self) -> float:
        """Round-trip slippage bps across A/H."""

        return float(
            self.slippage_a_buy_effective_bps
            + self.slippage_a_sell_effective_bps
            + self.slippage_h_buy_effective_bps
            + self.slippage_h_sell_effective_bps
        )


@dataclass(frozen=True)
class NetMetrics:
    """Net performance metrics under one cost configuration."""

    net_cagr: float
    net_sharpe: float
    max_dd: float


def generate_cost_grid(
    commission_bps_levels: Sequence[float] = (0, 2, 4, 6, 8, 10),
    slippage_bps_levels: Sequence[float] = (0, 5, 10, 15, 20),
    stamp_duty_bps_levels: Sequence[float] = (0, 5, 10, 15),
    *,
    commission_h_bps_levels: Sequence[float] | None = None,
    stamp_duty_h_bps_levels: Sequence[float] | None = None,
    commission_a_buy_bps_levels: Sequence[float] | None = None,
    commission_a_sell_bps_levels: Sequence[float] | None = None,
    commission_h_buy_bps_levels: Sequence[float] | None = None,
    commission_h_sell_bps_levels: Sequence[float] | None = None,
    stamp_duty_a_buy_bps_levels: Sequence[float] | None = None,
    stamp_duty_a_sell_bps_levels: Sequence[float] | None = None,
    stamp_duty_h_buy_bps_levels: Sequence[float] | None = None,
    stamp_duty_h_sell_bps_levels: Sequence[float] | None = None,
    slippage_a_buy_bps_levels: Sequence[float] | None = None,
    slippage_a_sell_bps_levels: Sequence[float] | None = None,
    slippage_h_buy_bps_levels: Sequence[float] | None = None,
    slippage_h_sell_bps_levels: Sequence[float] | None = None,
    borrow_bps_levels: Sequence[float] = (0.0,),
) -> list[CostParams]:
    """Generate Cartesian-product cost parameter grid.

    By default, H-leg commission/stamp grids share A-leg levels.
    """

    if not commission_bps_levels or not slippage_bps_levels or not stamp_duty_bps_levels:
        raise ValueError("Cost level sequences cannot be empty")

    if commission_h_bps_levels is None:
        h_comm_levels = commission_bps_levels
    else:
        h_comm_levels = commission_h_bps_levels

    if stamp_duty_h_bps_levels is None:
        h_stamp_levels = stamp_duty_bps_levels
    else:
        h_stamp_levels = stamp_duty_h_bps_levels

    comm_a_pairs = _build_side_pairs(
        base_levels=commission_bps_levels,
        buy_levels=commission_a_buy_bps_levels,
        sell_levels=commission_a_sell_bps_levels,
    )
    comm_h_pairs = _build_side_pairs(
        base_levels=h_comm_levels,
        buy_levels=commission_h_buy_bps_levels,
        sell_levels=commission_h_sell_bps_levels,
    )
    stamp_a_pairs = _build_side_pairs(
        base_levels=stamp_duty_bps_levels,
        buy_levels=stamp_duty_a_buy_bps_levels,
        sell_levels=stamp_duty_a_sell_bps_levels,
    )
    stamp_h_pairs = _build_side_pairs(
        base_levels=h_stamp_levels,
        buy_levels=stamp_duty_h_buy_bps_levels,
        sell_levels=stamp_duty_h_sell_bps_levels,
    )
    slippage_levels = _build_slippage_side_levels(
        base_levels=slippage_bps_levels,
        a_buy_levels=slippage_a_buy_bps_levels,
        a_sell_levels=slippage_a_sell_bps_levels,
        h_buy_levels=slippage_h_buy_bps_levels,
        h_sell_levels=slippage_h_sell_bps_levels,
    )
    if not borrow_bps_levels:
        raise ValueError("borrow_bps_levels cannot be empty")

    grid: list[CostParams] = []
    for comm_a_pair, comm_h_pair, stamp_a_pair, stamp_h_pair, slippage_level, borrow_bps in product(
        comm_a_pairs,
        comm_h_pairs,
        stamp_a_pairs,
        stamp_h_pairs,
        slippage_levels,
        borrow_bps_levels,
    ):
        comm_a_buy, comm_a_sell = comm_a_pair
        comm_h_buy, comm_h_sell = comm_h_pair
        stamp_a_buy, stamp_a_sell = stamp_a_pair
        stamp_h_buy, stamp_h_sell = stamp_h_pair
        slip_a_buy, slip_a_sell, slip_h_buy, slip_h_sell = slippage_level

        comm_a_base = float((comm_a_buy + comm_a_sell) / 2.0)
        comm_h_base = float((comm_h_buy + comm_h_sell) / 2.0)
        stamp_a_base = float((stamp_a_buy + stamp_a_sell) / 2.0)
        stamp_h_base = float((stamp_h_buy + stamp_h_sell) / 2.0)
        slip_base = float(np.mean([slip_a_buy, slip_a_sell, slip_h_buy, slip_h_sell]))

        grid.append(
            CostParams(
                commission_a_bps=comm_a_base,
                commission_h_bps=comm_h_base,
                stamp_duty_a_bps=stamp_a_base,
                stamp_duty_h_bps=stamp_h_base,
                slippage_bps=slip_base,
                commission_a_buy_bps=float(comm_a_buy),
                commission_a_sell_bps=float(comm_a_sell),
                commission_h_buy_bps=float(comm_h_buy),
                commission_h_sell_bps=float(comm_h_sell),
                stamp_duty_a_buy_bps=float(stamp_a_buy),
                stamp_duty_a_sell_bps=float(stamp_a_sell),
                stamp_duty_h_buy_bps=float(stamp_h_buy),
                stamp_duty_h_sell_bps=float(stamp_h_sell),
                slippage_a_buy_bps=float(slip_a_buy),
                slippage_a_sell_bps=float(slip_a_sell),
                slippage_h_buy_bps=float(slip_h_buy),
                slippage_h_sell_bps=float(slip_h_sell),
                borrow_bps=float(borrow_bps),
            )
        )

    return grid


def apply_costs_to_daily(
    daily: pd.DataFrame,
    cost_params: CostParams,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """Apply cost model to one daily strategy table.

    Cost formulation (daily return space):
    - Trade costs use buy/sell turnover decomposition for A/H separately.
    - Each side supports `commission + stamp + slippage` bps.
    - Borrow/financing placeholder:
      `borrow_cost_ret = short_notional * borrow_bps / 10000 / annualization`.

    Turnover here is a notional turnover proxy from weight changes.
    """

    _require_columns(daily, ["pnl_gross", "a_weight", "h_weight"])

    out = daily.copy()
    prev_a = out["a_weight"].shift(1).fillna(0.0)
    prev_h = out["h_weight"].shift(1).fillna(0.0)

    delta_a = out["a_weight"] - prev_a
    delta_h = out["h_weight"] - prev_h

    out["turnover_a_buy"] = delta_a.clip(lower=0.0)
    out["turnover_a_sell"] = (-delta_a).clip(lower=0.0)
    out["turnover_h_buy"] = delta_h.clip(lower=0.0)
    out["turnover_h_sell"] = (-delta_h).clip(lower=0.0)
    out["turnover_a"] = out["turnover_a_buy"] + out["turnover_a_sell"]
    out["turnover_h"] = out["turnover_h_buy"] + out["turnover_h_sell"]
    out["turnover"] = out["turnover_a"] + out["turnover_h"]

    a_buy_bps = (
        cost_params.commission_a_buy_effective_bps
        + cost_params.stamp_a_buy_effective_bps
        + cost_params.slippage_a_buy_effective_bps
    )
    a_sell_bps = (
        cost_params.commission_a_sell_effective_bps
        + cost_params.stamp_a_sell_effective_bps
        + cost_params.slippage_a_sell_effective_bps
    )
    h_buy_bps = (
        cost_params.commission_h_buy_effective_bps
        + cost_params.stamp_h_buy_effective_bps
        + cost_params.slippage_h_buy_effective_bps
    )
    h_sell_bps = (
        cost_params.commission_h_sell_effective_bps
        + cost_params.stamp_h_sell_effective_bps
        + cost_params.slippage_h_sell_effective_bps
    )

    out["trade_cost_ret"] = (
        out["turnover_a_buy"] * (a_buy_bps / 10000.0)
        + out["turnover_a_sell"] * (a_sell_bps / 10000.0)
        + out["turnover_h_buy"] * (h_buy_bps / 10000.0)
        + out["turnover_h_sell"] * (h_sell_bps / 10000.0)
    )
    short_notional = (-prev_a).clip(lower=0.0) + (-prev_h).clip(lower=0.0)
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    out["borrow_cost_ret"] = short_notional * (cost_params.borrow_bps / 10000.0) / annualization

    out["cost_ret"] = out["trade_cost_ret"] + out["borrow_cost_ret"]
    out["pnl_net"] = out["pnl_gross"] - out["cost_ret"]
    out["curve_gross"] = (1.0 + out["pnl_gross"]).cumprod()
    out["curve_net"] = (1.0 + out["pnl_net"]).cumprod()

    out["drawdown_gross"] = out["curve_gross"] / out["curve_gross"].cummax() - 1.0
    out["drawdown_net"] = out["curve_net"] / out["curve_net"].cummax() - 1.0

    return out


def compute_net_metrics(daily: pd.DataFrame, annualization: int = 252) -> NetMetrics:
    """Compute net CAGR, net Sharpe (research), and max drawdown."""

    _require_columns(daily, ["pnl_net", "curve_net", "drawdown_net"])

    n_obs = int(daily.shape[0])
    if n_obs <= 1:
        return NetMetrics(net_cagr=float("nan"), net_sharpe=float("nan"), max_dd=float("nan"))

    ending_curve = float(daily["curve_net"].iloc[-1])
    if ending_curve <= 0.0:
        net_cagr = -1.0
    else:
        net_cagr = float(ending_curve ** (annualization / n_obs) - 1.0)

    pnl_net = daily["pnl_net"].astype(float)
    std = float(pnl_net.std(ddof=0))
    if std > 0.0:
        net_sharpe = float(np.sqrt(annualization) * pnl_net.mean() / std)
    else:
        net_sharpe = 0.0

    max_dd = float(daily["drawdown_net"].min())
    return NetMetrics(net_cagr=net_cagr, net_sharpe=net_sharpe, max_dd=max_dd)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Validate required columns."""

    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"DataFrame missing required columns: {', '.join(missing)}")


def _build_side_pairs(
    *,
    base_levels: Sequence[float],
    buy_levels: Sequence[float] | None,
    sell_levels: Sequence[float] | None,
) -> list[tuple[float, float]]:
    """Build buy/sell level pairs, preserving legacy one-way grid size by default."""

    if not base_levels:
        raise ValueError("base_levels cannot be empty")

    if buy_levels is None and sell_levels is None:
        return [(float(level), float(level)) for level in base_levels]

    if buy_levels is None and sell_levels is not None:
        return [(float(level), float(level)) for level in sell_levels]
    if sell_levels is None and buy_levels is not None:
        return [(float(level), float(level)) for level in buy_levels]

    assert buy_levels is not None and sell_levels is not None
    if not buy_levels or not sell_levels:
        raise ValueError("buy_levels/sell_levels cannot be empty when provided")

    return [
        (float(buy_level), float(sell_level))
        for buy_level, sell_level in product(buy_levels, sell_levels)
    ]


def _build_slippage_side_levels(
    *,
    base_levels: Sequence[float],
    a_buy_levels: Sequence[float] | None,
    a_sell_levels: Sequence[float] | None,
    h_buy_levels: Sequence[float] | None,
    h_sell_levels: Sequence[float] | None,
) -> list[tuple[float, float, float, float]]:
    """Build A/H buy/sell slippage tuples.

    Legacy mode (all overrides `None`) returns tied tuples to avoid
    changing default grid cardinality.
    """

    if not base_levels:
        raise ValueError("base_levels cannot be empty")

    if (
        a_buy_levels is None
        and a_sell_levels is None
        and h_buy_levels is None
        and h_sell_levels is None
    ):
        return [
            (float(level), float(level), float(level), float(level))
            for level in base_levels
        ]

    a_pairs = _build_side_pairs(
        base_levels=base_levels,
        buy_levels=a_buy_levels,
        sell_levels=a_sell_levels,
    )
    h_pairs = _build_side_pairs(
        base_levels=base_levels,
        buy_levels=h_buy_levels,
        sell_levels=h_sell_levels,
    )

    return [
        (float(a_buy), float(a_sell), float(h_buy), float(h_sell))
        for (a_buy, a_sell), (h_buy, h_sell) in product(a_pairs, h_pairs)
    ]
