"""Configuration models and loader utilities."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level configuration."""

    name: str
    timezone: str


@dataclass(frozen=True)
class PairConfig:
    """A/H stock pair configuration."""

    pair_id: str
    a_symbol: str
    h_symbol: str
    fx_symbol: str


@dataclass(frozen=True)
class DataConfig:
    """Data source and symbol universe configuration."""

    source: str
    start_date: str
    end_date: str
    frequency: str
    seed: int
    pairs: list[PairConfig]


@dataclass(frozen=True)
class CoreConfig:
    """Core premium calculation configuration."""

    premium_method: str
    zscore_window: int


@dataclass(frozen=True)
class StatsConfig:
    """Statistical testing configuration."""

    adf_max_lag: int
    adf_alpha: float
    coint_trend: str


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest engine configuration."""

    initial_capital: float
    entry_z: float
    exit_z: float
    max_holding_days: int
    annualization: int


@dataclass(frozen=True)
class ReportConfig:
    """Reporting configuration."""

    output_dir: str
    cost_grid_bps: list[float]
    include_plotly: bool


@dataclass(frozen=True)
class LabConfig:
    """Top-level project configuration object."""

    project: ProjectConfig
    data: DataConfig
    core: CoreConfig
    stats: StatsConfig
    backtest: BacktestConfig
    report: ReportConfig


RawConfigDict = dict[str, Any]


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve repository root by locating `pyproject.toml`.

    Args:
        start: Optional starting path.

    Returns:
        Path to project root if found, otherwise current working directory.
    """

    candidate = (start or Path(__file__)).resolve()
    search_nodes = [candidate, *candidate.parents]
    for node in search_nodes:
        if node.is_dir() and (node / "pyproject.toml").exists():
            return node
        if node.is_file() and (node.parent / "pyproject.toml").exists():
            return node.parent
    return Path.cwd()


def load_config(path: str | Path) -> LabConfig:
    """Load project configuration from YAML or TOML.

    Args:
        path: Config file path.

    Returns:
        Parsed and typed configuration object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If suffix is unsupported.
        KeyError: If required keys are missing.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_raw_config(config_path)
    return _parse_config(raw)


def _load_raw_config(path: Path) -> RawConfigDict:
    """Load raw dictionary from a supported config file."""

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config must parse to a dictionary: {path}")
        return loaded
    if suffix == ".toml":
        with path.open("rb") as fh:
            loaded = tomllib.load(fh)
        if not isinstance(loaded, dict):
            raise ValueError(f"TOML config must parse to a dictionary: {path}")
        return loaded
    raise ValueError(f"Unsupported config format: {path.suffix}")


def _parse_config(raw: RawConfigDict) -> LabConfig:
    """Parse raw config dictionary into `LabConfig`."""

    project_raw = _required_dict(raw, "project")
    data_raw = _required_dict(raw, "data")
    core_raw = _required_dict(raw, "core")
    stats_raw = _required_dict(raw, "stats")
    backtest_raw = _required_dict(raw, "backtest")
    report_raw = _required_dict(raw, "report")

    pairs_raw = data_raw.get("pairs")
    if not isinstance(pairs_raw, list):
        raise KeyError("data.pairs must be a list")

    pairs = [
        PairConfig(
            pair_id=str(_required_value(item, "pair_id")),
            a_symbol=str(_required_value(item, "a_symbol")),
            h_symbol=str(_required_value(item, "h_symbol")),
            fx_symbol=str(_required_value(item, "fx_symbol")),
        )
        for item in pairs_raw
    ]

    return LabConfig(
        project=ProjectConfig(
            name=str(_required_value(project_raw, "name")),
            timezone=str(_required_value(project_raw, "timezone")),
        ),
        data=DataConfig(
            source=str(_required_value(data_raw, "source")),
            start_date=str(_required_value(data_raw, "start_date")),
            end_date=str(_required_value(data_raw, "end_date")),
            frequency=str(_required_value(data_raw, "frequency")),
            seed=int(_required_value(data_raw, "seed")),
            pairs=pairs,
        ),
        core=CoreConfig(
            premium_method=str(_required_value(core_raw, "premium_method")),
            zscore_window=int(_required_value(core_raw, "zscore_window")),
        ),
        stats=StatsConfig(
            adf_max_lag=int(_required_value(stats_raw, "adf_max_lag")),
            adf_alpha=float(_required_value(stats_raw, "adf_alpha")),
            coint_trend=str(_required_value(stats_raw, "coint_trend")),
        ),
        backtest=BacktestConfig(
            initial_capital=float(_required_value(backtest_raw, "initial_capital")),
            entry_z=float(_required_value(backtest_raw, "entry_z")),
            exit_z=float(_required_value(backtest_raw, "exit_z")),
            max_holding_days=int(_required_value(backtest_raw, "max_holding_days")),
            annualization=int(_required_value(backtest_raw, "annualization")),
        ),
        report=ReportConfig(
            output_dir=str(_required_value(report_raw, "output_dir")),
            cost_grid_bps=[float(x) for x in _required_list(report_raw, "cost_grid_bps")],
            include_plotly=bool(_required_value(report_raw, "include_plotly")),
        ),
    )


def _required_dict(raw: RawConfigDict, key: str) -> RawConfigDict:
    """Fetch a required dictionary field."""

    value = raw.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing or invalid config section: {key}")
    return value


def _required_list(raw: RawConfigDict, key: str) -> list[Any]:
    """Fetch a required list field."""

    value = raw.get(key)
    if not isinstance(value, list):
        raise KeyError(f"Missing or invalid config list: {key}")
    return value


def _required_value(raw: RawConfigDict, key: str) -> Any:
    """Fetch a required scalar field."""

    if key not in raw:
        raise KeyError(f"Missing required config key: {key}")
    return raw[key]
