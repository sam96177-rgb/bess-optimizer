"""Config loaders and dataclasses for the BESS optimizer.

Every market timing constant, battery parameter, and assumption lives in
YAML under config/ (SPEC.md Hard Rule 4) — nothing here hardcodes a value,
it only loads, validates, and fails loudly on what it finds.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

PriceBasis = Literal["per_mwh_energy", "per_block_total", "per_mw_hour"]
_VALID_PRICE_BASES = {"per_mwh_energy", "per_block_total", "per_mw_hour"}
_REQUIRED_MARKETS = {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
DstConvention = Literal["variable_period_count", "fixed_period_count"]
_VALID_DST_CONVENTIONS = {"variable_period_count", "fixed_period_count"}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level of {path}, got {type(data)}")
    return data


_VALID_TERMINAL_VALUES = {"mean_decision_da_price", "none"}


@dataclass(frozen=True)
class BatteryConfig:
    power_nominal_mw: float
    power_derating_factor: float
    energy_mwh: float
    round_trip_efficiency: float
    soc_min_fraction: float
    soc_max_fraction: float
    aux_load_mw: float
    degradation_cost_eur_mwh: float
    terminal_value: str

    @property
    def power_mw(self) -> float:
        """Operational power limit: nominal rating after derating."""
        return self.power_nominal_mw * self.power_derating_factor

    def __post_init__(self) -> None:
        if self.power_nominal_mw <= 0:
            raise ValueError(f"power_nominal_mw must be > 0, got {self.power_nominal_mw}")
        if not (0 < self.power_derating_factor <= 1):
            raise ValueError(f"power_derating_factor must be in (0, 1], got {self.power_derating_factor}")
        if self.aux_load_mw < 0:
            raise ValueError(f"aux_load_mw must be >= 0, got {self.aux_load_mw}")
        if self.degradation_cost_eur_mwh < 0:
            raise ValueError(f"degradation_cost_eur_mwh must be >= 0, got {self.degradation_cost_eur_mwh}")
        if self.terminal_value not in _VALID_TERMINAL_VALUES:
            raise ValueError(
                f"terminal_value must be one of {sorted(_VALID_TERMINAL_VALUES)}, got '{self.terminal_value}'"
            )
        if self.energy_mwh <= 0:
            raise ValueError(f"energy_mwh must be > 0, got {self.energy_mwh}")
        if not (0 < self.round_trip_efficiency <= 1):
            raise ValueError(
                f"round_trip_efficiency must be in (0, 1], got {self.round_trip_efficiency}"
            )
        if not (0 <= self.soc_min_fraction < self.soc_max_fraction <= 1):
            raise ValueError(
                "Require 0 <= soc_min_fraction < soc_max_fraction <= 1, got "
                f"soc_min_fraction={self.soc_min_fraction}, soc_max_fraction={self.soc_max_fraction}"
            )


def load_battery_config(path: Path | None = None) -> BatteryConfig:
    path = path or (CONFIG_DIR / "battery.yaml")
    data = _load_yaml(path)
    required = {
        "power_nominal_mw", "power_derating_factor", "energy_mwh", "round_trip_efficiency",
        "soc_min_fraction", "soc_max_fraction", "aux_load_mw", "degradation_cost_eur_mwh", "terminal_value",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path} missing required keys: {sorted(missing)}")
    return BatteryConfig(
        power_nominal_mw=float(data["power_nominal_mw"]),
        power_derating_factor=float(data["power_derating_factor"]),
        energy_mwh=float(data["energy_mwh"]),
        round_trip_efficiency=float(data["round_trip_efficiency"]),
        soc_min_fraction=float(data["soc_min_fraction"]),
        soc_max_fraction=float(data["soc_max_fraction"]),
        aux_load_mw=float(data["aux_load_mw"]),
        degradation_cost_eur_mwh=float(data["degradation_cost_eur_mwh"]),
        terminal_value=str(data["terminal_value"]),
    )


@dataclass(frozen=True)
class AssumptionsConfig:
    alpha_pos: float
    alpha_neg: float
    activation_energy_price_eur_mwh: float
    fcr_reserve_hours: float
    afrr_reserve_hours: float

    def __post_init__(self) -> None:
        for name, value in (("alpha_pos", self.alpha_pos), ("alpha_neg", self.alpha_neg)):
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.fcr_reserve_hours < 0 or self.afrr_reserve_hours < 0:
            raise ValueError(
                "Reserve durations must be >= 0, got "
                f"fcr_reserve_hours={self.fcr_reserve_hours}, afrr_reserve_hours={self.afrr_reserve_hours}"
            )


def load_assumptions_config(path: Path | None = None) -> AssumptionsConfig:
    path = path or (CONFIG_DIR / "assumptions.yaml")
    data = _load_yaml(path)
    try:
        activation = data["afrr_activation"]
        reserve = data["reserve_energy_duration"]
    except KeyError as e:
        raise ValueError(f"{path} missing required top-level section: {e}") from e

    required_activation = {"alpha_pos", "alpha_neg", "activation_energy_price_eur_mwh"}
    missing = required_activation - activation.keys()
    if missing:
        raise ValueError(f"{path}::afrr_activation missing required keys: {sorted(missing)}")

    required_reserve = {"fcr_hours", "afrr_hours"}
    missing = required_reserve - reserve.keys()
    if missing:
        raise ValueError(f"{path}::reserve_energy_duration missing required keys: {sorted(missing)}")

    return AssumptionsConfig(
        alpha_pos=float(activation["alpha_pos"]),
        alpha_neg=float(activation["alpha_neg"]),
        activation_energy_price_eur_mwh=float(activation["activation_energy_price_eur_mwh"]),
        fcr_reserve_hours=float(reserve["fcr_hours"]),
        afrr_reserve_hours=float(reserve["afrr_hours"]),
    )


@dataclass(frozen=True)
class SolverConfig:
    name: str
    time_limit_seconds: float
    mip_gap: float
    include_activation_in_soc: bool
    enforce_exclusivity: bool
    cyclic_soc: bool


def load_solver_config(path: Path | None = None) -> SolverConfig:
    path = path or (CONFIG_DIR / "solver.yaml")
    data = _load_yaml(path)
    try:
        solver = data["solver"]
        flags = data["model_flags"]
    except KeyError as e:
        raise ValueError(f"{path} missing required top-level section: {e}") from e

    required_solver = {"name", "time_limit_seconds", "mip_gap"}
    missing = required_solver - solver.keys()
    if missing:
        raise ValueError(f"{path}::solver missing required keys: {sorted(missing)}")

    required_flags = {"include_activation_in_soc", "enforce_exclusivity", "cyclic_soc"}
    missing = required_flags - flags.keys()
    if missing:
        raise ValueError(f"{path}::model_flags missing required keys: {sorted(missing)}")

    return SolverConfig(
        name=str(solver["name"]),
        time_limit_seconds=float(solver["time_limit_seconds"]),
        mip_gap=float(solver["mip_gap"]),
        include_activation_in_soc=bool(flags["include_activation_in_soc"]),
        enforce_exclusivity=bool(flags["enforce_exclusivity"]),
        cyclic_soc=bool(flags["cyclic_soc"]),
    )


@dataclass(frozen=True)
class MarketPriceBasisConfig:
    basis_by_market: dict[str, PriceBasis]
    # MW increment a reserve bid must be a multiple of (PLAN_V4_REVISION.md
    # section 4). 0 or absent means unconstrained -- day_ahead has no
    # granularity entry in market_prices.yaml's older, flat-dict callers.
    bid_granularity_mw: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = _REQUIRED_MARKETS - self.basis_by_market.keys()
        if missing:
            raise ValueError(f"market_prices.yaml missing required markets: {sorted(missing)}")
        invalid = {
            market: basis
            for market, basis in self.basis_by_market.items()
            if basis not in _VALID_PRICE_BASES
        }
        if invalid:
            raise ValueError(
                f"Invalid price_basis values (must be one of {sorted(_VALID_PRICE_BASES)}): {invalid}"
            )
        negative = {m: g for m, g in self.bid_granularity_mw.items() if g < 0}
        if negative:
            raise ValueError(f"bid_granularity_mw must be >= 0: {negative}")

    def basis_for(self, market: str) -> PriceBasis:
        try:
            return self.basis_by_market[market]
        except KeyError as e:
            raise KeyError(f"No price_basis configured for market '{market}'") from e

    def granularity_for(self, market: str) -> float:
        return self.bid_granularity_mw.get(market, 0.0)


def load_market_price_basis_config(path: Path | None = None) -> MarketPriceBasisConfig:
    path = path or (CONFIG_DIR / "market_prices.yaml")
    data = _load_yaml(path)
    return MarketPriceBasisConfig(
        basis_by_market=data["price_basis"], bid_granularity_mw=data.get("bid_granularity_mw", {}),
    )


@dataclass(frozen=True)
class PathsConfig:
    forecast_repo_path: Path
    # PLAN_V5_REVISION.md V5-7: the forecast repo's own venv interpreter --
    # its dependencies (entsoe-py, lightgbm, ...) differ from this repo's,
    # so pipeline/01_ingest.py and 02_backtest_forecasts.py run it in a
    # subprocess with THIS python, cwd set to that repo. Optional: most
    # callers of load_paths_config() never touch it.
    forecast_venv_python: Path | None = None

    def __post_init__(self) -> None:
        if not self.forecast_repo_path.exists():
            raise FileNotFoundError(
                f"forecast_repo_path does not exist: {self.forecast_repo_path}. "
                "config/paths.yaml must point at the sibling bess-market-forecast checkout."
            )
        if self.forecast_venv_python is not None and not self.forecast_venv_python.exists():
            raise FileNotFoundError(
                f"forecast_venv_python does not exist: {self.forecast_venv_python}. "
                "config/paths.yaml must point at the sibling bess-market-forecast repo's own venv interpreter."
            )

    @property
    def markets_yaml_path(self) -> Path:
        return self.forecast_repo_path / "config" / "markets.yaml"

    @property
    def study_yaml_path(self) -> Path:
        return self.forecast_repo_path / "config" / "study.yaml"

    @property
    def results_dir(self) -> Path:
        return self.forecast_repo_path / "results"


def load_paths_config(path: Path | None = None) -> PathsConfig:
    path = path or (CONFIG_DIR / "paths.yaml")
    data = _load_yaml(path)
    if "forecast_repo_path" not in data:
        raise ValueError(f"{path} missing required key: forecast_repo_path")

    def _resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    resolved = _resolve(data["forecast_repo_path"])
    venv_python = _resolve(data["forecast_venv_python"]) if "forecast_venv_python" in data else None
    return PathsConfig(forecast_repo_path=resolved, forecast_venv_python=venv_python)


@dataclass(frozen=True)
class ForecastBlendsConfig:
    """Named per-market model blends (PLAN_V4_REVISION.md section 3):
    forecast_model stays a flat scalar everywhere (batch resume, skip-log
    scanning, export, every dashboard page filter on it), so a blend is a
    name that src.data.forecast_loader.load_raw_y_pred resolves into one
    real model per market, not a dict passed around instead of the scalar.
    """

    blends: dict[str, dict[str, str]]

    def __post_init__(self) -> None:
        for name, per_market in self.blends.items():
            missing = _REQUIRED_MARKETS - per_market.keys()
            if missing:
                raise ValueError(f"forecast_blends.yaml blend '{name}' missing required markets: {sorted(missing)}")

    def resolve(self, name: str) -> dict[str, str] | None:
        """None if `name` is an ordinary (non-blend) model name."""
        return self.blends.get(name)


def load_forecast_blends_config(path: Path | None = None) -> ForecastBlendsConfig:
    path = path or (CONFIG_DIR / "forecast_blends.yaml")
    data = _load_yaml(path)
    return ForecastBlendsConfig(blends=data)


@dataclass(frozen=True)
class StudyWindow:
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"study window end {self.end} is before start {self.start}")


@dataclass(frozen=True)
class StudyConfig:
    """PLAN_V5_REVISION.md V5-6: the forecast repo's config/study.yaml,
    read here via forecast_repo_path -- same direction as
    load_market_timing_configs. Only this repo reads the `optimizer:`
    block; the forecast repo ignores it."""

    name: str
    test_window: dict[str, StudyWindow]
    models: list[str]
    strategies: list[str]
    blends: list[str]
    history_start: dt.date | None = None
    min_training_history_days: int | None = None
    # PLAN_V6_REVISION.md V6-4: the 60 days right before each market's
    # test_window start, read by pipeline/03_select_blend.py to pick
    # per_market_best on validation-only MAE. None for a study.yaml
    # predating V6-4.
    validation_window: dict[str, StudyWindow] | None = None

    @property
    def optimization_window(self) -> tuple[dt.date, dt.date]:
        """Intersection of every market's test window -- the range every
        market actually has test-window data for, not restated."""
        start = max(w.start for w in self.test_window.values())
        end = min(w.end for w in self.test_window.values())
        if end < start:
            raise ValueError(
                f"study '{self.name}': market test windows don't overlap "
                f"(intersection would be {start}..{end})"
            )
        return start, end

    @property
    def optimizer_models(self) -> list[str]:
        return self.models + self.blends


def load_study_config(paths_config: PathsConfig | None = None) -> StudyConfig:
    paths_config = paths_config or load_paths_config()
    path = paths_config.study_yaml_path
    data = _load_yaml(path)

    required = {"name", "test_window", "models", "optimizer"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path} missing required keys: {sorted(missing)}")

    test_window = {
        market: StudyWindow(start=window["start"], end=window["end"])
        for market, window in data["test_window"].items()
    }
    validation_window = None
    if "validation_window" in data:
        validation_window = {
            market: StudyWindow(start=window["start"], end=window["end"])
            for market, window in data["validation_window"].items()
        }
    optimizer = data["optimizer"]
    return StudyConfig(
        name=str(data["name"]),
        test_window=test_window,
        models=list(data["models"]),
        strategies=list(optimizer.get("strategies", [])),
        blends=list(optimizer.get("blends", [])),
        history_start=data["history_start"] if "history_start" in data else None,
        min_training_history_days=data.get("min_training_history_days"),
        validation_window=validation_window,
    )


@dataclass(frozen=True)
class MarketTimingConfig:
    """The subset of a market's bess-market-forecast markets.yaml entry
    that grid construction needs. Parsed live from that repo's own file
    (never duplicated as separate hardcoded numbers) — SPEC.md §1.

    gate_closure_local/gate_closure_offset_days are display-only (the
    PLAN_V5_1_REVISION.md dashboard overview's markets table); grid
    construction has no notion of a bidding deadline.
    """

    name: str
    timezone: str
    dst_convention: DstConvention
    product_length_hours: float
    delivery_start_local: dt.time
    gate_closure_local: dt.time | None = None
    gate_closure_offset_days: int | None = None

    def __post_init__(self) -> None:
        if self.dst_convention not in _VALID_DST_CONVENTIONS:
            raise ValueError(
                f"{self.name}: invalid dst_convention '{self.dst_convention}', "
                f"must be one of {sorted(_VALID_DST_CONVENTIONS)}"
            )
        if self.product_length_hours <= 0:
            raise ValueError(f"{self.name}: product_length_hours must be > 0, got {self.product_length_hours}")


def load_market_timing_configs(paths_config: PathsConfig | None = None) -> dict[str, MarketTimingConfig]:
    """Reads the sibling bess-market-forecast repo's config/markets.yaml
    directly (read-only) and returns a MarketTimingConfig per market.
    """
    paths_config = paths_config or load_paths_config()
    path = paths_config.markets_yaml_path
    data = _load_yaml(path)

    missing_markets = _REQUIRED_MARKETS - data.keys()
    if missing_markets:
        raise ValueError(f"{path} missing required markets: {sorted(missing_markets)}")

    required_fields = {"timezone", "dst_convention", "product_length_hours", "delivery_start_local"}
    configs: dict[str, MarketTimingConfig] = {}
    for market in _REQUIRED_MARKETS:
        entry = data[market]
        missing_fields = required_fields - entry.keys()
        if missing_fields:
            raise ValueError(f"{path}::{market} missing required keys: {sorted(missing_fields)}")
        gate_closure_local = entry.get("gate_closure_local")
        configs[market] = MarketTimingConfig(
            name=market,
            timezone=str(entry["timezone"]),
            dst_convention=entry["dst_convention"],
            product_length_hours=float(entry["product_length_hours"]),
            delivery_start_local=dt.datetime.strptime(str(entry["delivery_start_local"]), "%H:%M").time(),
            gate_closure_local=dt.datetime.strptime(str(gate_closure_local), "%H:%M").time() if gate_closure_local else None,
            gate_closure_offset_days=entry.get("gate_closure_offset_days"),
        )

    day_ahead = configs["day_ahead"]
    if day_ahead.dst_convention != "variable_period_count":
        raise ValueError(
            f"{path}: expected day_ahead.dst_convention == 'variable_period_count', "
            f"got '{day_ahead.dst_convention}'. SPEC.md's grid logic assumes this; stop and "
            "re-check rather than silently building on a changed convention."
        )

    ancillary = [configs[m] for m in ("fcr", "afrr_pos", "afrr_neg")]
    first = ancillary[0]
    for cfg in ancillary[1:]:
        if cfg.dst_convention != first.dst_convention or cfg.product_length_hours != first.product_length_hours:
            raise ValueError(
                "fcr/afrr_pos/afrr_neg are expected to share one block structure (same "
                f"dst_convention and product_length_hours), but {first.name} has "
                f"({first.dst_convention}, {first.product_length_hours}h) while {cfg.name} has "
                f"({cfg.dst_convention}, {cfg.product_length_hours}h). The model's shared block "
                "index b assumes these match — stop and resolve this before building the grid."
            )
    if first.dst_convention != "fixed_period_count":
        raise ValueError(
            f"{path}: expected fcr/afrr dst_convention == 'fixed_period_count', got "
            f"'{first.dst_convention}'."
        )

    return configs
