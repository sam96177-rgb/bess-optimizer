"""Physical battery parameters, derived quantities, and validation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import BatteryConfig


@dataclass(frozen=True)
class Battery:
    power_mw: float
    energy_mwh: float
    round_trip_efficiency: float
    soc_min_fraction: float
    soc_max_fraction: float

    @property
    def eta(self) -> float:
        """One-way efficiency, applied on both charge and discharge sides
        so the round trip multiplies out to round_trip_efficiency."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_fraction * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_fraction * self.energy_mwh


def battery_from_config(cfg: BatteryConfig) -> Battery:
    return Battery(
        power_mw=cfg.power_mw,
        energy_mwh=cfg.energy_mwh,
        round_trip_efficiency=cfg.round_trip_efficiency,
        soc_min_fraction=cfg.soc_min_fraction,
        soc_max_fraction=cfg.soc_max_fraction,
    )
