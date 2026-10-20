"""Strategies evaluated over the same historical day set, so they're
directly comparable (SPEC.md Milestone 6):

1. day_ahead_only    -- pure day-ahead arbitrage, no ancillary markets.
2. full_cooptimized  -- day-ahead, FCR and aFRR solved jointly.
3. full_sequential   -- gate-ordered clearing (FCR -> aFRR -> DA). Kept and
   tested, but not part of the default batch or the dashboard.

All three are the same MILP core (rolling.run_rolling); they differ only in
the flags below.
"""
from __future__ import annotations

from src.model.rolling import RollingRunResult, run_rolling

STRATEGY_FLAGS = {
    "day_ahead_only": {"include_ancillary": False},
    "full_cooptimized": {"include_ancillary": True},
    "full_sequential": {"include_ancillary": True, "sequential": True},
}
DEFAULT_STRATEGIES = ["day_ahead_only", "full_cooptimized"]


def run_strategy(name: str, **kwargs) -> RollingRunResult:
    if name not in STRATEGY_FLAGS:
        raise ValueError(f"Unknown strategy '{name}'. Must be one of {list(STRATEGY_FLAGS)}")
    return run_rolling(**kwargs, **STRATEGY_FLAGS[name])


def run_all_benchmarks(strategies: list[str] = DEFAULT_STRATEGIES, **kwargs) -> dict[str, RollingRunResult]:
    return {name: run_strategy(name, **kwargs) for name in strategies}
