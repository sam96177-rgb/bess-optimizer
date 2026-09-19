"""Gate-ordered solving of one delivery day (PLAN_V2_DISPATCH.md Phase 2).

On D-1 the markets clear in this order: FCR (~08:00), aFRR (~09:00),
day-ahead (12:00). Each stage is a full model solve, but only its own
market's decision is committed, and it sees only what exists at its gate:

  1. FCR stage:  prices = reserve-stage view (DA proxy, since the real DA
                 forecast has an 11:00 cutoff). Commit c_fcr.
  2. aFRR stage: same view, c_fcr fixed. Commit c_pos, c_neg.
  3. DA stage:   the real day-ahead forecast, all reserves fixed. Commit
                 the energy dispatch (p_ch, p_dis, u, e).

If every stage saw identical prices this would reproduce the simultaneous
optimum exactly; any revenue gap versus the simultaneous model comes purely
from the earlier stages deciding with less information.
"""
from __future__ import annotations

import pandas as pd

from src.config import SolverConfig
from src.model.battery import Battery
from src.model.build import AncillaryInputs, FixedReserves, build_model
from src.model.solve import SolveResult, solve_model


def solve_day_sequential(
    steps: pd.DatetimeIndex,
    reserve_stage_day_ahead: pd.Series,
    final_day_ahead: pd.Series,
    battery: Battery,
    delta_t_hours: float,
    solver_config: SolverConfig,
    ancillary: AncillaryInputs,
    cyclic_soc: bool = True,
    enforce_exclusivity: bool = True,
    initial_soc_mwh: float | None = None,
) -> SolveResult:
    def _solve(day_ahead: pd.Series, fixed: FixedReserves | None) -> SolveResult:
        model = build_model(
            steps=steps,
            day_ahead_prices=day_ahead,
            battery=battery,
            delta_t_hours=delta_t_hours,
            cyclic_soc=cyclic_soc,
            enforce_exclusivity=enforce_exclusivity,
            initial_soc_mwh=initial_soc_mwh,
            ancillary=ancillary,
            fixed_reserves=fixed,
        )
        return solve_model(model, solver_config)

    fcr_stage = _solve(reserve_stage_day_ahead, None)
    afrr_stage = _solve(reserve_stage_day_ahead, FixedReserves(fcr=fcr_stage.c_fcr))
    return _solve(
        final_day_ahead,
        FixedReserves(fcr=fcr_stage.c_fcr, pos=afrr_stage.c_pos, neg=afrr_stage.c_neg),
    )
