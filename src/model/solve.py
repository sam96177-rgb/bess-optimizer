"""Solves a built linopy model, extracts the solution into plain pandas
objects, and fails loudly (with whatever diagnostic linopy/HiGHS gives)
on anything other than an optimal solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import linopy
import pandas as pd

from src.config import SolverConfig


class InfeasibleModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class SolveResult:
    objective_value: float
    p_ch: pd.Series
    p_dis: pd.Series
    u: pd.Series
    e: pd.Series
    c_fcr: pd.Series | None
    c_pos: pd.Series | None
    c_neg: pd.Series | None


def solve_model(model: linopy.Model, solver_config: SolverConfig) -> SolveResult:
    status, condition = model.solve(
        solver_name=solver_config.name,
        time_limit=solver_config.time_limit_seconds,
        mip_rel_gap=solver_config.mip_gap,
    )
    if status != "ok" or condition != "optimal":
        raise InfeasibleModelError(
            f"Solve did not reach an optimal solution: status={status!r}, condition={condition!r}. "
            "Refusing to extract a solution from a non-optimal solve."
        )

    variables = model.variables
    has_ancillary = "c_fcr" in variables

    def _series(name: str) -> pd.Series:
        return variables[name].solution.to_series()

    return SolveResult(
        objective_value=float(model.objective.value),
        p_ch=_series("p_ch"),
        p_dis=_series("p_dis"),
        u=_series("u"),
        e=_series("e"),
        c_fcr=_series("c_fcr") if has_ancillary else None,
        c_pos=_series("c_pos") if has_ancillary else None,
        c_neg=_series("c_neg") if has_ancillary else None,
    )
