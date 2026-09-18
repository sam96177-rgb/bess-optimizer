"""Standalone worker: solves exactly one day's model, then exits.

rolling.py's run_rolling dispatches each day here via subprocess instead
of calling build_model/solve_model in-process. This was added after
observing that a single long-running process solving ~150+ days in a row
degraded from a measured ~4.5s/day model-build cost to 50-120+s/day as
the run progressed, even though each individual HiGHS solve stayed under
0.1s -- consistent with a memory/state accumulation issue in linopy or
highspy across many sequential Model() instantiations in one process,
not a problem with the model itself. Running each day in its own fresh
process sidesteps that regardless of its root cause.
"""
from __future__ import annotations

import pickle
import sys

from src.model.build import build_model
from src.model.solve import solve_model


def main() -> None:
    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path, "rb") as f:
        task = pickle.load(f)

    model = build_model(
        steps=task["day_prices"].grid.steps,
        day_ahead_prices=task["day_prices"].day_ahead,
        battery=task["battery"],
        delta_t_hours=task["day_prices"].grid.delta_t_hours,
        cyclic_soc=task["cyclic_soc"],
        enforce_exclusivity=task["enforce_exclusivity"],
        initial_soc_mwh=task["initial_soc_mwh"],
        ancillary=task["ancillary"],
    )
    result = solve_model(model, task["solver_config"])

    with open(output_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
