"""Digital twin: replays a solved run's planned schedule step by step under
the battery's real physics and random aFRR activation, and logs where the
plan cannot be delivered (PLAN_V2_DISPATCH.md Phase 3).

What the planner assumes vs what the twin does
  - The MILP treats aFRR activation as its *expected* energy (alpha * c * dt
    every step). The twin instead draws each step's activation at random: a
    step is called upward with probability alpha_pos / m, downward with
    probability alpha_neg / m (never both), and a called step delivers a
    random fraction xi ~ Beta of the committed capacity with mean m
    (call_degree_mean). The mean equals the plan's expectation for any m, so
    any deviation is variance the plan could not see. m = 1 recovers
    "always the full committed capacity"; smaller m means more frequent,
    smaller calls, and a much gentler SOC random walk.
  - The twin enforces the SOC window. Planned dispatch and aux load come
    first; activation only gets what SOC room is left. What cannot be
    delivered is logged as a shortfall, and SOC is clipped to the window.

ASSUMPTIONS (flagged, not measurements)
  - alpha_pos / alpha_neg: config/assumptions.yaml PLACEHOLDERS.
  - Activation draws are independent per step; real calls are bursty and
    the TSO pushes them back toward zero.
  - call_degree_mean / call_degree_concentration (TwinConfig) are
    PLACEHOLDERS for the size distribution of a call.
  - FCR activation energy is assumed to net to zero (as in the planner).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AssumptionsConfig
from src.data.grid import DayGrid
from src.model.battery import Battery

EPS = 1e-9
MODES = ("stochastic", "expected")


@dataclass(frozen=True)
class TwinConfig:
    n_draws: int = 200
    seed: int = 0
    mode: str = "stochastic"  # "expected" uses alpha * c every step: must reproduce the plan exactly
    n_path_draws: int = 3  # draws whose full step-by-step path is kept
    call_degree_mean: float = 0.25  # PLACEHOLDER: mean fraction of committed capacity delivered in a called step
    call_degree_concentration: float = 4.0  # PLACEHOLDER: Beta(m*k, (1-m)*k); larger = less spread around m

    def __post_init__(self) -> None:
        if not (0 < self.call_degree_mean <= 1):
            raise ValueError(f"call_degree_mean must be in (0, 1], got {self.call_degree_mean}")
        if self.call_degree_concentration <= 0:
            raise ValueError(f"call_degree_concentration must be > 0, got {self.call_degree_concentration}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{self.mode}'")
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {self.n_draws}")


@dataclass(frozen=True)
class DayPlan:
    """One day's planned schedule, indexed by the day's steps / blocks."""

    day: dt.date
    grid: DayGrid
    p_ch: pd.Series
    p_dis: pd.Series
    e: pd.Series
    c_fcr: pd.Series | None = None
    c_pos: pd.Series | None = None
    c_neg: pd.Series | None = None


def _per_step(series: pd.Series | None, grid: DayGrid) -> np.ndarray:
    if series is None:
        return np.zeros(len(grid.steps))
    return np.array([float(series.loc[b]) for b in grid.step_block_index])


def replay_day(
    plan: DayPlan,
    initial_soc_mwh: float,
    battery: Battery,
    assumptions: AssumptionsConfig,
    cfg: TwinConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary, paths): one summary row per draw, and step-by-step
    paths for the first cfg.n_path_draws draws."""
    m = cfg.call_degree_mean
    q_pos, q_neg = assumptions.alpha_pos / m, assumptions.alpha_neg / m
    if q_pos + q_neg > 1:
        raise ValueError(
            f"alpha_pos + alpha_neg must be <= call_degree_mean (a step is called in at most one direction, "
            f"and a called step delivers on average {m} of the committed capacity), got "
            f"{assumptions.alpha_pos} + {assumptions.alpha_neg} vs {m}"
        )
    grid = plan.grid
    steps = grid.steps
    n_steps, n_draws = len(steps), cfg.n_draws
    dt_h = grid.delta_t_hours
    eta = battery.eta
    soc_min, soc_max = battery.soc_min_mwh, battery.soc_max_mwh

    p_ch = plan.p_ch.reindex(steps).to_numpy(dtype=float)
    p_dis = plan.p_dis.reindex(steps).to_numpy(dtype=float)
    planned_e = plan.e.reindex(steps).to_numpy(dtype=float)
    c_fcr, c_pos, c_neg = (_per_step(s, grid) for s in (plan.c_fcr, plan.c_pos, plan.c_neg))

    if cfg.mode == "expected":
        up_frac = np.full((n_steps, n_draws), assumptions.alpha_pos)
        down_frac = np.full((n_steps, n_draws), assumptions.alpha_neg)
    else:
        u = rng.random((n_steps, n_draws))
        degree = (
            np.ones((n_steps, n_draws)) if m >= 1
            else rng.beta(m * cfg.call_degree_concentration, (1 - m) * cfg.call_degree_concentration, (n_steps, n_draws))
        )
        up_frac = (u < q_pos) * degree
        down_frac = ((u >= q_pos) & (u < q_pos + q_neg)) * degree
    a_up = up_frac * c_pos[:, None]  # MW called from the battery
    a_down = down_frac * c_neg[:, None]  # MW absorbed by the battery

    e = np.full(n_draws, float(initial_soc_mwh))
    e_path = np.empty((n_steps, n_draws))
    dispatch_short = np.zeros(n_draws)  # MWh of planned dispatch/aux the battery could not honour
    up_short = np.zeros(n_draws)
    down_short = np.zeros(n_draws)
    up_delivered = np.zeros(n_draws)
    down_delivered = np.zeros(n_draws)
    soc_violation_steps = np.zeros(n_draws, dtype=int)
    power_violation_steps = np.zeros(n_draws, dtype=int)
    step_violation = np.zeros((n_steps, n_draws), dtype=bool)

    for t in range(n_steps):
        base = e + eta * p_ch[t] * dt_h - p_dis[t] * dt_h / eta - battery.aux_load_mw * dt_h
        low = np.maximum(soc_min - base, 0.0)
        base = np.maximum(base, soc_min)
        high = np.maximum(base - soc_max, 0.0)
        base = np.minimum(base, soc_max)
        dispatch_short_t = low * eta + high / eta

        need_up = a_up[t] * dt_h / eta
        got_up = np.minimum(need_up, np.maximum(base - soc_min, 0.0))
        base = base - got_up
        need_down = eta * a_down[t] * dt_h
        got_down = np.minimum(need_down, np.maximum(soc_max - base, 0.0))
        e = base + got_down

        up_short_t = (need_up - got_up) * eta
        down_short_t = (need_down - got_down) / eta
        violated = (dispatch_short_t > EPS) | (up_short_t > EPS) | (down_short_t > EPS)

        dispatch_short += dispatch_short_t
        up_short += up_short_t
        down_short += down_short_t
        up_delivered += got_up * eta
        down_delivered += got_down / eta
        soc_violation_steps += violated
        step_violation[t] = violated
        power_violation_steps += (
            (p_dis[t] + c_fcr[t] + a_up[t] > battery.power_mw + 1e-6)
            | (p_ch[t] + c_fcr[t] + a_down[t] > battery.power_mw + 1e-6)
        )
        e_path[t] = e

    price = assumptions.activation_energy_price_eur_mwh
    expected_activation_mwh = float(
        ((assumptions.alpha_pos * c_pos + assumptions.alpha_neg * c_neg) * dt_h).sum()
    )
    summary = pd.DataFrame(
        {
            "solve_date": plan.day,
            "draw": np.arange(n_draws),
            "n_soc_violation_steps": soc_violation_steps,
            "n_power_violation_steps": power_violation_steps,
            "dispatch_shortfall_mwh": dispatch_short,
            "activation_shortfall_up_mwh": up_short,
            "activation_shortfall_down_mwh": down_short,
            "activation_delivered_mwh": up_delivered + down_delivered,
            "activation_expected_mwh": expected_activation_mwh,
            "min_soc_mwh": e_path.min(axis=0),
            "max_soc_mwh": e_path.max(axis=0),
            "max_abs_soc_deviation_mwh": np.abs(e_path - planned_e[:, None]).max(axis=0),
            "end_soc_deviation_mwh": e_path[-1] - planned_e[-1],
            "activation_revenue_eur": (up_delivered + down_delivered) * price,
            "activation_is_assumption": True,
        }
    )

    k = min(cfg.n_path_draws, n_draws)
    paths = pd.DataFrame(
        {
            "solve_date": plan.day,
            "draw": np.repeat(np.arange(k), n_steps),
            "target_time": np.tile(steps, k),
            "planned_e_mwh": np.tile(planned_e, k),
            "twin_e_mwh": e_path[:, :k].T.reshape(-1),
            "activation_up_mw": a_up[:, :k].T.reshape(-1),
            "activation_down_mw": a_down[:, :k].T.reshape(-1),
            "violation": step_violation[:, :k].T.reshape(-1),
        }
    )
    return summary, paths


def battery_from_meta(meta: dict) -> Battery:
    b = dict(meta["battery"])
    b.pop("power_mw", None)  # derived from nominal x derating
    try:
        return Battery(**b)
    except TypeError as exc:
        raise ValueError(
            f"run_meta battery block {sorted(b)} does not match the current Battery fields; "
            "this run predates the v2 battery model and cannot be replayed."
        ) from exc


def assumptions_from_meta(meta: dict) -> AssumptionsConfig:
    return AssumptionsConfig(**meta["assumptions"])


def _day_plan(day: dt.date, rows: pd.DataFrame, grid: DayGrid) -> DayPlan:
    def step_series(variable: str) -> pd.Series:
        s = rows[rows["variable"] == variable].set_index("target_time")["value"]
        return s.reindex(grid.steps)

    def block_series(variable: str) -> pd.Series | None:
        s = rows[rows["variable"] == variable]
        if s.empty:
            return None
        return s.set_index("block_id")["value"].astype(float)

    plan = DayPlan(
        day=day, grid=grid, p_ch=step_series("p_ch"), p_dis=step_series("p_dis"), e=step_series("e"),
        c_fcr=block_series("c_fcr"), c_pos=block_series("c_pos"), c_neg=block_series("c_neg"),
    )
    for name in ("p_ch", "p_dis", "e"):
        if getattr(plan, name).isna().any():
            raise ValueError(f"{day}: planned '{name}' does not cover every step of the day's grid")
    return plan


def replay_run(
    run_dir: Path,
    grid_for_day,
    cfg: TwinConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Replays every solved day of one results/<solve_id>/ run.

    `grid_for_day(day) -> DayGrid`. Each day starts from the plan's own
    initial SOC (cyclic runs: the day's planned end SOC; chained runs: the
    previous day's planned end, soc_min for the first), so days are replayed
    independently and deviations do not accumulate across midnight.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    battery = battery_from_meta(meta)
    assumptions = assumptions_from_meta(meta)
    cyclic = bool(meta["model_flags"]["cyclic_soc"])
    long_format = pd.read_parquet(run_dir / "long_format.parquet")

    summaries, paths = [], []
    previous_end = battery.soc_min_mwh
    for day, rows in long_format.groupby("solve_date", sort=True):
        plan = _day_plan(day, rows, grid_for_day(day))
        initial = float(plan.e.iloc[-1]) if cyclic else previous_end
        previous_end = float(plan.e.iloc[-1])
        # Per-day stream from (seed, day ordinal): a day's draws do not depend on which other days are replayed.
        rng = np.random.default_rng([cfg.seed, day.toordinal()])
        s, p = replay_day(plan, initial, battery, assumptions, cfg, rng)
        summaries.append(s)
        paths.append(p)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    path_df = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    twin_meta = {
        "solve_id": meta["solve_id"],
        "mode": cfg.mode,
        "n_draws": cfg.n_draws,
        "seed": cfg.seed,
        "alpha_pos": assumptions.alpha_pos,
        "alpha_neg": assumptions.alpha_neg,
        "activation_energy_price_eur_mwh": assumptions.activation_energy_price_eur_mwh,
        "call_degree_mean": cfg.call_degree_mean,
        "call_degree_concentration": cfg.call_degree_concentration,
        "assumptions": [
            "alpha_pos/alpha_neg are PLACEHOLDERS (config/assumptions.yaml)",
            "call_degree_mean/concentration (size of a call) are PLACEHOLDERS",
            "activation calls are independent per step (no burstiness or mean reversion)",
            "FCR activation energy assumed to net to zero",
        ],
        "n_days": int(summary["solve_date"].nunique()) if not summary.empty else 0,
    }
    return summary, path_df, twin_meta


def write_twin_outputs(run_dir: Path, summary: pd.DataFrame, paths: pd.DataFrame, twin_meta: dict) -> None:
    run_dir = Path(run_dir)
    summary.to_parquet(run_dir / "twin_summary.parquet", index=False)
    paths.to_parquet(run_dir / "twin_paths.parquet", index=False)
    (run_dir / "twin_meta.json").write_text(json.dumps(twin_meta, indent=2), encoding="utf-8")  # written last
