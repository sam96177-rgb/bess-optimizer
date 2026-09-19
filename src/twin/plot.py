"""Twin SOC vs planned SOC for one day."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.model.battery import Battery  # noqa: E402


def plot_day(paths: pd.DataFrame, day: dt.date, battery: Battery, out_path: Path) -> Path:
    rows = paths[paths["solve_date"] == day]
    if rows.empty:
        raise ValueError(f"No twin paths stored for {day}")
    fig, ax = plt.subplots(figsize=(11, 4.5))

    first = rows[rows["draw"] == rows["draw"].min()].sort_values("target_time")
    ax.step(first["target_time"], first["planned_e_mwh"], where="post", color="black", lw=2, label="planned SOC")
    for draw, group in rows.groupby("draw"):
        group = group.sort_values("target_time")
        ax.step(group["target_time"], group["twin_e_mwh"], where="post", lw=1, alpha=0.8, label=f"twin draw {draw}")
        hit = group[group["violation"]]
        if not hit.empty:
            ax.scatter(hit["target_time"], hit["twin_e_mwh"], color="red", s=14, zorder=5)
    ax.axhline(battery.soc_min_mwh, color="grey", ls="--", lw=1)
    ax.axhline(battery.soc_max_mwh, color="grey", ls="--", lw=1, label="SOC window")
    ax.set_ylabel("SOC (MWh)")
    ax.set_title(f"Digital twin vs plan, {day}  (red = plan not deliverable; aFRR activation is a PLACEHOLDER assumption)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
