"""PLAN_V5_1_REVISION.md section 2: shared Streamlit-only look for the
dashboard, matching the study report
(https://claude.ai/artifact/UUTiDztG73g33b89BEiKWX). Colors and fonts for
Streamlit's own widgets come from .streamlit/config.toml; this module adds
only the report's table/card/bar-chart CSS, which config.toml has no knob
for. analysis.py stays pure pandas -- every function here renders directly
with `st.*` and returns None, except where noted.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loader import load_model_info  # noqa: E402
from src.config import load_study_config  # noqa: E402

# Same values as the report's :root (light) and its dark-mode override --
# chosen here in Python (via st.context.theme.type) rather than a CSS media
# query, because Streamlit's own theme choice (config.toml or the viewer's
# in-app override) doesn't otherwise reach a plain @media(prefers-color-scheme)
# rule.
_LIGHT = dict(
    rule="#d8dfe4", muted="#566573", accent="#0d6b86", accent_soft="#e3f0f4",
    amber="#9a6412", amber_soft="#fbf0dd", bar_old="#b8c5cf", bar_new="#0d6b86",
    good="#22794b", paper="#ffffff", ink="#18202b",
)
_DARK = dict(
    rule="#2a3742", muted="#98a9b6", accent="#56b7d4", accent_soft="#16303a",
    amber="#e3a94f", amber_soft="#2e2414", bar_old="#3f4f5c", bar_new="#56b7d4",
    good="#5cc68b", paper="#161f27", ink="#e4ebf0",
)
_DISPLAY_FONT = '"IBM Plex Sans Condensed","Arial Narrow",Arial,sans-serif'
_MONO_FONT = '"IBM Plex Mono",Consolas,"Courier New",monospace'


def inject_theme() -> None:
    """Call once per page. Adds the report's table/card/bar CSS as variables
    under :root, picking the light or dark value set from the viewer's
    active Streamlit theme."""
    theme_type = getattr(st.context.theme, "type", None) or "light"
    palette = _DARK if theme_type == "dark" else _LIGHT
    st.html(f"""
<style>
:root {{
  --bessr-rule: {palette['rule']};
  --bessr-muted: {palette['muted']};
  --bessr-accent: {palette['accent']};
  --bessr-accent-soft: {palette['accent_soft']};
  --bessr-amber: {palette['amber']};
  --bessr-amber-soft: {palette['amber_soft']};
  --bessr-bar-old: {palette['bar_old']};
  --bessr-bar-new: {palette['bar_new']};
  --bessr-good: {palette['good']};
  --bessr-paper: {palette['paper']};
  --bessr-ink: {palette['ink']};
  --bessr-display: {_DISPLAY_FONT};
  --bessr-mono: {_MONO_FONT};
}}
.bessr-tablewrap {{ overflow-x: auto; }}
table.bessr-table {{ border-collapse: collapse; width: 100%; font-size: .92rem; font-variant-numeric: tabular-nums; }}
table.bessr-table th, table.bessr-table td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--bessr-rule); vertical-align: top; }}
table.bessr-table th {{ font-family: var(--bessr-display); font-weight: 600; font-size: .9rem; color: var(--bessr-muted); }}
table.bessr-table td.r, table.bessr-table th.r {{ text-align: right; font-family: var(--bessr-mono); font-size: .86rem; }}
table.bessr-table tr.hi td {{ background: var(--bessr-accent-soft); }}
.bessr-model {{ border: 1px solid var(--bessr-rule); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 8px; background: var(--bessr-paper); }}
.bessr-model h4 {{ margin: 0; font-family: var(--bessr-display); font-size: 1.1rem; }}
.bessr-model p {{ font-size: .94rem; margin: 0; }}
.bessr-model dl {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 0; font-size: .86rem; }}
.bessr-model dt {{ color: var(--bessr-muted); }}
.bessr-model dd {{ margin: 0; }}
.bessr-model .bessr-dropped {{ color: var(--bessr-muted); font-size: .82rem; margin-top: 2px; }}
.bessr-tag {{ display: inline-block; font-family: var(--bessr-mono); font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; padding: 2px 7px; border-radius: 999px; background: var(--bessr-accent-soft); color: var(--bessr-accent); align-self: flex-start; }}
.bessr-bars {{ display: grid; grid-template-columns: 140px 1fr; gap: 6px 12px; align-items: center; }}
.bessr-bars .lbl {{ font-family: var(--bessr-mono); font-size: .8rem; }}
.bessr-pair {{ display: flex; flex-direction: column; gap: 3px; }}
.bessr-bar {{ height: 12px; border-radius: 2px; position: relative; display: flex; align-items: center; }}
.bessr-bar.old {{ background: var(--bessr-bar-old); }}
.bessr-bar.new {{ background: var(--bessr-bar-new); }}
.bessr-bar i {{ position: absolute; left: calc(100% + 6px); font-style: normal; font-family: var(--bessr-mono); font-size: .72rem; color: var(--bessr-ink); white-space: nowrap; }}
.bessr-track {{ padding-right: 52px; }}
.bessr-axis {{ display: grid; grid-template-columns: 140px 1fr; gap: 12px; }}
.bessr-ticks {{ position: relative; height: 16px; margin-right: 52px; border-top: 1px solid var(--bessr-rule); }}
.bessr-ticks span {{ position: absolute; top: 2px; transform: translateX(-50%); font-family: var(--bessr-mono); font-size: .68rem; color: var(--bessr-muted); }}
.bessr-legend {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: .84rem; color: var(--bessr-muted); margin-bottom: 8px; }}
.bessr-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.bessr-sw {{ width: 14px; height: 10px; border-radius: 2px; display: inline-block; }}
.bessr-note {{ border-left: 3px solid var(--bessr-amber); background: var(--bessr-amber-soft); padding: 12px 14px; font-size: .95rem; }}
.bessr-note b {{ color: var(--bessr-amber); }}
.bessr-key {{ border-left: 3px solid var(--bessr-accent); background: var(--bessr-accent-soft); padding: 12px 14px; }}
</style>
""")


AGREEMENT_FORMATS = {
    "charge_timing_pct": "{:.0f}%", "charge_amount_pct": "{:.0f}%", "charge_mwh_per_day": "{:.1f}",
    "discharge_timing_pct": "{:.0f}%", "discharge_amount_pct": "{:.0f}%", "discharge_mwh_per_day": "{:.1f}",
    "wrong_way_qh_per_day": "{:.1f}",
    "net_eur": "{:,.0f}", "capture_pct": "{:.1f}%", "value_vs_baseline_eur": "{:+,.0f}",
}
AGREEMENT_CAPTION = (
    "**Timing** = share of the model's charge (or discharge) quarter-hours where the ceiling plan also "
    "charges (or discharges) within 30 minutes. **Amount** = 100% minus the daily MWh gap to the ceiling: "
    "ceiling charges 14 MWh, model charges 17, gap 3, amount = 1 - 3/14 = 79%. "
    "**Wrong way** = quarter-hours per day where the model does the opposite of the ceiling."
)


def _format_cell(value, fmt) -> str:
    if pd.isna(value):
        return "n/a"
    if callable(fmt):
        return fmt(value)
    if fmt:
        return fmt.format(value)
    return html.escape(str(value))


def styled_table(
    df: pd.DataFrame,
    formats: dict[str, str] | None = None,
    highlight_row=None,
    align_right: list[str] | None = None,
) -> None:
    """Render `df` as the report's table: hairline rows, right-aligned mono
    numbers, the `highlight_row` index label tinted. `formats` maps column
    name to a str.format spec (e.g. "{:,.0f}") or a one-arg callable.
    `align_right` defaults to every column (the common case: an all-numeric
    table next to a label index)."""
    formats = formats or {}
    align = set(df.columns) if align_right is None else set(align_right)

    header = "<th></th>" + "".join(
        f'<th class="r">{html.escape(str(c))}</th>' if c in align else f"<th>{html.escape(str(c))}</th>"
        for c in df.columns
    )
    body_rows = []
    for idx, row in df.iterrows():
        row_class = ' class="hi"' if highlight_row is not None and idx == highlight_row else ""
        cells = f"<td>{html.escape(str(idx))}</td>"
        for c in df.columns:
            cell_class = ' class="r"' if c in align else ""
            cells += f"<td{cell_class}>{_format_cell(row[c], formats.get(c))}</td>"
        body_rows.append(f"<tr{row_class}>{cells}</tr>")

    st.html(
        '<div class="bessr-tablewrap"><table class="bessr-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def model_card(name: str) -> None:
    """The intro card for one forecast name, from app/model_info.yaml."""
    card = load_model_info().get(name)
    if card is None:
        st.caption(f"No description yet for `{name}`.")
        return
    dropped_note = '<p class="bessr-dropped">Dropped from the current study.</p>' if card.get("dropped") else ""
    st.html(
        '<div class="bessr-model">'
        f'<span class="bessr-tag">{html.escape(card["kind"])}</span>'
        f"<h4><code>{html.escape(name)}</code></h4>"
        f'<p>{html.escape(card["summary"])}</p>'
        "<dl>"
        f'<dt>Strength</dt><dd>{html.escape(card["strength"])}</dd>'
        f'<dt>Weakness</dt><dd>{html.escape(card["weakness"])}</dd>'
        f'<dt>Learns</dt><dd>{html.escape(card["learns"])}</dd>'
        "</dl>"
        f"{dropped_note}"
        "</div>"
    )


def model_picker(models: list[str], key: str) -> str | None:
    """`st.pills` over `models`; shows the picked one's model_card. Returns
    the pick (None if nothing is selected)."""
    picked = st.pills("Model", models, default=models[0] if models else None, key=key)
    if picked is not None:
        model_card(picked)
    return picked


def study_filter(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Sidebar study selector, defaulting to the current config/study.yaml
    name. Runs with no `study` field in run_meta.json (pre-v5) are grouped
    under "pre-v5". Every page calls this before its own filters."""
    if catalogue.empty:
        return catalogue
    study_col = catalogue["study"] if "study" in catalogue.columns else pd.Series(index=catalogue.index, dtype=object)
    labels = study_col.fillna("pre-v5")
    options = sorted(labels.unique())
    current = load_study_config().name
    default_index = options.index(current) if current in options else 0
    choice = st.sidebar.selectbox("Study", options, index=default_index, key="bessr_study_filter")
    return catalogue[labels == choice].reset_index(drop=True)


def capture_bars(v_old: dict[str, float], v_new: dict[str, float], title: str | None = None) -> None:
    """The report's paired bar chart: one row per key present in both
    `v_old` and `v_new` (percentages 0-100), old faded behind new."""
    rows = []
    for model, new_pct in v_new.items():
        if model not in v_old:
            continue
        old_pct = v_old[model]
        rows.append(
            f'<span class="lbl">{html.escape(model)}</span>'
            '<div class="bessr-track bessr-pair">'
            f'<div class="bessr-bar old" style="width:{min(old_pct, 100):.1f}%"><i>{old_pct:.1f}</i></div>'
            f'<div class="bessr-bar new" style="width:{min(new_pct, 100):.1f}%"><i>{new_pct:.1f}</i></div>'
            "</div>"
        )
    if not rows:
        st.info("No model has a capture figure on both sides of this comparison.")
        return
    heading = f'<h4 style="font-family:var(--bessr-display);font-size:1rem;margin:0 0 4px 0;">{html.escape(title)}</h4>' if title else ""
    ticks = "".join(f'<span style="left:{p}%">{p}%</span>' for p in (0, 25, 50, 75, 100))
    st.html(
        '<div class="bessr-legend">'
        '<span><i class="bessr-sw" style="background:var(--bessr-bar-old)"></i>Previous study</span>'
        '<span><i class="bessr-sw" style="background:var(--bessr-bar-new)"></i>Current study</span>'
        "</div>"
        f"{heading}"
        f'<div class="bessr-bars">{"".join(rows)}</div>'
        f'<div class="bessr-axis"><span></span><div class="bessr-ticks">{ticks}</div></div>'
    )
