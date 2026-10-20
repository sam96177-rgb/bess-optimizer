"""PLAN_V5_1_REVISION.md section 2: app/ui.py's styled_table renders the
report's table markup (right-aligned mono numbers, the selected row tinted).
Run through streamlit.testing.v1.AppTest since st.html needs a script context."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import load_study_config

APP_DIR = Path(__file__).resolve().parents[1] / "app"
CURRENT_STUDY = load_study_config().name

_SCRIPT = f"""
import sys
sys.path.insert(0, {str(APP_DIR)!r})
import pandas as pd
import ui

df = pd.DataFrame({{"net_eur": [100.0, 200.0]}}, index=["seasonal_naive", "linear_lags"])
ui.styled_table(df, formats={{"net_eur": "{{:,.0f}}"}}, highlight_row="linear_lags", align_right=["net_eur"])
"""


def _render_table_html() -> str:
    at = AppTest.from_string(_SCRIPT).run()
    assert not at.exception, [e.value for e in at.exception]
    html_blocks = at.get("html")
    assert html_blocks, "styled_table did not emit any st.html block"
    return html_blocks[-1].value


def test_styled_table_right_aligns_the_numeric_column():
    html = _render_table_html()
    assert '<th class="r">net_eur</th>' in html
    assert '<td class="r">100</td>' in html
    assert '<td class="r">200</td>' in html


def test_styled_table_highlights_the_selected_row():
    html = _render_table_html()
    assert '<tr class="hi">' in html
    assert "linear_lags" in html


_STUDY_FILTER_SCRIPT = f"""
import sys
sys.path.insert(0, {str(APP_DIR)!r})
import pandas as pd
import streamlit as st
import ui

catalogue = pd.DataFrame({{"study": [{CURRENT_STUDY!r}, None, {CURRENT_STUDY!r}], "solve_id": ["a", "b", "c"]}})
filtered = ui.study_filter(catalogue)
st.write(len(filtered))
"""


def test_study_filter_defaults_to_the_current_study_and_hides_pre_v5_runs():
    at = AppTest.from_string(_STUDY_FILTER_SCRIPT).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.selectbox[0].options == sorted(["pre-v5", CURRENT_STUDY])
    assert at.sidebar.selectbox[0].value == CURRENT_STUDY
    assert at.get("markdown")[-1].value == "`2`"  # only the two rows tagged with the current study

    at.sidebar.selectbox[0].set_value("pre-v5").run()
    assert at.get("markdown")[-1].value == "`1`"  # only the study=None row
