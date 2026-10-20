"""Entry point: sets page config and wires up the dashboard pages.
Each page reads only results/<solve_id>/ sidecars and the sibling
forecast repo's results/*.parquet (see app/data_loader.py) -- this file
contains no data logic of its own.
"""
import streamlit as st

from ui import inject_theme

st.set_page_config(page_title="BESS Optimizer Dashboard", layout="wide")
inject_theme()

pages = [
    st.Page("pages/study_overview.py", title="Study Overview", icon="📋", default=True),
    st.Page("pages/forecast_comparison.py", title="Forecast Comparison", icon="📐"),
    st.Page("pages/dispatch_explorer.py", title="Dispatch Explorer", icon="🔋"),
]

nav = st.navigation(pages)
nav.run()
