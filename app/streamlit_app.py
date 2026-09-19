"""Entry point: sets page config and wires up the three dashboard pages.
Each page reads only results/<solve_id>/ sidecars and the sibling
forecast repo's results/*.parquet (see app/data_loader.py) -- this file
contains no data logic of its own.
"""
import streamlit as st

st.set_page_config(page_title="BESS Optimizer Dashboard", layout="wide")

pages = [
    st.Page("pages/dispatch_explorer.py", title="Dispatch Explorer", icon="🔋", default=True),
    st.Page("pages/forecast_error_cost.py", title="Forecast Error Cost", icon="📉"),
    st.Page("pages/run_catalogue.py", title="Run Catalogue", icon="🗂️"),
]

nav = st.navigation(pages)
nav.run()
