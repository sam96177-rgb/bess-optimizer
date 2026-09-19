"""Every dashboard page must render without raising, whatever is (or isn't)
in results/ -- an empty results dir just shows each page's own warning."""
import glob
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP_DIR / "streamlit_app.py"] + sorted((APP_DIR / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_all_expected_pages_are_registered():
    names = {p.name for p in (APP_DIR / "pages").glob("*.py")}
    assert {
        "dispatch_explorer.py", "forecast_error_cost.py", "oracle_capture.py",
        "sequential_clearing.py", "digital_twin.py", "run_catalogue.py",
    } <= names
    nav_source = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    for name in names:
        assert f"pages/{name}" in nav_source, f"{name} is not in the navigation"
