"""Every dashboard page must render without raising, whatever is (or isn't)
in results/ -- an empty results dir just shows each page's own warning."""
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
    assert names == {"study_overview.py", "dispatch_explorer.py", "forecast_comparison.py"}
    nav_source = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    for name in names:
        assert f"pages/{name}" in nav_source, f"{name} is not in the navigation"


@pytest.mark.parametrize("model", ["seasonal_naive", "per_market_best"])
def test_dispatch_explorer_renders_reserve_price_panel_for_a_reserve_strategy(model):
    at = AppTest.from_file(str(APP_DIR / "pages" / "dispatch_explorer.py"), default_timeout=180).run()
    model_box, strategy_box = at.sidebar.selectbox[1], at.sidebar.selectbox[2]
    if model not in model_box.options or "full_cooptimized" not in strategy_box.options:
        pytest.skip("no full_cooptimized run for this model in results/")
    at.sidebar.selectbox[1].select(model).run()
    at.sidebar.selectbox[2].select("full_cooptimized").run()
    assert not at.exception, [e.value for e in at.exception]


def test_load_forecast_prices_takes_a_market_and_resolves_blends():
    import sys
    sys.path.insert(0, str(APP_DIR))
    from data_loader import load_forecast_prices

    fcr = load_forecast_prices("per_market_best", "fcr")
    assert list(fcr.columns) == ["forecast", "y_true"] and not fcr.empty
    assert not load_forecast_prices("seasonal_naive").equals(load_forecast_prices("seasonal_naive", "fcr"))
