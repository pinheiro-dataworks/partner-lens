"""Render every tab of the app against the versioned artifacts in data/processed/.

This is the check that the served artifacts and the app code agree: a column
renamed in the pipeline but not in the app fails here, not on the deployed
site.
"""
import pytest

from src import config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = config.ROOT_DIR / "app" / "streamlit_app.py"
SERVED_ARTIFACTS = ["sellers_segmented", "cluster_profiles", "model_diagnostics", "data_quality_report"]


@pytest.fixture(scope="module")
def app():
    missing = [name for name in SERVED_ARTIFACTS if not config.PROCESSED_FILES[name].exists()]
    if missing:
        pytest.fail(f"served artifacts missing from data/processed/: {missing}")
    return streamlit_testing.AppTest.from_file(str(APP_PATH), default_timeout=120).run()


def test_overview_renders(app):
    assert not app.exception
    assert app.title[0].value == "Executive Overview"


@pytest.mark.parametrize("tab", ["Segment Explorer", "Seller Lookup", "Methodology"])
def test_tab_renders(app, tab):
    app.sidebar.radio[0].set_value(tab).run()
    assert not app.exception, [e.message for e in app.exception]
