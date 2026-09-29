"""Smoke-tests the Streamlit app end to end with Streamlit's AppTest. Skipped if
pipeline outputs are absent (run scripts/run_pipeline.py first)."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "reports/results/pipeline_artifacts.joblib").exists(),
                                reason="pipeline outputs missing")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    at.run()
    return at


def test_app_loads_without_exception(app):
    assert not app.exception
    assert len(app.tabs) == 6
    assert app.title[0].value.startswith("Readmission risk")


def test_overview_shows_headline_metrics(app):
    labels = [m.label for m in app.metric]
    assert "Aggregate AUC" in labels and "Calibration error (ECE)" in labels


def test_operating_point_slider_recomputes(app):
    app.slider[0].set_value(0.30).run()
    assert not app.exception
    assert "0." in " ".join(c.value for c in app.caption if "threshold" in c.value)


def test_score_button_returns_probability_and_reliability(app):
    app.button[0].click().run()
    assert not app.exception
    assert any(m.label.startswith("Readmission probability") for m in app.metric)
    texts = [e.value for e in list(app.warning) + list(app.success)]
    assert any("Reliability for this age band" in t for t in texts)
