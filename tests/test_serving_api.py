import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from readmit_audit.api import create_app
from readmit_audit.serving import ReadmissionScorer, reliability_for_age

AUDIT = pd.DataFrame([
    {"group": "[20-30)", "auc": 0.80, "ci_lo": 0.70, "ci_hi": 0.88},
    {"group": "[80-90)", "auc": 0.55, "ci_lo": 0.50, "ci_hi": 0.60},
    {"group": "[10-20)", "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan},
])


class TestReliability:
    def test_reduced_when_ci_entirely_below_overall(self):
        assert reliability_for_age("[80-90)", AUDIT, 0.68)["level"] == "reduced"

    def test_standard_when_interval_reaches_overall(self):
        assert reliability_for_age("[20-30)", AUDIT, 0.68)["level"] == "standard"

    def test_insufficient_for_nan_or_absent_band(self):
        assert reliability_for_age("[10-20)", AUDIT, 0.68)["level"] == "insufficient_data"
        assert reliability_for_age("[40-50)", AUDIT, 0.68)["level"] == "insufficient_data"

    def test_unknown_when_age_missing(self):
        assert reliability_for_age(np.nan, AUDIT, 0.68)["level"] == "unknown"
        assert reliability_for_age(None, AUDIT, 0.68)["level"] == "unknown"


class TestScorer:
    def test_scores_partial_records_and_ignores_extras(self, tiny_artifact_path):
        s = ReadmissionScorer.from_path(tiny_artifact_path)
        out = s.score([{"age": "[50-60)", "gender": "Male", "not_a_feature": 1}])
        assert 0 <= out[0]["readmission_probability"] <= 1

    def test_unseen_category_does_not_crash(self, tiny_artifact_path):
        s = ReadmissionScorer.from_path(tiny_artifact_path)
        assert s.score([{"age": "[50-60)", "race": "NeverSeen"}])

    def test_question_mark_treated_as_missing(self, tiny_artifact_path):
        s = ReadmissionScorer.from_path(tiny_artifact_path)
        assert s.score([{"age": "[50-60)", "race": "?"}])

    def test_missing_artifact_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_pipeline"):
            ReadmissionScorer.from_path(tmp_path / "nope.joblib")


@pytest.fixture
def client(tiny_artifact_path):
    with TestClient(create_app(tiny_artifact_path)) as c:
        yield c


class TestApi:
    def test_health_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_model_info(self, client):
        j = client.get("/model-info").json()
        assert j["overall_auc"] == 0.68 and "[80-90)" in j["age_band_auc"]

    def test_predict_returns_reliability_flag_per_record(self, client):
        r = client.post("/predict", json={"records": [{"age": "[80-90)"}, {"age": "[20-30)"}]})
        assert r.status_code == 200
        p = r.json()["predictions"]
        assert [x["reliability"]["level"] for x in p] == ["reduced", "standard"]

    def test_predict_rejects_empty_batch(self, client):
        assert client.post("/predict", json={"records": []}).status_code == 422

    def test_predict_rejects_oversized_batch(self, client):
        assert client.post("/predict", json={"records": [{"age": "[50-60)"}] * 1001}).status_code == 422

    def test_predict_rejects_wrong_shape(self, client):
        assert client.post("/predict", json={"nope": 1}).status_code == 422

    def test_bad_value_type_gives_422_not_500(self, client):
        r = client.post("/predict", json={"records": [{"number_inpatient": "not-a-number", "age": "[50-60)"}]})
        assert r.status_code in (200, 422)  # must never be a server error

    def test_drift_check_needs_enough_records(self, client):
        assert client.post("/drift-check", json={"records": [{"age": "[50-60)"}] * 10}).status_code == 422

    def test_drift_check_flags_shifted_input(self, client):
        rows = [{"age": "[80-90)", "number_inpatient": 6, "time_in_hospital": 13, "gender": "Male"}] * 300
        j = client.post("/drift-check", json={"records": rows}).json()
        assert j["n_major"] >= 1 and j["features"][0]["status"] == "major"

    def test_missing_model_gives_503(self, tmp_path):
        with TestClient(create_app(str(tmp_path / "missing.joblib"))) as c:
            assert c.get("/health").status_code == 503
            assert c.post("/predict", json={"records": [{"age": "x"}]}).status_code == 503
