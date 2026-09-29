import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tiny_artifact_path(tmp_path_factory):
    """A small, real fitted pipeline in the exact serving-artifact schema, so
    API/serving tests do not depend on running the full pipeline."""
    import joblib
    from readmit_audit.modeling import build_strong_pipeline
    from readmit_audit.serving import make_serving_artifact
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({
        "age": rng.choice(["[20-30)", "[50-60)", "[80-90)"], n),
        "gender": rng.choice(["Female", "Male"], n),
        "race": rng.choice(["Caucasian", "AfricanAmerican", np.nan], n, p=[.6, .3, .1]),
        "number_inpatient": rng.poisson(0.6, n),
        "time_in_hospital": rng.integers(1, 14, n),
    })
    y = (rng.uniform(size=n) < 0.05 + 0.08 * X["number_inpatient"].clip(0, 4)).astype(int)
    cfg = {"models": {"strong": {"max_iter": 30, "learning_rate": 0.1}}, "split": {"random_state": 0}}
    model = build_strong_pipeline(X, cfg).fit(X, y)
    audit = pd.DataFrame([
        {"group": "[20-30)", "n": 500, "auc": 0.80, "ci_lo": 0.70, "ci_hi": 0.88},
        {"group": "[50-60)", "n": 500, "auc": 0.68, "ci_lo": 0.64, "ci_hi": 0.72},
        {"group": "[80-90)", "n": 500, "auc": 0.55, "ci_lo": 0.50, "ci_hi": 0.60},
        {"group": "[10-20)", "n": 50, "auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan},
    ])
    art = make_serving_artifact(model, X.columns, audit, overall_auc=0.68, flag_threshold=0.2,
                                flag_fraction=0.2, reference=X)
    p = tmp_path_factory.mktemp("model") / "model_for_serving.joblib"
    joblib.dump(art, p)
    return str(p)
