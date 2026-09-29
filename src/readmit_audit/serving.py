"""Scoring layer shared by the API and the dashboard (no web framework needed).

Each prediction carries a `reliability` note derived from the subgroup audit:
if the model is significantly worse for the patient's age band than in aggregate,
the score is returned with a reduced-reliability flag so a human can weigh it.
This is decision support, not a gate: the flag never blocks or downgrades care.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

AGE_BAND_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]


def make_serving_artifact(model, feature_columns, age_audit: pd.DataFrame, overall_auc: float,
                          flag_threshold: float, flag_fraction: float, reference: pd.DataFrame,
                          model_version: str = "0.1.0") -> dict:
    """Single place defining the serving artifact schema (used by the pipeline and tests)."""
    return {"model": model, "feature_columns": list(feature_columns), "age_audit": age_audit,
            "overall_auc": float(overall_auc), "flag_threshold": float(flag_threshold),
            "flag_fraction": float(flag_fraction), "reference": reference, "model_version": model_version}


def reliability_for_age(age_band, age_audit: pd.DataFrame, overall_auc: float) -> dict:
    """standard | reduced | insufficient_data | unknown, with a plain-language note."""
    if age_band is None or (isinstance(age_band, float) and np.isnan(age_band)):
        return {"level": "unknown", "note": "Age band missing; subgroup reliability cannot be assessed."}
    row = age_audit[age_audit["group"] == age_band]
    if row.empty or row["ci_hi"].isna().iloc[0]:
        return {"level": "insufficient_data",
                "note": f"Too little data in age band {age_band} to assess model reliability."}
    r = row.iloc[0]
    if r["ci_hi"] < overall_auc:
        return {"level": "reduced", "band_auc": round(float(r["auc"]), 3),
                "note": (f"Model ranks risk less reliably for age band {age_band} "
                         f"(AUC {r['auc']:.2f} vs {overall_auc:.2f} overall). Treat the score as one input; "
                         "clinical judgement should lead.")}
    return {"level": "standard", "band_auc": round(float(r["auc"]), 3), "note": "No reliability concern detected."}


class ReadmissionScorer:
    def __init__(self, artifact: dict):
        self.a = artifact
        self.model = artifact["model"]
        self.columns = artifact["feature_columns"]

    @classmethod
    def from_path(cls, path) -> "ReadmissionScorer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at {p}. Run scripts/run_pipeline.py first.")
        return cls(joblib.load(p))

    def to_frame(self, records: list) -> pd.DataFrame:
        """Records may omit fields (imputed) or include extras (ignored)."""
        df = pd.DataFrame(records).replace("?", np.nan)
        return df.reindex(columns=self.columns)

    def score(self, records: list) -> list:
        X = self.to_frame(records)
        probs = self.model.predict_proba(X)[:, 1]
        out = []
        for i, pr in enumerate(probs):
            age = X["age"].iloc[i] if "age" in X.columns else None
            out.append({"readmission_probability": round(float(pr), 4),
                        "flagged_high_risk": bool(pr >= self.a["flag_threshold"]),
                        "reliability": reliability_for_age(age, self.a["age_audit"], self.a["overall_auc"])})
        return out

    def info(self) -> dict:
        aa = self.a["age_audit"]
        return {"model_version": self.a["model_version"], "overall_auc": round(self.a["overall_auc"], 3),
                "flag_threshold": round(self.a["flag_threshold"], 4),
                "flag_fraction": self.a["flag_fraction"], "n_features": len(self.columns),
                "age_band_auc": {r["group"]: (None if pd.isna(r["auc"]) else round(float(r["auc"]), 3))
                                 for _, r in aa.iterrows()}}
