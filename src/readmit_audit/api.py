"""FastAPI service: risk scoring with subgroup-reliability notes, plus drift checks.

Run:  uvicorn readmit_audit.api:app --app-dir src --port 8000
Model path: env READMIT_MODEL_PATH (default reports/results/model_for_serving.joblib).
Patient records are never logged or stored; only request counts are logged.
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .monitoring import feature_drift_report
from .serving import ReadmissionScorer

logger = logging.getLogger("readmit_api")
MAX_BATCH, MIN_DRIFT_SAMPLE = 1000, 200


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH)


class DriftRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(min_length=MIN_DRIFT_SAMPLE, max_length=20000)


def create_app(model_path: str = None) -> FastAPI:
    path = model_path or os.environ.get("READMIT_MODEL_PATH", "reports/results/model_for_serving.joblib")
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            state["scorer"] = ReadmissionScorer.from_path(path)
            logger.info("Model loaded")
        except FileNotFoundError as e:
            state["error"] = str(e)
            logger.error(str(e))
        yield

    app = FastAPI(title="Readmission risk API", version="0.1.0", lifespan=lifespan,
                  description="Decision support only. Scores carry a per-age-band reliability note.")

    def scorer() -> ReadmissionScorer:
        if "scorer" not in state:
            raise HTTPException(status_code=503, detail=state.get("error", "Model not loaded"))
        return state["scorer"]

    @app.get("/health")
    def health():
        if "scorer" not in state:
            raise HTTPException(status_code=503, detail=state.get("error", "Model not loaded"))
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info():
        return scorer().info()

    @app.post("/predict")
    def predict(req: PredictRequest):
        s = scorer()
        try:
            preds = s.score(req.records)
        except Exception as e:  # malformed value types etc.
            logger.warning(f"Scoring failed for batch of {len(req.records)}: {type(e).__name__}")
            raise HTTPException(status_code=422, detail=f"Could not score records: {type(e).__name__}: {e}")
        logger.info(f"Scored {len(preds)} records")
        return {"model_version": s.a["model_version"], "predictions": preds}

    @app.post("/drift-check")
    def drift_check(req: DriftRequest):
        s = scorer()
        cur = s.to_frame(req.records)
        rep = feature_drift_report(s.a["reference"], cur)
        return {"n_records": len(cur),
                "n_major": int((rep["status"] == "major").sum()),
                "n_moderate": int((rep["status"] == "moderate").sum()),
                "features": rep.head(15).round(4).astype(object).where(rep.head(15).notna(), None)
                             .to_dict(orient="records")}

    return app


app = create_app()
