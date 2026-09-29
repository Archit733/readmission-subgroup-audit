"""Baseline and stronger model, each a single sklearn Pipeline
(preprocess -> [scale] -> model), so raw dataframes can be scored directly."""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import build_preprocessor

logger = logging.getLogger(__name__)


def build_baseline_pipeline(X: pd.DataFrame, cfg: dict) -> Pipeline:
    p = cfg["models"]["baseline"]
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=p.get("max_iter", 1000),
                                     random_state=cfg["split"]["random_state"],
                                     class_weight=p.get("class_weight"))),
    ])


def build_strong_pipeline(X: pd.DataFrame, cfg: dict) -> Pipeline:
    """HistGradientBoosting: ships with scikit-learn, no libomp dependency."""
    p = cfg["models"]["strong"]
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 300), learning_rate=p.get("learning_rate", 0.05),
            random_state=cfg["split"]["random_state"])),
    ])


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    logger.info(f"Training {pipeline.steps[-1][1].__class__.__name__} on {X_train.shape[0]} rows")
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_proba_positive(pipeline: Pipeline, X_test) -> np.ndarray:
    return pipeline.predict_proba(X_test)[:, 1]
