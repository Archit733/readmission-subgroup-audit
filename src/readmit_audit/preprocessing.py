"""Shared preprocessing: one ColumnTransformer (impute + ordinal-encode) used as
the first step of every model pipeline.

Why: the earlier `.astype('category').cat.codes` approach never saved the
category->code mapping, so a saved model could not be reliably applied to new
raw data. With OrdinalEncoder inside the Pipeline, the mapping is saved with the
model and unseen categories map to a fixed unknown_value instead of crashing.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def infer_column_types(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return cat_cols, num_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols, num_cols = infer_column_types(X)
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat_pipeline, cat_cols), ("num", num_pipeline, num_cols)])
