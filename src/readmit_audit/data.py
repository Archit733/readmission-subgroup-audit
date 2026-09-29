"""Data loading, cleaning, and train/test splitting."""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)


def load_raw(cfg: dict) -> pd.DataFrame:
    """Load the raw CSV and apply the binary target + basic cleaning."""
    path = cfg["data"]["path"]
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = cfg["data"]["target_column"]
    positive_label = cfg["data"]["positive_label"]
    df["target"] = (df[target_col] == positive_label).astype(int)
    logger.info(f"Positive rate: {df['target'].mean():.4f}")

    df = df.replace("?", np.nan)
    return df


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop leakage/administrative columns per config. Keeps subgroup columns."""
    drop_cols = cfg["data"]["drop_columns"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["target"])
    return X


def split_data(df: pd.DataFrame, X: pd.DataFrame, cfg: dict):
    """Train/test split. If split.group_column is set (default: patient_nbr), whole
    patients go to one side only -- the dataset has several encounters per patient,
    and a naive row split lets the same patient appear in train and test (leakage).
    Uses StratifiedGroupKFold so the positive rate stays balanced across sides.
    Returns X_train, X_test, y_train, y_test, idx_train, idx_test."""
    y = df["target"]
    test_size = cfg["split"]["test_size"]
    random_state = cfg["split"]["random_state"]
    group_col = cfg["split"].get("group_column")
    if group_col:
        n_splits = max(2, int(round(1 / test_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        tr, te = next(sgkf.split(X, y, groups=df[group_col]))
        idx_train, idx_test = df.index[tr], df.index[te]
        return X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test], idx_train, idx_test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def get_subgroups(df: pd.DataFrame, idx_test, cfg: dict) -> pd.DataFrame:
    """Return the raw (unencoded) subgroup label columns for the test set, aligned to position 0..n-1."""
    cols = cfg["data"]["subgroup_columns"]
    return df.loc[idx_test, cols].reset_index(drop=True)
