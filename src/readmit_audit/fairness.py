"""Independent cross-check of the hand-written subgroup audit against Fairlearn,
plus threshold-based group metrics at a realistic operating point.

Why both: bootstrap AUC (audit.py) answers "does the model RANK patients equally
well in every group?". Deployment, though, means a threshold: e.g. "flag the top
20% highest-risk patients for follow-up". Selection rate and true-positive rate at
that threshold are what a care-management team actually experiences.
"""
import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
                               false_positive_rate)
from sklearn.metrics import roc_auc_score


def fairlearn_auc_by_group(y_true, y_prob, groups) -> pd.Series:
    mf = MetricFrame(metrics=roc_auc_score, y_true=np.asarray(y_true),
                     y_pred=np.asarray(y_prob), sensitive_features=np.asarray(groups))
    return mf.by_group


def crosscheck_auc(audit_table: pd.DataFrame, y_true, y_prob, groups, atol=1e-9) -> pd.DataFrame:
    """Compare audit.subgroup_audit point estimates with Fairlearn's independent
    computation. Returns the merged table with an `agrees` column."""
    fl = fairlearn_auc_by_group(y_true, y_prob, groups).rename("fairlearn_auc")
    merged = audit_table.merge(fl, left_on="group", right_index=True, how="left")
    merged["agrees"] = np.isclose(merged["auc"], merged["fairlearn_auc"], atol=atol)
    return merged


def top_fraction_threshold(y_prob, fraction: float) -> float:
    """Score threshold that flags the top `fraction` of patients by risk."""
    return float(np.quantile(np.asarray(y_prob), 1 - fraction))


def threshold_metrics_by_group(y_true, y_prob, groups, threshold: float,
                               min_n: int = 100) -> pd.DataFrame:
    """Selection rate, TPR, FPR per group at a fixed threshold (groups below min_n dropped)."""
    y_true = np.asarray(y_true); y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "tpr": true_positive_rate,
                 "fpr": false_positive_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=np.asarray(groups))
    out = mf.by_group.copy()
    out["n"] = pd.Series(np.asarray(groups)).value_counts().reindex(out.index).values
    return out[out["n"] >= min_n].reset_index().rename(columns={"index": "group", out.index.name or "index": "group"})


def max_gap(table: pd.DataFrame, col: str) -> float:
    return float(table[col].max() - table[col].min())


def bootstrap_tpr_ci(y_true, y_pred, n_boot=1000, confidence=0.95, random_state=42, min_pos=10):
    """Bootstrap CI for true-positive rate (recall). Returns (tpr, lo, hi, n_pos);
    NaN interval when the group has fewer than `min_pos` actual positives."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos < min_pos:
        return np.nan, np.nan, np.nan, n_pos
    hits = y_pred[pos]
    tpr = float(hits.mean())
    rng = np.random.default_rng(random_state)
    boots = rng.choice(hits, size=(n_boot, n_pos), replace=True).mean(axis=1)
    a = 1 - confidence
    lo, hi = np.percentile(boots, [100 * a / 2, 100 * (1 - a / 2)])
    return tpr, float(lo), float(hi), n_pos
