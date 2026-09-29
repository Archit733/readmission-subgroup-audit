"""Core audit logic: bootstrapped AUC confidence intervals, subgroup breakdown,
and calibration (ECE). This is the module the rest of the project depends on
for statistical validity -- keep it small, keep it tested.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, confidence: float = 0.95,
                      min_n: int = 20, random_state: int = 42):
    """Bootstrap a confidence interval for AUC-ROC.

    Returns (point_estimate, ci_lo, ci_hi). Returns (nan, nan, nan) if there
    isn't enough data or class diversity to compute AUC meaningfully -- this
    is a deliberate guard, not a silent failure: callers should check for NaN
    rather than trust a CI computed on too little data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    if n < min_n or len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    point = roc_auc_score(y_true, y_prob)

    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], y_prob[idx]
        if len(np.unique(yb)) < 2:
            continue  # a resample with only one class can't produce an AUC; skip it
        boots.append(roc_auc_score(yb, pb))

    if len(boots) < 50:
        # too few valid bootstrap resamples to trust a percentile CI
        return point, np.nan, np.nan

    alpha = 1 - confidence
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def subgroup_audit(y_true: pd.Series, y_prob: pd.Series, subgroup_labels: pd.Series,
                    cfg: dict, group_order: list = None) -> pd.DataFrame:
    """Run bootstrap_auc_ci for every level of one subgroup column.
    Groups smaller than cfg['audit']['min_subgroup_n'] are skipped, not
    force-fit -- an unreliable CI is worse than no result.
    """
    n_boot = cfg["audit"]["n_bootstrap"]
    confidence = cfg["audit"]["confidence_level"]
    min_n = cfg["audit"]["min_subgroup_n"]
    random_state = cfg["split"]["random_state"]

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    subgroup_labels = pd.Series(subgroup_labels).reset_index(drop=True)

    groups = group_order if group_order is not None else subgroup_labels.dropna().unique()
    rows = []
    for g in groups:
        mask = (subgroup_labels == g).values
        n = mask.sum()
        if n < min_n:
            logger.debug(f"Skipping group '{g}': n={n} < min_subgroup_n={min_n}")
            continue
        point, lo, hi = bootstrap_auc_ci(y_true[mask], y_prob[mask], n_boot=n_boot,
                                          confidence=confidence, random_state=random_state)
        rows.append({
            "group": g, "n": int(n), "pos_rate": float(y_true[mask].mean()),
            "auc": point, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def groups_differ(row_a: pd.Series, row_b: pd.Series) -> bool:
    """True if two subgroup_audit() rows have non-overlapping AUC confidence
    intervals -- the minimum bar for calling a gap 'real' rather than noise."""
    if any(pd.isna([row_a["ci_lo"], row_a["ci_hi"], row_b["ci_lo"], row_b["ci_hi"]])):
        return False
    return bool((row_a["ci_hi"] < row_b["ci_lo"]) or (row_b["ci_hi"] < row_a["ci_lo"]))


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: for each predicted-probability bin, mean predicted
    probability vs. observed frequency, plus bin size (for weighting/plotting)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        rows.append({"bin": b, "n": int(n), "mean_predicted": conf,
                      "observed_freq": acc, "gap": abs(conf - acc)})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: n-weighted average |predicted - observed| across bins."""
    tbl = calibration_table(y_true, y_prob, n_bins=n_bins)
    if tbl.empty:
        return np.nan
    return float((tbl["gap"] * tbl["n"]).sum() / tbl["n"].sum())
