"""Post-deployment monitoring: input drift (PSI) and per-subgroup performance
alerts. The point of the project applies here too: an aggregate metric can stay
flat while one subgroup degrades, so alerts are evaluated per subgroup."""
import numpy as np
import pandas as pd

from .audit import bootstrap_auc_ci

PSI_STABLE, PSI_MAJOR = 0.10, 0.25   # conventional industry thresholds


def psi_status(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "stable" if value < PSI_STABLE else ("moderate" if value < PSI_MAJOR else "major")


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4, min_share: float = 0.01) -> float:
    """Population Stability Index between a reference and a current sample.
    Numeric with many distinct values: quantile bins from the reference.
    Otherwise (categorical / few distinct values): category proportions.
    Categories under `min_share` of the reference are pooled into "__other__": without
    this, high-cardinality columns (e.g. ~550 ICD-9 codes) show large PSI from sampling
    noise alone at batch sizes of a few thousand. Unseen categories in `actual` land in
    "__other__" too, so a burst of brand-new codes still registers."""
    e = pd.Series(expected).dropna(); a = pd.Series(actual).dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan
    numeric = pd.api.types.is_numeric_dtype(e) and e.nunique() > n_bins
    if numeric:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e_p = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        a_p = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    else:
        e_s, a_s = e.astype(str), a.astype(str)
        share = e_s.value_counts(normalize=True)
        keep = set(share[share >= min_share].index)
        e_s = e_s.where(e_s.isin(keep), "__other__"); a_s = a_s.where(a_s.isin(keep), "__other__")
        cats = sorted(set(e_s) | set(a_s))
        e_p = e_s.value_counts(normalize=True).reindex(cats).fillna(0)
        a_p = a_s.value_counts(normalize=True).reindex(cats).fillna(0)
    e_p, a_p = np.clip(e_p.values, eps, None), np.clip(a_p.values, eps, None)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None) -> pd.DataFrame:
    cols = columns or [c for c in reference.columns if c in current.columns]
    rows = [{"feature": c, "psi": psi(reference[c], current[c])} for c in cols]
    out = pd.DataFrame(rows)
    out["status"] = out["psi"].apply(psi_status)
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def degradation_alert(batch_ci_hi: float, reference_auc: float) -> bool:
    """Alert only when the batch AUC is significantly BELOW the reference: the whole
    bootstrap interval sits under the reference point estimate. Noise alone rarely does that."""
    return bool(not np.isnan(batch_ci_hi) and batch_ci_hi < reference_auc)


def batch_subgroup_monitor(y, p, groups, batch_ids, reference: dict, min_n: int = 150,
                           n_boot: int = 300, random_state: int = 42, window: int = 1) -> pd.DataFrame:
    """Per batch x subgroup AUC with bootstrap CI and an alert flag.
    `reference` maps group -> reference AUC (and key "__overall__" for the aggregate).
    `window` > 1 pools the current and previous window-1 batches (trailing window):
    single small batches are too noisy for subgroup AUC, so pooling trades a little
    detection delay for much narrower intervals."""
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups); batch_ids = np.asarray(batch_ids)
    rows = []
    for b in sorted(np.unique(batch_ids)):
        mb = (batch_ids <= b) & (batch_ids > b - window)
        for g in ["__overall__"] + sorted(set(groups[mb])):
            m = mb if g == "__overall__" else mb & (groups == g)
            if m.sum() < min_n or g not in reference:
                continue
            auc, lo, hi = bootstrap_auc_ci(y[m], p[m], n_boot=n_boot, random_state=random_state)
            rows.append({"batch": int(b), "window": window, "group": g, "n": int(m.sum()), "auc": auc, "ci_lo": lo, "ci_hi": hi,
                         "reference_auc": reference[g], "alert": degradation_alert(hi, reference[g])})
    return pd.DataFrame(rows)
