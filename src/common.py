"""Shared cohort, features, models, CV and bootstrap code. All choices come from CFG."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "diabetic_data.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CFG = {
    "seed": 42,
    "n_splits": 5,
    "n_boot": 2000,
    "exclude_dispositions": [11, 19, 20, 21, 13, 14],  # expired + hospice
    "bands": [("<40", 0, 40), ("40-59", 40, 60), ("60-79", 60, 80), ("80+", 80, 101)],
    "delta_pair": ("40-59", "80+"),
    "top_specialties": 15,
}
BANDS = [b[0] for b in CFG["bands"]]

NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
            "age_mid"]
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
            "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"]
CAT_COLS = ["race", "gender", "admission_type_id", "discharge_disposition_id",
            "admission_source_id", "specialty", "dg1", "dg2", "dg3", "max_glu_serum",
            "A1Cresult", "change", "diabetesMed"] + MED_COLS


# ----------------------------------------------------------------- cohort
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA, na_values="?", keep_default_na=False, low_memory=False,
                     dtype={"diag_1": str, "diag_2": str, "diag_3": str})
    # "None" is a real category for max_glu_serum / A1Cresult, not missing
    return df


def build_cohort(df: pd.DataFrame, first_only: bool = True, exclude: bool = True) -> pd.DataFrame:
    df = df.copy()
    if exclude:
        df = df[~df["discharge_disposition_id"].isin(CFG["exclude_dispositions"])]
    df = df.sort_values("encounter_id")
    if first_only:
        df = df.drop_duplicates("patient_nbr", keep="first")
    df = df.reset_index(drop=True)
    df["y"] = (df["readmitted"] == "<30").astype(int)
    lower = df["age"].str.extract(r"\[(\d+)-")[0].astype(int)
    df["age_mid"] = lower + 5
    df["age_band"] = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, lo, hi in CFG["bands"]:
        df.loc[(lower >= lo) & (lower < hi), "age_band"] = name
    assert df["age_band"].notna().all()
    return add_features(df)


def icd9_group(code) -> str:
    """Strack et al. (2014) grouping of ICD-9 codes."""
    if pd.isna(code):
        return "Missing"
    s = str(code)
    if s[0] in "VE":
        return "Other"
    v = float(s)
    if 390 <= v <= 459 or int(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "Digestive"
    if int(v) == 250:
        return "Diabetes"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    for i in (1, 2, 3):
        df[f"dg{i}"] = df[f"diag_{i}"].map(icd9_group)
        df[f"icd3_{i}"] = df[f"diag_{i}"].fillna("Missing").str.split(".").str[0]
    top = df["medical_specialty"].value_counts().index[: CFG["top_specialties"]]
    df["specialty"] = np.where(df["medical_specialty"].isin(top), df["medical_specialty"],
                               np.where(df["medical_specialty"].isna(), "Missing", "Other"))
    df["specialty_full"] = df["medical_specialty"].fillna("Missing")
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[c] = df[c].astype(str)
    for c in CAT_COLS:
        df[c] = df[c].fillna("Missing").astype(str)
    return df


# ----------------------------------------------------------------- models
def feature_frame(df: pd.DataFrame, num=None, cat=None) -> pd.DataFrame:
    num = NUM_COLS if num is None else num
    cat = CAT_COLS if cat is None else cat
    return df[num + cat].copy()


def make_model(name: str, num=None, cat=None, seed: int | None = None):
    num = NUM_COLS if num is None else num
    cat = CAT_COLS if cat is None else cat
    seed = CFG["seed"] if seed is None else seed
    if name in ("lr", "lr_C"):
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                  sparse_output=True), cat),
        ])
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"))])
    if name in ("gbm", "gbm_big"):
        import lightgbm as lgb
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)
        if name == "gbm_big":
            params = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8)
        return LGBMCat(cat=cat, random_state=seed, verbosity=-1, n_jobs=2, **params)
    raise ValueError(name)


class LGBMCat:
    """LightGBM with categoricals cast to pandas 'category' using categories seen in training."""

    def __init__(self, cat, **params):
        self.cat, self.params = cat, params

    def _prep(self, X):
        X = X.copy()
        for c in self.cat:
            X[c] = pd.Categorical(X[c], categories=self.cats_[c])
        return X

    def fit(self, X, y, sample_weight=None):
        import lightgbm as lgb
        self.cats_ = {c: sorted(X[c].unique()) for c in self.cat}
        self.m_ = lgb.LGBMClassifier(**self.params)
        self.m_.fit(self._prep(X), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self._prep(X))


def fit_model(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, clf__sample_weight=sample_weight)
    return model.fit(X, y, sample_weight=sample_weight)


def cv_folds(df: pd.DataFrame, seed: int | None = None, n_splits: int | None = None):
    seed = CFG["seed"] if seed is None else seed
    n_splits = CFG["n_splits"] if n_splits is None else n_splits
    band_idx = df["age_band"].map({b: i for i, b in enumerate(BANDS)}).values
    strat = df["y"].values * 10 + band_idx
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(df)), strat, groups=df["patient_nbr"].values))


def oof_predict(df, model_name, X=None, folds=None, weight_fn=None, y=None, **mk):
    X = feature_frame(df) if X is None else X
    y = df["y"].values if y is None else y
    folds = cv_folds(df) if folds is None else folds
    oof = np.full(len(df), np.nan)
    for tr, te in folds:
        m = make_model(model_name, **mk)
        w = None if weight_fn is None else weight_fn(df.iloc[tr])
        fit_model(m, X.iloc[tr], y[tr], w)
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


# ----------------------------------------------------------------- metrics
def auc(y, p, w=None):
    y = np.asarray(y)
    if w is not None:
        keep = np.asarray(w) > 0
        y, p, w = y[keep], np.asarray(p)[keep], np.asarray(w)[keep]
    if y.size == 0 or y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p, sample_weight=w)


def band_table(df, p, y=None, w=None):
    y = df["y"].values if y is None else y
    rows = {}
    for b in BANDS:
        m = (df["age_band"] == b).values
        rows[b] = auc(y[m], p[m], None if w is None else w[m])
    rows["overall"] = auc(y, p, w)
    return rows


class FastAUC:
    """Weighted AUC for fixed scores and labels under many different row weights.
    Sorting is done once; each call is O(n). Ties count one half (same as sklearn)."""

    def __init__(self, y, p, idx=None):
        idx = np.arange(len(y)) if idx is None else np.flatnonzero(idx) if np.asarray(idx).dtype == bool else idx
        order = idx[np.argsort(p[idx], kind="mergesort")]
        ps = p[order]
        self.order = order
        self.y = y[order].astype(float)
        self.gid = np.concatenate([[0], np.cumsum(ps[1:] != ps[:-1])])
        self.ng = self.gid[-1] + 1 if len(ps) else 0

    def __call__(self, w):
        ws = w[self.order]
        pos = np.bincount(self.gid, weights=ws * self.y, minlength=self.ng)
        neg = np.bincount(self.gid, weights=ws * (1 - self.y), minlength=self.ng)
        P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return np.nan
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def cluster_bootstrap(df, preds: dict, n_boot=None, y=None, extra=None, seed=None,
                      cluster_col="patient_nbr"):
    """Paired patient-level bootstrap. preds: name -> OOF prediction array (aligned with df).
    Each replicate draws patients with replacement and turns the draw into row weights,
    so every model and band is evaluated on the same resample (paired).
    extra(w) -> dict of further statistics computed on the same resample."""
    n_boot = CFG["n_boot"] if n_boot is None else n_boot
    rng = np.random.default_rng(CFG["seed"] if seed is None else seed)
    y = df["y"].values if y is None else y
    codes, uniq = pd.factorize(df[cluster_col])
    n_pat = len(uniq)
    masks = {b: (df["age_band"] == b).values for b in BANDS}
    fast = {}
    for name, p in preds.items():
        for b in BANDS:
            fast[(name, b)] = FastAUC(y, p, masks[b])
        fast[(name, "overall")] = FastAUC(y, p)
    reps = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)
        w = counts[codes].astype(float)
        rec = {k: f(w) for k, f in fast.items()}
        if extra is not None:
            rec.update(extra(w))
        reps.append(rec)
    return pd.DataFrame(reps)


def ci(s, level=0.95):
    s = pd.Series(s).dropna()
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def fmt(point, lo, hi, d=3):
    return f"{point:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def summarise(df, preds: dict, boot: pd.DataFrame, y=None) -> pd.DataFrame:
    """Per-model table: per-band AUC with CI, delta and max-min with paired CI."""
    y = df["y"].values if y is None else y
    b1, b2 = CFG["delta_pair"]
    out = []
    for name, p in preds.items():
        pt = band_table(df, p, y)
        row = {"model": name}
        for b in BANDS + ["overall"]:
            lo, hi = ci(boot[(name, b)])
            row[b] = pt[b]
            row[f"{b}_lo"], row[f"{b}_hi"] = lo, hi
        d = boot[(name, b1)] - boot[(name, b2)]
        row["delta"] = pt[b1] - pt[b2]
        row["delta_lo"], row["delta_hi"] = ci(d)
        mm = boot[[(name, b) for b in BANDS]].max(axis=1) - boot[[(name, b) for b in BANDS]].min(axis=1)
        vals = [pt[b] for b in BANDS]
        row["maxmin"] = max(vals) - min(vals)
        row["maxmin_lo"], row["maxmin_hi"] = ci(mm)
        out.append(row)
    return pd.DataFrame(out)


def print_summary(tab: pd.DataFrame, title: str):
    print(f"\n=== {title} ===")
    for _, r in tab.iterrows():
        bands = "  ".join(f"{b}: {r[b]:.3f} [{r[b+'_lo']:.3f},{r[b+'_hi']:.3f}]" for b in BANDS)
        print(f"{r['model']:>14s} | overall {fmt(r['overall'], r['overall_lo'], r['overall_hi'])}")
        print(f"{'':>14s} | {bands}")
        print(f"{'':>14s} | delta(40-59 minus 80+) = {fmt(r['delta'], r['delta_lo'], r['delta_hi'])}"
              f"   max-min = {fmt(r['maxmin'], r['maxmin_lo'], r['maxmin_hi'])}")


def band_counts(df, y=None) -> pd.DataFrame:
    y = df["y"].values if y is None else y
    rows = []
    for b in BANDS:
        m = (df["age_band"] == b).values
        rows.append({"band": b, "n": int(m.sum()), "positives": int(y[m].sum()),
                     "prevalence": float(y[m].mean())})
    rows.append({"band": "overall", "n": len(df), "positives": int(y.sum()),
                 "prevalence": float(y.mean())})
    return pd.DataFrame(rows)
