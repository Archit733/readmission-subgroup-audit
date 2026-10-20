"""Phase 1: does the age gap survive patient-level analysis?

Conditions
  A  all encounters, random row-level 80/20 split, row bootstrap   (the capstone's design)
  B  all encounters, patient-grouped 5-fold CV, patient bootstrap
  C  B + death/hospice exclusions
  D  first encounter per patient + exclusions                       (primary)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (BANDS, CFG, RESULTS, band_counts, build_cohort, cluster_bootstrap,
                    feature_frame, fit_model, load_raw, make_model, oof_predict,
                    print_summary, summarise)

MODELS = ["lr", "gbm"]
raw = load_raw()
tables = []


def run(label, df, preds, n_boot, cluster_col="patient_nbr", sub=None):
    d = df if sub is None else df.iloc[sub].reset_index(drop=True)
    boot = cluster_bootstrap(d, preds, n_boot=n_boot, cluster_col=cluster_col)
    tab = summarise(d, preds, boot).assign(condition=label, n=len(d),
                                           patients=d["patient_nbr"].nunique())
    print_summary(tab, f"{label}  (n={len(d)}, patients={d['patient_nbr'].nunique()})")
    tables.append(tab)
    return boot


# ---- A: capstone design -------------------------------------------------
dfA = build_cohort(raw, first_only=False, exclude=False)
dfA["row_id"] = np.arange(len(dfA))
XA = feature_frame(dfA)
tr, te = train_test_split(np.arange(len(dfA)), test_size=0.2, random_state=CFG["seed"],
                          stratify=dfA["y"])
predsA = {}
for m in MODELS:
    model = make_model(m)
    fit_model(model, XA.iloc[tr], dfA["y"].values[tr])
    predsA[m] = model.predict_proba(XA.iloc[te])[:, 1]
# how many test rows belong to a patient who also appears in training?
leak = dfA["patient_nbr"].iloc[te].isin(set(dfA["patient_nbr"].iloc[tr])).mean()
print(f"A: {leak:.1%} of test encounters belong to a patient also present in training")
run("A_row_split_all", dfA, predsA, 1000, cluster_col="row_id", sub=te)

# ---- B, C, D ------------------------------------------------------------
oof_store = {}
for label, first, excl, nb in [("B_grouped_all", False, False, 1000),
                               ("C_grouped_excl", False, True, 1000),
                               ("D_primary", True, True, CFG["n_boot"])]:
    df = build_cohort(raw, first_only=first, exclude=excl)
    preds = {m: oof_predict(df, m) for m in MODELS}
    oof_store[label] = preds
    boot = run(label, df, preds, nb)
    if label == "D_primary":
        counts = band_counts(df)
        print("\nPrimary cohort base rates\n", counts.to_string(index=False))
        counts.to_csv(RESULTS / "p1_base_rates.csv", index=False)
        pd.DataFrame({"patient_nbr": df["patient_nbr"], "age": df["age"], "age_band": df["age_band"],
                      "y": df["y"], **{f"oof_{m}": p for m, p in preds.items()}}).to_csv(
            RESULTS / "oof_primary.csv", index=False)
        boot.to_pickle(RESULTS / "boot_primary.pkl")

        # secondary: original 10-year buckets, for continuity with the capstone
        rows = []
        y = df["y"].values
        from common import FastAUC
        rng = np.random.default_rng(CFG["seed"])
        for a in sorted(df["age"].unique(), key=lambda s: int(s[1:].split("-")[0])):
            msk = (df["age"] == a).values
            for m in MODELS:
                f = FastAUC(y, preds[m], msk)
                pt = f(np.ones(len(df)))
                reps = [f(np.bincount(rng.integers(0, len(df), len(df)), minlength=len(df)).astype(float))
                        for _ in range(500)]
                lo, hi = np.nanpercentile(reps, [2.5, 97.5])
                rows.append({"age": a, "model": m, "n": int(msk.sum()), "pos": int(y[msk].sum()),
                             "auc": pt, "lo": lo, "hi": hi})
        dec = pd.DataFrame(rows)
        print("\nPrimary cohort, 10-year buckets\n",
              dec.pivot(index="age", columns="model", values="auc").round(3).to_string())
        dec.to_csv(RESULTS / "p1_decades.csv", index=False)

# ---- seed robustness of delta in the primary cohort ---------------------
df = build_cohort(raw)
y = df["y"].values
b1, b2 = CFG["delta_pair"]
rows = []
from common import cv_folds, band_table
for s in range(5):
    folds = cv_folds(df, seed=100 + s)
    for m in MODELS:
        p = oof_predict(df, m, folds=folds)
        bt = band_table(df, p)
        rows.append({"seed": 100 + s, "model": m, **bt, "delta": bt[b1] - bt[b2]})
seeds = pd.DataFrame(rows)
print("\nDelta across 5 CV re-splits (primary cohort)\n",
      seeds.groupby("model")["delta"].agg(["mean", "std", "min", "max"]).round(4).to_string())
seeds.to_csv(RESULTS / "p1_seed_robustness.csv", index=False)

out = pd.concat(tables, ignore_index=True)
out.to_csv(RESULTS / "p1_leakage.csv", index=False)
print("\nsaved results/p1_*.csv")
