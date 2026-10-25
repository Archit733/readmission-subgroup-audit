"""Phase 4: can the gap be closed?

1. Band-specific models (train and test within band, same folds).
2. Reweighting: 80+ upweighted x2 and x5 in training.
3. Richer features: 3-digit ICD-9 codes for diag_1..3 and full medical_specialty (pre-specified).
4. Exploratory: geriatric-syndrome flags from the three recorded diagnosis codes.
All compared with the baseline on the same patients with a paired patient-level bootstrap.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from common import (BANDS, CAT_COLS, CFG, NUM_COLS, RESULTS, build_cohort, cluster_bootstrap,
                    cv_folds, load_raw, oof_predict, print_summary, summarise)

b1, b2 = CFG["delta_pair"]
df = build_cohort(load_raw())
y = df["y"].values
folds = cv_folds(df)
base = pd.read_csv(RESULTS / "oof_primary.csv")
preds = {"lr": base["oof_lr"].values, "gbm": base["oof_gbm"].values}

# ---- 1. band-specific models ------------------------------------------------
for m in ["lr", "gbm"]:
    p = np.full(len(df), np.nan)
    for b in BANDS:
        idx = np.flatnonzero((df["age_band"] == b).values)
        pos = {v: i for i, v in enumerate(idx)}
        sub = df.iloc[idx].reset_index(drop=True)
        f_b = [(np.array([pos[i] for i in tr if i in pos]), np.array([pos[i] for i in te if i in pos]))
               for tr, te in folds]
        p[idx] = oof_predict(sub, m, folds=f_b)
    preds[f"{m}_bandspecific"] = p

# ---- 2. reweighting -----------------------------------------------------------
for k in [2, 5]:
    wf = lambda d, k=k: np.where(d["age_band"].values == b2, float(k), 1.0)
    for m in ["lr", "gbm"]:
        preds[f"{m}_w80x{k}"] = oof_predict(df, m, folds=folds, weight_fn=wf)

# ---- 3. richer features (pre-specified) -----------------------------------------
cat_rich = [c for c in CAT_COLS if c != "specialty"] + ["specialty_full", "icd3_1", "icd3_2", "icd3_3"]
for m in ["lr", "gbm"]:
    preds[f"{m}_richICD"] = oof_predict(df, m, folds=folds, X=df[NUM_COLS + cat_rich],
                                        num=NUM_COLS, cat=cat_rich)

# ---- 4. exploratory: geriatric-syndrome flags ----------------------------------
GERI = {
    "dementia": lambda v: v in (290, 294, 331),
    "delirium": lambda v: v in (293,),
    "parkinson": lambda v: v in (332,),
    "fall_fracture": lambda v: v in (820, 821, 805, 808, 812, 813),
    "pressure_ulcer": lambda v: v in (707,),
    "malnutrition_dehydration": lambda v: v in (260, 261, 262, 263, 276),
    "uti": lambda v: v in (599,),
    "pneumonia_aspiration": lambda v: v in (480, 481, 482, 485, 486, 507),
}
codes = df[["icd3_1", "icd3_2", "icd3_3"]].map(lambda s: int(s) if s.isdigit() else -1)
geri_cols = []
for name, f in GERI.items():
    df[f"geri_{name}"] = codes.map(f).any(axis=1).astype(int)
    geri_cols.append(f"geri_{name}")
df["geri_count"] = df[geri_cols].sum(axis=1)
print("share with each geriatric flag by band\n",
      df.groupby("age_band")[geri_cols + ["geri_count"]].mean().loc[BANDS].round(3).T.to_string())
num_g = NUM_COLS + geri_cols + ["geri_count"]
for m in ["lr", "gbm"]:
    preds[f"{m}_geri"] = oof_predict(df, m, folds=folds, X=df[num_g + CAT_COLS], num=num_g, cat=CAT_COLS)

# ---- compare ------------------------------------------------------------------
boot = cluster_bootstrap(df, preds)
tab = summarise(df, preds, boot)
print_summary(tab, "interventions")
rows = []
for name in preds:
    ref = "lr" if name.startswith("lr") else "gbm"
    g80 = boot[(name, b2)] - boot[(ref, b2)]
    dd = (boot[(name, b1)] - boot[(name, b2)]) - (boot[(ref, b1)] - boot[(ref, b2)])
    gall = boot[(name, "overall")] - boot[(ref, "overall")]
    rows.append({"variant": name, "baseline": ref,
                 "auc80": tab.set_index("model").loc[name, b2],
                 "auc80_change": g80.mean(), "auc80_change_lo": g80.quantile(.025),
                 "auc80_change_hi": g80.quantile(.975),
                 "delta": tab.set_index("model").loc[name, "delta"],
                 "delta_change": dd.mean(), "delta_change_lo": dd.quantile(.025),
                 "delta_change_hi": dd.quantile(.975),
                 "overall_change": gall.mean()})
comp = pd.DataFrame(rows)
print("\nPaired changes vs baseline of the same model family")
print(comp.round(4).to_string(index=False))
tab.to_csv(RESULTS / "p4_interventions.csv", index=False)
comp.to_csv(RESULTS / "p4_interventions_paired.csv", index=False)
print("\nsaved results/p4_*.csv")
