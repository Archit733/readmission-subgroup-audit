"""Phase 2: rule out cheap explanations for the age gap (primary cohort)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import (BANDS, CAT_COLS, CFG, NUM_COLS, RESULTS, FastAUC, band_counts, build_cohort,
                    ci, cluster_bootstrap, cv_folds, fmt, load_raw, oof_predict,
                    print_summary, summarise)

MODELS = ["lr", "gbm"]
b1, b2 = CFG["delta_pair"]
raw = load_raw()
df = build_cohort(raw)
oof = pd.read_csv(RESULTS / "oof_primary.csv")
assert (oof["patient_nbr"].values == df["patient_nbr"].values).all()
preds = {m: oof[f"oof_{m}"].values for m in MODELS}
y = df["y"].values
out = {}

# ---- 1. base rates --------------------------------------------------------
print("1. BASE RATES\n", band_counts(df).to_string(index=False))

# ---- 2. label definitions -------------------------------------------------
print("\n2. LABEL DEFINITIONS")
label_rows = []
for name, keep, lab in [
    ("lt30_vs_NO (drop >30)", df["readmitted"] != ">30", (df["readmitted"] == "<30").astype(int)),
    ("any_vs_NO", np.ones(len(df), bool), (df["readmitted"] != "NO").astype(int)),
]:
    d = df[np.asarray(keep)].reset_index(drop=True)
    d["y"] = np.asarray(lab)[np.asarray(keep)]
    p = {m: oof_predict(d, m) for m in MODELS}
    boot = cluster_bootstrap(d, p, n_boot=1000)
    tab = summarise(d, p, boot).assign(label=name, n=len(d), prevalence=d["y"].mean())
    print_summary(tab, f"{name}  n={len(d)} prevalence={d['y'].mean():.3f}")
    label_rows.append(tab)
pd.concat(label_rows).to_csv(RESULTS / "p2_label_definitions.csv", index=False)

# ---- 3. case mix ----------------------------------------------------------
print("\n3. CASE MIX (reweight 40-59 to look like 80+, and the reverse)")
pair = df["age_band"].isin([b1, b2]).values
d = df[pair].reset_index(drop=True)
d["y_old"] = (d["age_band"] == b2).astype(int)
num_noage = [c for c in NUM_COLS if c != "age_mid"]
# cross-fitted propensity P(80+ | features other than age), GBM
folds = cv_folds(d.assign(y=d["y_old"]))
prop = oof_predict(d, "gbm", y=d["y_old"].values, folds=folds, num=num_noage, cat=CAT_COLS,
                   X=d[num_noage + CAT_COLS])
prop_auc = FastAUC(d["y_old"].values, prop)(np.ones(len(d)))
print(f"   propensity model AUC (how separable 80+ is from 40-59 without age): {prop_auc:.3f}")
odds = prop / (1 - prop)
young = (d["age_band"] == b1).values
w_y = np.where(young, odds, 0.0)
w_y = np.minimum(w_y, np.quantile(w_y[young], 0.99))
w_o = np.where(~young, 1 / odds, 0.0)
w_o = np.minimum(w_o, np.quantile(w_o[~young], 0.99))
ess = lambda w: w.sum() ** 2 / (w ** 2).sum()
print(f"   ESS 40-59 reweighted: {ess(w_y[young]):.0f} of {young.sum()};"
      f"  80+ reweighted: {ess(w_o[~young]):.0f} of {(~young).sum()}")

# balance check: standardised mean differences before/after for key numerics
bal = []
for c in num_noage:
    xo, xy = d.loc[~young, c].values, d.loc[young, c].values
    sd = np.sqrt((xo.var() + xy.var()) / 2)
    smd_before = (xy.mean() - xo.mean()) / sd
    smd_after = (np.average(xy, weights=w_y[young]) - xo.mean()) / sd
    bal.append({"feature": c, "smd_before": smd_before, "smd_after": smd_after})
bal = pd.DataFrame(bal)
print(bal.round(3).to_string(index=False))
bal.to_csv(RESULTS / "p2_casemix_balance.csv", index=False)

rng = np.random.default_rng(CFG["seed"])
cm_rows = []
yd = d["y"].values
for m in MODELS:
    p = oof.loc[pair, f"oof_{m}"].values
    fy, fo = FastAUC(yd, p, young), FastAUC(yd, p, ~young)
    one = np.ones(len(d))
    pt = {"auc_young": fy(one), "auc_old": fo(one), "auc_young_rw": fy(w_y), "auc_old_rw": fo(w_o)}
    reps = []
    for _ in range(CFG["n_boot"]):
        c = np.bincount(rng.integers(0, len(d), len(d)), minlength=len(d)).astype(float)
        reps.append({"d": fy(c) - fo(c), "d_rw_young": fy(c * w_y) - fo(c),
                     "d_rw_old": fy(c) - fo(c * w_o)})
    reps = pd.DataFrame(reps)
    row = {"model": m, **pt,
           "delta": pt["auc_young"] - pt["auc_old"],
           "delta_young_rw": pt["auc_young_rw"] - pt["auc_old"],
           "delta_old_rw": pt["auc_young"] - pt["auc_old_rw"]}
    for k, col in [("delta", "d"), ("delta_young_rw", "d_rw_young"), ("delta_old_rw", "d_rw_old")]:
        row[k + "_lo"], row[k + "_hi"] = ci(reps[col])
    cm_rows.append(row)
    print(f"   {m}: AUC 40-59 {pt['auc_young']:.3f} -> reweighted to 80+ case mix {pt['auc_young_rw']:.3f}"
          f" | AUC 80+ {pt['auc_old']:.3f} -> reweighted to 40-59 mix {pt['auc_old_rw']:.3f}")
    print(f"        delta unweighted {fmt(row['delta'], row['delta_lo'], row['delta_hi'])}"
          f" | 40-59 reweighted {fmt(row['delta_young_rw'], row['delta_young_rw_lo'], row['delta_young_rw_hi'])}"
          f" | 80+ reweighted {fmt(row['delta_old_rw'], row['delta_old_rw_lo'], row['delta_old_rw_hi'])}")
pd.DataFrame(cm_rows).to_csv(RESULTS / "p2_casemix.csv", index=False)

# ---- 4. discharge destination ----------------------------------------------
print("\n4. DISCHARGE DESTINATION (model trained on everyone, evaluated within destination)")
dest_names = {"1": "home", "6": "home_health", "3": "SNF"}
share = pd.crosstab(df["age_band"], df["discharge_disposition_id"].map(dest_names).fillna("other"),
                    normalize="index").loc[BANDS]
print("   share of each band by destination\n", share.round(3).to_string())
dest_rows = []
for code, name in dest_names.items():
    sel = (df["discharge_disposition_id"] == code).values
    d = df[sel].reset_index(drop=True)
    p = {m: preds[m][sel] for m in MODELS}
    boot = cluster_bootstrap(d, p, n_boot=1000)
    tab = summarise(d, p, boot).assign(destination=name, n=len(d), prevalence=d["y"].mean())
    for b in BANDS:
        tab[f"n_{b}"] = int((d["age_band"] == b).sum())
    print_summary(tab, f"{name}  n={len(d)}  prevalence={d['y'].mean():.3f}  "
                       + " ".join(f"{b}:{int((d['age_band'] == b).sum())}" for b in BANDS))
    dest_rows.append(tab)
pd.concat(dest_rows).to_csv(RESULTS / "p2_destination.csv", index=False)
share.to_csv(RESULTS / "p2_destination_share.csv")

# ---- 5. calibration and risk spread ----------------------------------------
print("\n5. CALIBRATION AND SPREAD OF PREDICTED RISK")


def calib(yv, pv):
    lp = np.log(pv / (1 - pv)).reshape(-1, 1)
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(lp, yv).coef_[0, 0]
    # calibration-in-the-large: intercept with logit(p) as offset, solved by Newton steps
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp[:, 0] + a)))
        a += (yv - q).sum() / (q * (1 - q)).sum()
    bins = np.clip((pv * 10).astype(int), 0, 9)
    ece = sum(abs(yv[bins == k].mean() - pv[bins == k].mean()) * (bins == k).mean()
              for k in range(10) if (bins == k).any())
    return slope, a, ece, lp[:, 0].std()


cal_rows = []
for m in MODELS:
    for b in BANDS + ["overall"]:
        msk = np.ones(len(df), bool) if b == "overall" else (df["age_band"] == b).values
        s, a, e, sd = calib(y[msk], preds[m][msk])
        reps = []
        for _ in range(300):
            i = rng.choice(np.flatnonzero(msk), msk.sum())
            reps.append(calib(y[i], preds[m][i])[0])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        cal_rows.append({"model": m, "band": b, "slope": s, "slope_lo": lo, "slope_hi": hi,
                         "intercept": a, "ece": e, "sd_logit": sd,
                         "mean_pred": preds[m][msk].mean(), "observed": y[msk].mean()})
cal = pd.DataFrame(cal_rows)
print(cal.round(3).to_string(index=False))
cal.to_csv(RESULTS / "p2_calibration.csv", index=False)
print("\nsaved results/p2_*.csv")
