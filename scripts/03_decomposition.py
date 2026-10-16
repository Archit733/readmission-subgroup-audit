"""Phase 3: bias / variance / noise decomposition of the age gap (Chen, Johansson & Sontag 2018).

1. Variance: learning curves and a sample-size-matched comparison.
2. Bias:     model-class comparison incl. explicit age x feature interactions.
3. Noise:    within-band k-NN separability at matched n.
4. Which signals carry discrimination in each band: per-band permutation importance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (BANDS, CAT_COLS, CFG, NUM_COLS, RESULTS, FastAUC, band_table, build_cohort,
                    cluster_bootstrap, cv_folds, feature_frame, fit_model, load_raw, make_model,
                    oof_predict, print_summary, summarise)

b1, b2 = CFG["delta_pair"]
raw = load_raw()
df = build_cohort(raw)
y = df["y"].values
X = feature_frame(df)
folds = cv_folds(df)
rng = np.random.default_rng(CFG["seed"])
one = np.ones(len(df))
part = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------------------------------------------------------------- 1. variance
if part in ("all", "lc"):
    print("1a. LEARNING CURVES (fraction of each training fold; per-band AUC of pooled OOF predictions)")
    fracs = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rows = []
    for m in ["lr", "gbm"]:
        for f in fracs:
            for r in range(1 if f == 1.0 else 3):
                oof = np.full(len(df), np.nan)
                for tr, te in folds:
                    sub = rng.choice(tr, int(f * len(tr)), replace=False) if f < 1 else tr
                    mod = make_model(m)
                    fit_model(mod, X.iloc[sub], y[sub])
                    oof[te] = mod.predict_proba(X.iloc[te])[:, 1]
                bt = band_table(df, oof)
                rows.append({"model": m, "frac": f, "rep": r, "n_train": int(f * len(folds[0][0])), **bt})
            last = pd.DataFrame(rows).query("model == @m and frac == @f")[BANDS + ["overall"]].mean()
            print(f"   {m} frac={f:<4} " + "  ".join(f"{b}:{last[b]:.3f}" for b in BANDS + ["overall"]))
    lc = pd.DataFrame(rows)
    lc.to_csv(RESULTS / "p3_learning_curves.csv", index=False)

    # power-law fit AUC(n) = a - b * n^-c per band, plus gain between fraction 0.5 and 1.0
    fits = []
    g = lc.groupby(["model", "frac"])[BANDS].mean().reset_index()
    for m in ["lr", "gbm"]:
        gm = g[g["model"] == m]
        n = gm["frac"].values * len(folds[0][0])
        for b in BANDS:
            a_hat = np.nan
            try:
                popt, _ = curve_fit(lambda n, a, bb, c: a - bb * n ** (-c), n, gm[b].values,
                                    p0=[gm[b].max(), 1.0, 0.5], bounds=([0.4, 0, 0.01], [1.0, 1e4, 2.0]),
                                    maxfev=20000)
                a_hat = popt[0]
            except Exception:
                pass
            gain = gm.loc[gm["frac"] == 1.0, b].values[0] - gm.loc[gm["frac"] == 0.5, b].values[0]
            fits.append({"model": m, "band": b, "asymptote": a_hat, "gain_0.5_to_1.0": gain,
                         "auc_full": gm.loc[gm["frac"] == 1.0, b].values[0]})
    fits = pd.DataFrame(fits)
    print(fits.round(4).to_string(index=False))
    fits.to_csv(RESULTS / "p3_learning_curve_fits.csv", index=False)

    print("\n1b. SAMPLE-SIZE-MATCHED BAND-SPECIFIC MODELS (train and test within band, equal n)")
    n_match = int((df["age_band"] == b2).sum())
    rows = []
    for m in ["lr", "gbm"]:
        for b in [b1, "60-79", b2]:
            idx_all = np.flatnonzero((df["age_band"] == b).values)
            reps = 5 if len(idx_all) > n_match else 1
            for r in range(reps):
                idx = rng.choice(idx_all, n_match, replace=False) if reps > 1 else idx_all
                d = df.iloc[idx].reset_index(drop=True)
                skf = StratifiedKFold(5, shuffle=True, random_state=CFG["seed"] + r)
                p = oof_predict(d, m, folds=list(skf.split(np.zeros(len(d)), d["y"])))
                rows.append({"model": m, "band": b, "rep": r, "n": len(d),
                             "auc": FastAUC(d["y"].values, p)(np.ones(len(d)))})
    mt = pd.DataFrame(rows)
    print(mt.groupby(["model", "band"])["auc"].agg(["mean", "std", "count"]).round(4).to_string())
    mt.to_csv(RESULTS / "p3_matched_n.csv", index=False)

# ---------------------------------------------------------------- 2. bias
if part in ("all", "bias"):
    print("\n2. MODEL CLASS")
    d = df.copy()
    inter_num, inter_cat = [], []
    for b in BANDS:
        ind = (d["age_band"] == b).astype(float)
        for c in [c for c in NUM_COLS if c != "age_mid"]:
            d[f"{c}_x_{b}"] = d[c] * ind
            inter_num.append(f"{c}_x_{b}")
    for c in ["discharge_disposition_id", "admission_source_id", "admission_type_id", "dg1", "dg2",
              "dg3", "specialty", "insulin", "A1Cresult", "change"]:
        d[f"{c}_x_band"] = d["age_band"] + "|" + d[c]
        inter_cat.append(f"{c}_x_band")
    num_i, cat_i = NUM_COLS + inter_num, CAT_COLS + inter_cat
    oof = pd.read_csv(RESULTS / "oof_primary.csv")
    preds = {"lr": oof["oof_lr"].values, "gbm": oof["oof_gbm"].values}
    preds["lr_interact"] = oof_predict(d, "lr", X=d[num_i + cat_i], num=num_i, cat=cat_i)
    preds["gbm_big"] = oof_predict(df, "gbm_big")
    boot = cluster_bootstrap(df, preds)
    tab = summarise(df, preds, boot)
    print_summary(tab, "model class")
    # paired differences in delta relative to LR
    comp = []
    for m in preds:
        dd = (boot[(m, b1)] - boot[(m, b2)]) - (boot[("lr", b1)] - boot[("lr", b2)])
        d80 = boot[(m, b2)] - boot[("lr", b2)]
        comp.append({"model": m, "delta_minus_lr_delta": dd.mean(),
                     "lo": dd.quantile(.025), "hi": dd.quantile(.975),
                     "auc80_gain_vs_lr": d80.mean(), "auc80_gain_lo": d80.quantile(.025),
                     "auc80_gain_hi": d80.quantile(.975)})
    comp = pd.DataFrame(comp)
    print(comp.round(4).to_string(index=False))
    tab.to_csv(RESULTS / "p3_model_class.csv", index=False)
    comp.to_csv(RESULTS / "p3_model_class_paired.csv", index=False)
    pd.DataFrame(preds).assign(patient_nbr=df["patient_nbr"]).to_csv(RESULTS / "oof_model_class.csv", index=False)

# ---------------------------------------------------------------- 3. noise
if part in ("all", "knn"):
    print("\n3. k-NN SEPARABILITY WITHIN BAND AT MATCHED n (model-light estimate of how much the features"
          " separate outcomes)")
    num = [c for c in NUM_COLS if c != "age_mid"]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                                   sparse_output=False), CAT_COLS)])
    rows = []
    for n_match, bands in [(int((df["age_band"] == b2).sum()), [b1, "60-79", b2]),
                           (int((df["age_band"] == "<40").sum()), BANDS)]:
        for b in bands:
            idx_all = np.flatnonzero((df["age_band"] == b).values)
            reps = 3 if len(idx_all) > n_match else 1
            for r in range(reps):
                idx = rng.choice(idx_all, n_match, replace=False) if reps > 1 else idx_all
                d = df.iloc[idx].reset_index(drop=True)
                yy = d["y"].values
                skf = StratifiedKFold(5, shuffle=True, random_state=CFG["seed"] + r)
                for k in [10, 25, 50, 100]:
                    p = np.zeros(len(d))
                    for tr, te in skf.split(np.zeros(len(d)), yy):
                        P = pre.fit(d.iloc[tr][num + CAT_COLS])
                        knn = KNeighborsClassifier(k, n_jobs=2).fit(P.transform(d.iloc[tr][num + CAT_COLS]), yy[tr])
                        p[te] = knn.predict_proba(P.transform(d.iloc[te][num + CAT_COLS]))[:, 1]
                    rows.append({"n_match": n_match, "band": b, "rep": r, "k": k,
                                 "knn_auc": FastAUC(yy, p)(np.ones(len(d)))})
    kn = pd.DataFrame(rows)
    print(kn.groupby(["n_match", "k", "band"])["knn_auc"].mean().unstack().round(3).to_string())
    kn.to_csv(RESULTS / "p3_knn.csv", index=False)

# ---------------------------------------------------------------- 4. importance
if part in ("all", "imp"):
    print("\n4. PER-BAND PERMUTATION IMPORTANCE (GBM; drop in within-band AUC when one feature is shuffled)")
    feats = NUM_COLS + CAT_COLS
    rows = []
    for fi, (tr, te) in enumerate(folds):
        mod = make_model("gbm")
        fit_model(mod, X.iloc[tr], y[tr])
        Xte = X.iloc[te].reset_index(drop=True)
        yte = y[te]
        bte = df["age_band"].values[te]
        base_p = mod.predict_proba(Xte)[:, 1]
        base = {b: FastAUC(yte, base_p, bte == b)(np.ones(len(te))) for b in BANDS}
        for f in feats:
            for r in range(2):
                Xp = Xte.copy()
                # shuffle within band so the feature's band-specific distribution is kept
                for b in BANDS:
                    ii = np.flatnonzero(bte == b)
                    Xp.loc[ii, f] = Xte.loc[rng.permutation(ii), f].values
                pp = mod.predict_proba(Xp)[:, 1]
                for b in BANDS:
                    rows.append({"fold": fi, "feature": f, "rep": r, "band": b,
                                 "drop": base[b] - FastAUC(yte, pp, bte == b)(np.ones(len(te)))})
    imp = pd.DataFrame(rows).groupby(["feature", "band"])["drop"].mean().unstack()[BANDS]
    imp = imp.sort_values(b1, ascending=False)
    imp["sum_all"] = imp[BANDS].sum(axis=1)
    print(imp.head(15).round(4).to_string())
    print("   total importance mass per band:", imp[BANDS].clip(lower=0).sum().round(4).to_dict())
    imp.to_csv(RESULTS / "p3_importance.csv")

    print("\n   single-feature AUC within band (direction-free, |AUC-0.5|+0.5)")
    sf = []
    for f in ["number_inpatient", "number_emergency", "number_outpatient", "time_in_hospital",
              "num_medications", "number_diagnoses", "num_lab_procedures"]:
        row = {"feature": f}
        for b in BANDS:
            msk = (df["age_band"] == b).values
            a = FastAUC(y, df[f].values.astype(float), msk)(one)
            row[b] = abs(a - 0.5) + 0.5
        sf.append(row)
    sf = pd.DataFrame(sf)
    print(sf.round(3).to_string(index=False))
    sf.to_csv(RESULTS / "p3_single_feature_auc.csv", index=False)

    print("\n   prior inpatient visits by band (share with >=1, mean)")
    print(df.groupby("age_band")["number_inpatient"].agg(
        share_ge1=lambda s: (s >= 1).mean(), mean="mean").loc[BANDS].round(3).to_string())
print("\nsaved results/p3_*.csv")
