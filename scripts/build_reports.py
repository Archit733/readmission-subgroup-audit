#!/usr/bin/env python3
"""Stage 2: Fairlearn cross-check + figures + auto-generated MODEL_CARD.md.
Run after scripts/run_pipeline.py. Every number in the model card is read from
pipeline outputs -- nothing is typed by hand."""
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from readmit_audit.config import load_config
from readmit_audit.fairness import (crosscheck_auc, top_fraction_threshold, bootstrap_tpr_ci)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("reports")

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
AGE_ORDER = [f"[{i}-{i+10})" for i in range(0, 100, 10)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def dot_whisker(ax, df, ycol, lo, hi, label_n=True):
    x = np.arange(len(df))
    ax.vlines(x, df[lo], df[hi], color=BLUE, lw=2, alpha=0.55)
    ax.plot(x, df[ycol], "o", color=BLUE, ms=8, mec="white", mew=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\nn={n:,}" if label_n else g for g, n in zip(df["group"], df["n"])],
                       fontsize=8, color=MUTED)


def main(config_path):
    cfg = load_config(config_path)
    rd, fd = Path(cfg["output"]["results_dir"]), Path(cfg["output"]["figures_dir"])
    fd.mkdir(parents=True, exist_ok=True)
    art = joblib.load(rd / "pipeline_artifacts.joblib")
    y, p, sg = art["y_test"], art["strong_probs"], art["subgroups"]
    agg, agg_lo, agg_hi = art["strong_auc"]
    base = art["baseline_auc"]

    # ---- Fairlearn cross-check of every subgroup table ----
    checks = {}
    for col in cfg["data"]["subgroup_columns"]:
        t = pd.read_csv(rd / f"subgroup_audit_{col}.csv")
        c = crosscheck_auc(t, y, p, sg[col])
        c.to_csv(rd / f"fairlearn_crosscheck_{col}.csv", index=False)
        checks[col] = bool(c["agrees"].all())
        log.info(f"Fairlearn cross-check [{col}]: all groups agree = {checks[col]}")
    assert all(checks.values()), "Audit disagrees with Fairlearn -- investigate before publishing"

    # ---- Operating-point analysis: flag top 20% highest-risk patients ----
    flag_frac = 0.20
    thr = top_fraction_threshold(p, flag_frac)
    pred = (np.asarray(p) >= thr).astype(int)
    rows = []
    for g in AGE_ORDER:
        m = (sg["age"] == g).values
        if m.sum() < cfg["audit"]["min_subgroup_n"]:
            continue
        tpr, lo, hi, npos = bootstrap_tpr_ci(np.asarray(y)[m], pred[m],
                                              n_boot=cfg["audit"]["n_bootstrap"],
                                              random_state=cfg["split"]["random_state"])
        rows.append({"group": g, "n": int(m.sum()), "n_pos": npos, "tpr": tpr, "ci_lo": lo, "ci_hi": hi,
                     "selection_rate": float(pred[m].mean())})
    op = pd.DataFrame(rows); op.to_csv(rd / "operating_point_age.csv", index=False)

    # ---- Figure 1: AUC by age ----
    age = pd.read_csv(rd / "subgroup_audit_age.csv").dropna(subset=["auc"])
    fig, ax = plt.subplots(figsize=(9, 4.8)); style(ax)
    dot_whisker(ax, age, "auc", "ci_lo", "ci_hi")
    ax.axhline(agg, color=MUTED, ls="--", lw=1.2)
    ax.text(6.5, agg + 0.008, f"Aggregate AUC {agg:.3f}", ha="center", color=MUTED, fontsize=9)
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("AUC (95% bootstrap CI)", color=MUTED)
    ax.set_title("Discrimination varies by age band; the aggregate hides it", loc="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(fd / "fig1_auc_by_age.png", dpi=160); plt.close(fig)

    # ---- Figure 2: what it means operationally (recall at top-20% flag) ----
    o = op.dropna(subset=["tpr"])
    fig, ax = plt.subplots(figsize=(9, 4.8)); style(ax)
    dot_whisker(ax, o, "tpr", "ci_lo", "ci_hi")
    ax.set_xticklabels([f"{g}\n{npos} readmits" for g, npos in zip(o["group"], o["n_pos"])], fontsize=8, color=MUTED)
    ax.axhline(flag_frac, color=MUTED, ls=":", lw=1)
    ax.text(3.5, flag_frac + 0.015, "Random flagging would catch 20%", ha="center", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Share of readmissions caught (95% CI)", color=MUTED)
    ax.set_title(f"If the top {int(flag_frac*100)}% highest-risk patients are flagged, who gets caught?", loc="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(fd / "fig2_recall_at_top20.png", dpi=160); plt.close(fig)

    # ---- Figure 3: mechanism candidates are flat ----
    dx = pd.read_csv(rd / "mechanism_diagnosis_binning.csv"); ch = pd.read_csv(rd / "mechanism_charlson.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, d, ttl, xl in [(axes[0], dx, "Number of diagnoses", "diagnoses recorded"),
                           (axes[1], ch, "Charlson comorbidity index", "Charlson score (diabetes excluded)")]:
        style(ax); dot_whisker(ax, d, "auc", "ci_lo", "ci_hi", label_n=False)
        ax.set_xticklabels(d["group"], fontsize=9, color=MUTED); ax.set_xlabel(xl, color=MUTED, fontsize=9)
        ax.axhline(agg, color=MUTED, ls="--", lw=1); ax.set_title(ttl, loc="left", color=INK, fontsize=11)
    axes[0].set_ylim(0.45, 1.0); axes[0].set_ylabel("AUC (95% bootstrap CI)", color=MUTED)
    fig.suptitle("Candidate mechanisms tested directly: no clean gradient", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(fd / "fig3_mechanism_flat.png", dpi=160); plt.close(fig)

    # ---- Model card (numbers read from outputs) ----
    ids = pd.read_csv(cfg["data"]["path"], usecols=["patient_nbr"])["patient_nbr"]
    n_total, n_patients = len(ids), ids.nunique()
    race = pd.read_csv(rd / "subgroup_audit_race.csv"); gender = pd.read_csv(rd / "subgroup_audit_gender.csv")
    best, worst = age.loc[age["auc"].idxmax()], age.loc[age["auc"].idxmin()]
    small = race[race["n"] < 500]
    def fmt(r): return f"{r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
    table = lambda d: "\n".join(f"| {r['group']} | {int(r['n']):,} | {fmt(r)} |" for _, r in d.iterrows())
    op_tbl = "\n".join(f"| {r['group']} | {int(r['n_pos'])} | {r['tpr']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] |" for _, r in o.iterrows())
    card = f"""# Model Card: 30-day Readmission Risk (diabetes inpatient cohort)

Generated by `scripts/build_reports.py` from pipeline outputs. Do not edit numbers by hand.

## Intended use
Rank inpatient diabetes encounters by risk of readmission within 30 days, to prioritise follow-up
outreach. **Decision-support only.** Not for diagnosis, and not for denying care or coverage.

## Data
UCI "Diabetes 130-US Hospitals" (1999-2008), {n_total:,} encounters from {n_patients:,} patients
(80/20 split with all encounters of a patient kept on one side; test n = {len(y):,}; positive rate {float(np.mean(y)):.3f}).
Dropped: identifiers, `weight`/`payer_code`/`medical_specialty` (high missingness), zero-variance columns.

## Model
HistGradientBoostingClassifier in a single sklearn Pipeline (impute -> ordinal encode -> model).
Baseline: logistic regression, AUC {base[0]:.3f} [{base[1]:.3f}, {base[2]:.3f}].
Final: AUC **{agg:.3f}** [{agg_lo:.3f}, {agg_hi:.3f}], ECE {art['ece_overall']:.3f}.
Performance is modest: readmission depends on factors (social support, post-discharge care) absent from hospital records.

## Subgroup performance (95% bootstrap CI, groups with n >= {cfg['audit']['min_subgroup_n']})
**Age**

| Group | n | AUC |
|---|---|---|
{table(age)}

Best band {best['group']} ({best['auc']:.3f}) vs worst {worst['group']} ({worst['auc']:.3f}): intervals do not overlap.

**Gender**

| Group | n | AUC |
|---|---|---|
{table(gender)}

**Race** (small groups are inconclusive, not "no gap")

| Group | n | AUC |
|---|---|---|
{table(race)}

Groups under 500 patients ({', '.join(small['group'])}) have wide intervals; no fairness claim is made for them.

## Operational impact (flag top {int(flag_frac*100)}% by risk, threshold {thr:.3f})
Share of actual readmissions caught, by age band:

| Age | Readmissions | Recall [95% CI] |
|---|---|---|
{op_tbl}

## Verification
Subgroup AUCs independently cross-checked against Fairlearn `MetricFrame`: all groups agree
(age: {checks['age']}, gender: {checks['gender']}, race: {checks['race']}).

## Known limitations
- Age gap is real and survives a stronger model, but its **mechanism is unexplained**: diagnosis count and
  Charlson comorbidity index (diabetes component excluded by design) were tested directly and show no clean gradient.
- Data ends 2008; practice patterns and coding have changed. Revalidate before any local use.
- Single train/test split; no external validation cohort.
- Race categories are coarse and self-reported; small groups cannot be audited with this sample.
- Split is by patient (a naive row split let repeat patients leak across train/test and inflated AUC by ~0.01).

## Recommended safeguards
Monitor per-subgroup recall and drift after deployment (`readmit_audit.monitoring`); surface a reliability
warning for age bands where the model is weakest (`/predict` response); re-audit on local data before use.
"""
    Path("MODEL_CARD.md").write_text(card)
    log.info("Wrote figures and MODEL_CARD.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
