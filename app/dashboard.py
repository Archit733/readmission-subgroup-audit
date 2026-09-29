"""Streamlit dashboard for exploring the audit. Run from the project root:
    streamlit run app/dashboard.py
Reads outputs of scripts/run_pipeline.py, build_reports.py, simulate_monitoring.py."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from readmit_audit.fairness import bootstrap_tpr_ci, top_fraction_threshold
from readmit_audit.serving import AGE_BAND_ORDER, ReadmissionScorer

RES = Path(os.environ.get("READMIT_RESULTS_DIR", ROOT / "reports" / "results"))
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"

st.set_page_config(page_title="Readmission subgroup audit", layout="wide")


@st.cache_data
def load_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_artifacts():
    p = RES / "pipeline_artifacts.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def load_scorer():
    p = RES / "model_for_serving.joblib"
    return ReadmissionScorer.from_path(p) if p.exists() else None


def dot_whisker(df, ycol, lo, hi, labels, ylabel, hline=None, hline_label=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    x = np.arange(len(df))
    ax.vlines(x, df[lo], df[hi], color=BLUE, lw=2, alpha=0.55)
    ax.plot(x, df[ycol], "o", color=BLUE, ms=8, mec="white", mew=1.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if hline is not None:
        ax.axhline(hline, color=MUTED, ls="--", lw=1)
        ax.text(1.01, hline, hline_label, transform=ax.get_yaxis_transform(), ha="left", va="center",
                color=MUTED, fontsize=8, clip_on=False)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    return fig


art = load_artifacts()
st.title("Readmission risk: subgroup audit")
st.caption("Does one aggregate number hide who the model fails? Decision-support research tool, not for clinical use.")

if art is None:
    st.error("No results found. Run `python scripts/run_pipeline.py` first.")
    st.stop()

agg, agg_lo, agg_hi = art["strong_auc"]
tabs = st.tabs(["Overview", "Subgroup audit", "Operating point", "Mechanism tests", "Monitoring", "Score a patient"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aggregate AUC", f"{agg:.3f}")
    c1.caption(f"95% CI {agg_lo:.3f} to {agg_hi:.3f}")
    c2.metric("Baseline (logistic) AUC", f"{art['baseline_auc'][0]:.3f}")
    c3.metric("Calibration error (ECE)", f"{art['ece_overall']:.3f}")
    c4.metric("Test encounters", f"{len(art['y_test']):,}")
    age = load_csv("subgroup_audit_age.csv").dropna(subset=["auc"])
    best, worst = age.loc[age["auc"].idxmax()], age.loc[age["auc"].idxmin()]
    st.info(f"**The aggregate hides a gap.** Age band {best['group']} AUC {best['auc']:.2f} vs {worst['group']} "
            f"AUC {worst['auc']:.2f}. Intervals do not overlap, and the gap persists with a stronger model. "
            "The mechanism is unexplained: diagnosis count and Charlson index were tested and do not account for it.")
    _fig1 = ROOT / "reports" / "figures" / "fig1_auc_by_age.png"
    if _fig1.exists():
        st.image(str(_fig1))

with tabs[1]:
    axis = st.radio("Subgroup axis", ["age", "gender", "race"], horizontal=True)
    t = load_csv(f"subgroup_audit_{axis}.csv").dropna(subset=["auc"])
    cc = load_csv(f"fairlearn_crosscheck_{axis}.csv")
    st.pyplot(dot_whisker(t, "auc", "ci_lo", "ci_hi", [f"{g}\nn={n:,}" for g, n in zip(t["group"], t["n"])],
                          "AUC (95% bootstrap CI)", agg, f"aggregate {agg:.3f}", (0.4, 1.0)))
    small = t[t["n"] < 500]
    if len(small):
        st.warning("Small groups (" + ", ".join(small["group"]) + ") have wide intervals. That is inconclusive, not 'no gap'.")
    st.dataframe(t.round(3), hide_index=True, use_container_width=True)
    if cc is not None:
        st.success(f"Independent Fairlearn cross-check: all {len(cc)} groups agree = {bool(cc['agrees'].all())}")

with tabs[2]:
    frac = st.slider("Flag this share of highest-risk patients", 0.05, 0.50, 0.20, 0.05)
    y, p, sg = np.asarray(art["y_test"]), art["strong_probs"], art["subgroups"]["age"]
    thr = top_fraction_threshold(p, frac); pred = (p >= thr).astype(int)
    rows = []
    for g in AGE_BAND_ORDER:
        m = (sg == g).values
        if m.sum() < 100:
            continue
        tpr, lo, hi, npos = bootstrap_tpr_ci(y[m], pred[m], n_boot=300)
        rows.append({"group": g, "n": int(m.sum()), "readmits": npos, "recall": tpr, "ci_lo": lo, "ci_hi": hi})
    op = pd.DataFrame(rows).dropna(subset=["recall"])
    st.pyplot(dot_whisker(op, "recall", "ci_lo", "ci_hi", [f"{g}\n{r} readmits" for g, r in zip(op["group"], op["readmits"])],
                          "Share of readmissions caught (95% CI)", frac, f"random = {frac:.0%}", (0, 1)))
    st.caption(f"Risk-score threshold at this flag rate: {thr:.3f}. Recall is the share of patients who were "
               "actually readmitted that the flag would have caught.")

with tabs[3]:
    st.write("Candidate explanations for the age gap, tested directly (not through the age proxy). Neither shows a clean gradient.")
    for name, f in [("Number of diagnoses", "mechanism_diagnosis_binning.csv"), ("Charlson comorbidity index (diabetes excluded)", "mechanism_charlson.csv")]:
        d = load_csv(f)
        if d is not None:
            st.subheader(name)
            st.pyplot(dot_whisker(d.dropna(subset=["auc"]), "auc", "ci_lo", "ci_hi", d.dropna(subset=["auc"])["group"].tolist(),
                                  "AUC (95% CI)", agg, "aggregate", (0.4, 1.0)))

with tabs[4]:
    perf, drift = load_csv("monitoring_performance.csv"), load_csv("monitoring_drift.csv")
    if perf is None:
        st.info("Run `python scripts/simulate_monitoring.py` to populate this tab.")
    else:
        st.write("Simulated stream: batches 0-3 clean, then increasing synthetic drift toward older patients with "
                 "growing label noise for age 70+. Input drift (PSI) rises first; outcome-based alerts lag because "
                 "readmission labels only arrive 30 days later.")
        fig_path = ROOT / "reports" / "figures" / "fig4_monitoring_simulation.png"
        if fig_path.exists():
            st.image(str(fig_path))
        st.dataframe(drift.round(3), hide_index=True, use_container_width=True)
        st.dataframe(perf[perf["alert"]].round(3), hide_index=True, use_container_width=True)

with tabs[5]:
    scorer = load_scorer()
    st.write("Illustration of the API's behaviour. Unspecified fields are imputed. **Not for clinical use.**")
    c1, c2, c3 = st.columns(3)
    age_band = c1.selectbox("Age band", AGE_BAND_ORDER, index=7)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    race = c2.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    inpatient = c2.number_input("Prior inpatient visits (past year)", 0, 20, 0)
    n_diag = c3.number_input("Number of diagnoses", 1, 16, 8)
    stay = c3.number_input("Days in hospital", 1, 14, 4)
    if st.button("Score"):
        r = scorer.score([{"age": age_band, "gender": gender, "race": race, "number_inpatient": inpatient,
                           "number_diagnoses": n_diag, "time_in_hospital": stay}])[0]
        st.metric("Readmission probability (30 days)", f"{r['readmission_probability']:.1%}")
        st.write("Flagged high-risk:", "yes" if r["flagged_high_risk"] else "no")
        lvl = r["reliability"]["level"]
        (st.warning if lvl in ("reduced", "insufficient_data", "unknown") else st.success)(
            f"Reliability for this age band: **{lvl}**. {r['reliability']['note']}")
