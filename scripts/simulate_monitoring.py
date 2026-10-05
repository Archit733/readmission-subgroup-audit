#!/usr/bin/env python3
"""Run the monitoring layer against a simulated drifting stream (see readmit_audit/simulation.py).
Requires reports/results from scripts/run_pipeline.py."""
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from readmit_audit.config import load_config
from readmit_audit.monitoring import batch_subgroup_monitor, feature_drift_report, PSI_STABLE, PSI_MAJOR
from readmit_audit.simulation import make_stream

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("monitor-sim")
BLUE, INK, MUTED, GRID, ALERT = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#c0392b"


def coarse_age(a):
    return "70+" if a in ("[70-80)", "[80-90)", "[90-100)") else ("50-69" if a in ("[50-60)", "[60-70)") else "<50")


def main(config_path, n_batches=10, clean_batches=4, window=3):
    cfg = load_config(config_path)
    rd, fd = Path(cfg["output"]["results_dir"]), Path(cfg["output"]["figures_dir"])
    art = joblib.load(rd / "pipeline_artifacts.joblib"); serv = joblib.load(rd / "model_for_serving.joblib")
    y, p, X = np.asarray(art["y_test"]), art["strong_probs"], art["X_test"].reset_index(drop=True)
    grp = art["subgroups"]["age"].map(coarse_age)

    ys, ps, Xs, gs, bs, meta = make_stream(y, p, X, grp, target_group="70+", n_batches=n_batches,
                                           clean_batches=clean_batches, seed=cfg["split"]["random_state"])
    # Reference = performance on the clean batches (what a team would have observed at launch)
    clean = bs < clean_batches
    reference = {g: roc_auc_score(ys[clean & (gs == g)], ps[clean & (gs == g)]) for g in np.unique(gs)}
    reference["__overall__"] = roc_auc_score(ys[clean], ps[clean])
    log.info("Reference AUC (clean batches): " + ", ".join(f"{k}={v:.3f}" for k, v in reference.items()))

    perf = batch_subgroup_monitor(ys, ps, gs, bs, reference, min_n=150, n_boot=300,
                                  random_state=cfg["split"]["random_state"], window=window)
    perf = perf.merge(meta, on="batch"); perf.to_csv(rd / "monitoring_performance.csv", index=False)

    drift_rows = []
    for b in range(n_batches):
        rep = feature_drift_report(serv["reference"], Xs[bs == b])
        age_psi = float(rep.loc[rep["feature"] == "age", "psi"].iloc[0])
        drift_rows.append({"batch": b, "age_psi": age_psi, "max_psi": float(rep["psi"].max()),
                           "max_psi_feature": rep.iloc[0]["feature"], "n_major": int((rep["status"] == "major").sum())})
    drift = pd.DataFrame(drift_rows).merge(meta, on="batch"); drift.to_csv(rd / "monitoring_drift.csv", index=False)

    print("\nDrift (PSI vs training reference):"); print(drift.round(3).to_string(index=False))
    print("\nPerformance alerts (batch AUC interval entirely below reference):")
    al = perf[perf["alert"]]
    print(al[["batch", "group", "n", "auc", "reference_auc", "severity"]].round(3).to_string(index=False) if len(al) else "none")

    # ---- Figure ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax in (a1, a2):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
        ax.axvspan(-0.5, clean_batches - 0.5, color=GRID, alpha=0.5, lw=0)
        ax.set_xlabel("batch (shaded = clean launch window)", color=MUTED, fontsize=9)
    a1.plot(drift["batch"], drift["age_psi"], "o-", color=BLUE, lw=2, ms=7, mec="white")
    a1.axhline(PSI_STABLE, color=MUTED, ls=":", lw=1); a1.axhline(PSI_MAJOR, color=MUTED, ls="--", lw=1)
    a1.text(n_batches - 1, PSI_STABLE + 0.01, "moderate", ha="right", color=MUTED, fontsize=8)
    a1.text(n_batches - 1, PSI_MAJOR + 0.01, "major", ha="right", color=MUTED, fontsize=8)
    a1.set_ylabel("PSI of age mix vs training", color=MUTED); a1.set_title("Input drift", loc="left", color=INK, fontsize=11)
    for g, col, lab in [("__overall__", MUTED, "Overall"), ("70+", BLUE, "Age 70+")]:
        d = perf[perf["group"] == g]
        a2.vlines(d["batch"], d["ci_lo"], d["ci_hi"], color=col, lw=2, alpha=0.4)
        a2.plot(d["batch"], d["auc"], "o-", color=col, lw=2, ms=7, mec="white", label=lab)
        a2.axhline(reference[g], color=col, ls=":", lw=1)
        bad = d[d["alert"]]
        a2.plot(bad["batch"], bad["auc"], "o", ms=14, mfc="none", mec=ALERT, mew=2)
        for _, r in bad.iterrows():
            a2.annotate("ALERT", (r["batch"], r["ci_lo"]), textcoords="offset points", xytext=(0, -14),
                        ha="center", color=ALERT, fontsize=8)
    a2.set_ylabel("AUC (95% CI), dotted = launch reference", color=MUTED)
    a2.set_title(f"Performance alerts (ringed), trailing {window}-batch window", loc="left", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(fd / "fig4_monitoring_simulation.png", dpi=160)
    log.info("Wrote monitoring_performance.csv, monitoring_drift.csv, fig4_monitoring_simulation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--window", type=int, default=3)
    a = ap.parse_args(); main(a.config, window=a.window)
