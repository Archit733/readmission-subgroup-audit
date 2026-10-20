"""Figures for the write-up, built only from results/*.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BANDS, FIGURES, RESULTS

FIGURES.mkdir(exist_ok=True)
SERIES = {"<40": "#2a78d6", "40-59": "#eb6834", "60-79": "#1baf7a", "80+": "#6250d6"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})

# ---- Figure 1: what happens to the gap as the cohort is cleaned -------------------
p1 = pd.read_csv(RESULTS / "p1_leakage.csv")
cond = ["A_row_split_all", "B_grouped_all", "C_grouped_excl", "D_primary"]
labels = ["A: row split,\nall encounters", "B: patient-grouped\nCV", "C: + death/hospice\nexcluded",
          "D: first encounter\n(primary)"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
lr = p1[p1["model"] == "lr"].set_index("condition").loc[cond]
x = np.arange(len(cond))
for i, b in enumerate(BANDS):
    off = (i - 1.5) * 0.12
    ax.errorbar(x + off, lr[b], yerr=[lr[b] - lr[b + "_lo"], lr[b + "_hi"] - lr[b]], fmt="o",
                ms=5, color=SERIES[b], elinewidth=1.4, capsize=0, label=b)
ax.set_xticks(x, labels, fontsize=7.5)
ax.set_ylabel("AUC (95% CI)")
ax.set_title("Per-band AUC, logistic regression", loc="left", fontsize=10, color=INK)
ax.legend(title="Age band", frameon=False, fontsize=8, title_fontsize=8, ncol=4, loc="lower left")
ax.set_ylim(0.53, 0.81)
ax = axes[1]
for j, (m, mk) in enumerate([("lr", "o"), ("gbm", "s")]):
    t = p1[p1["model"] == m].set_index("condition").loc[cond]
    ax.errorbar(x + (j - 0.5) * 0.18, t["delta"], yerr=[t["delta"] - t["delta_lo"], t["delta_hi"] - t["delta"]],
                fmt=mk, ms=5, color=INK2 if m == "lr" else "#2a78d6", elinewidth=1.4, capsize=0,
                label={"lr": "Logistic regression", "gbm": "LightGBM"}[m])
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x, ["A", "B", "C", "D"])
ax.set_ylabel("AUC(40-59) - AUC(80+)")
ax.set_title("Age gap with paired 95% CI", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "fig1_cohort_cleaning.png", dpi=200)

# ---- Figure 2: learning curves -------------------------------------------------
lc = pd.read_csv(RESULTS / "p3_learning_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, m, name in [(axes[0], "lr", "Logistic regression"), (axes[1], "gbm", "LightGBM")]:
    g = lc[lc["model"] == m].groupby("n_train")[BANDS].agg(["mean", "std"])
    n = g.index.values
    for b in BANDS:
        ax.plot(n, g[(b, "mean")], "-o", color=SERIES[b], lw=2, ms=4)
        nudge = {"40-59": 0.004, "60-79": -0.004}.get(b, 0)
        ax.text(n[-1] * 1.06, g[(b, "mean")].iloc[-1] + nudge, b, color=INK2, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training patients (log scale)")
    ax.set_title(name, loc="left", fontsize=10, color=INK)
    ax.set_xlim(n[0] * 0.8, n[-1] * 1.5)
axes[0].set_ylabel("Within-band AUC")
fig.suptitle("Learning curves run parallel: more data lifts every band, the gap stays",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(FIGURES / "fig2_learning_curves.png", dpi=200)

# ---- Figure 3: interventions ----------------------------------------------------
p4 = pd.read_csv(RESULTS / "p4_interventions_paired.csv")
p3 = pd.read_csv(RESULTS / "p3_model_class_paired.csv")
names = {"lr_bandspecific": "LR, band-specific models", "gbm_bandspecific": "GBM, band-specific models",
         "lr_w80x2": "LR, 80+ weighted x2", "gbm_w80x2": "GBM, 80+ weighted x2",
         "lr_w80x5": "LR, 80+ weighted x5", "gbm_w80x5": "GBM, 80+ weighted x5",
         "lr_richICD": "LR, 3-digit ICD codes", "gbm_richICD": "GBM, 3-digit ICD codes",
         "lr_geri": "LR, geriatric flags", "gbm_geri": "GBM, geriatric flags"}
rows = p4[p4["variant"].isin(names)].copy()
extra = p3[p3["model"].isin(["lr_interact", "gbm_big"])].rename(
    columns={"model": "variant", "auc80_gain_vs_lr": "auc80_change", "auc80_gain_lo": "auc80_change_lo",
             "auc80_gain_hi": "auc80_change_hi"})
names.update({"lr_interact": "LR + age x feature interactions*", "gbm_big": "Larger GBM*"})
rows = pd.concat([rows, extra[["variant", "auc80_change", "auc80_change_lo", "auc80_change_hi"]]])
rows["label"] = rows["variant"].map(names)
rows = rows.iloc[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
yy = np.arange(len(rows))
ax.errorbar(rows["auc80_change"], yy, xerr=[rows["auc80_change"] - rows["auc80_change_lo"],
                                              rows["auc80_change_hi"] - rows["auc80_change"]],
            fmt="o", color="#6250d6", ms=5, elinewidth=1.4, capsize=0)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, rows["label"], fontsize=8)
ax.set_xlabel("Change in AUC for 80+ vs baseline (paired 95% CI)")
ax.set_title("No intervention improves discrimination for 80+", loc="left", fontsize=10, color=INK)
ax.grid(axis="y", visible=False)
fig.text(0.01, 0.01, "* compared with the logistic-regression baseline", fontsize=7, color=INK2)
fig.tight_layout()
fig.savefig(FIGURES / "fig3_interventions.png", dpi=200)
print("saved figures/fig1-3")
