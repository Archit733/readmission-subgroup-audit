# Project explainer: start to end

## Goal

Show that a model's single headline number can hide a group it fails, and build the tooling to detect that,
quantify it honestly, and keep watching for it after deployment.

Three principles run through every component:

1. An aggregate metric can hide subgroup failure.
2. Every subgroup claim carries uncertainty. Non-overlapping 95% intervals means a real gap. A small group is
   "inconclusive", never "no gap".
3. Candidate causes are tested directly, never through a proxy.

The model is deliberately ordinary. The audit around it is the product.

## How the project evolved

1. **Course version (Lab 6).** Ten numbered scripts on the diabetes readmission data: logistic baseline, LightGBM,
   subgroup audit. It found an age gap.
2. **A claim was retracted.** A first explanation (older patients have more diagnoses; r = -0.80 over 9 age-band
   points) fell apart when tested directly with finer bins and then with the Charlson comorbidity index.
3. **Leakage found and fixed.** 101,766 visits come from 71,518 patients. Splitting by patient lowered AUC from
   0.688 to 0.676. The age gap survived.
4. **Industrial version (this repo).** The same ideas rebuilt as a package with config, tests, API, dashboard,
   drift monitoring and a model card.

## Data

UCI "Diabetes 130-US Hospitals", 1999-2008. 101,766 visits, 71,518 patients, 130 hospitals. Label: readmitted within
30 days (about 11%). Test set: 20,354 visits, held out by patient. The dataset is a testbed; the method does not
depend on diabetes.

## Components

| Component | File | What it does |
|---|---|---|
| Patient-level split | `data.py` | Keeps every visit of a patient on one side of the split. |
| Preprocessing | `preprocessing.py` | Imputes and encodes inside one sklearn Pipeline; unseen categories map to -1. |
| Models | `modeling.py` | Logistic baseline (AUC 0.640) and gradient-boosted trees (AUC 0.676). |
| Subgroup audit | `audit.py` | AUC and 95% bootstrap interval (1,000 resamples) per age band, gender and race; groups under 100 patients skipped. |
| Calibration | `audit.py` | ECE about 0.005: probabilities are trustworthy overall. |
| Fairlearn cross-check | `fairness.py` | Independent recomputation of every subgroup AUC; the build fails on disagreement. |
| Operating point | `fairness.py` | Flag the top 20% by risk; recall per group with bootstrap CI. |
| Mechanism tests | `mechanism.py` | Diagnosis count and Charlson index (diabetes excluded) tested directly against AUC. |
| Scoring API | `serving.py`, `api.py` | FastAPI: probability, high-risk flag, per-age-band reliability note; drift-check endpoint; no patient data logged. |
| Dashboard | `app/dashboard.py` | Six-tab Streamlit explorer. |
| Monitoring | `monitoring.py`, `simulation.py` | PSI input drift plus per-subgroup performance alerts on a simulated stream. |
| Tests and docs | `tests/`, `MODEL_CARD.md` | 92 tests; model card generated from the results. |

## Definitions (plain text)

* AUC: probability that the model ranks a random readmitted patient above a random non-readmitted one. 0.5 is a coin
  flip; 1.0 is perfect.
* ECE = sum over bins of (bin size / total) * |actual rate - predicted rate|
* Recall for a group = readmitted patients flagged / all readmitted patients in that group.
* PSI = sum over bins of (actual share - expected share) * ln(actual share / expected share). Below 0.10 stable,
  0.10 to 0.25 moderate, above 0.25 major.

## Results

| Finding | Result |
|---|---|
| Aggregate AUC | 0.676 [0.665, 0.687] (baseline 0.640) |
| Age 20-30 | 0.776 [0.688, 0.848] |
| Age 90-100 | 0.550 [0.466, 0.623] (no overlap: a real gap) |
| Gender | Female 0.680 vs Male 0.672, no meaningful gap |
| Race | Caucasian 0.673 vs African American 0.668, no gap; Hispanic, Asian, Other (n = 114 to 392) inconclusive |
| Recall when flagging top 20% | 64% at 20-30; 23% [13%, 33%] at 90-100; random flagging 20% |
| Mechanism | Diagnosis count and Charlson index show no clean gradient. Cause unexplained. |
| Monitoring simulation | Age PSI crosses 0.10 at batch 6 and 0.25 at batch 8; AUC alerts only at batch 9 |

## What the project does not show

* Why the age gap exists.
* That the finding holds on newer or external data (this data ends in 2008).
* Anything about real-world drift: the monitoring is a stress test on synthetic drift, and it does not demonstrate
  "aggregate quiet while a subgroup alerts".
* Stability across splits: one seeded split.
* Dockerfile and CI were written but not executed in the authoring environment.
