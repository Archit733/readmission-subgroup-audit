# Readmission Subgroup Audit: from capstone to deployable pipeline

**Question:** a readmission-risk model reports one AUC. Does that number hold for every patient group, and
what happens operationally when it doesn't? The audit is the product; the model is deliberately ordinary.

**Data:** UCI "Diabetes 130-US Hospitals" (101,766 encounters, 71,518 patients, 1999-2008).

## Headline results (all read from pipeline outputs; see `MODEL_CARD.md`)

| | |
|---|---|
| Aggregate AUC | 0.676 [0.665, 0.687] (logistic baseline 0.640) |
| Best vs worst age band | [20-30) 0.776 vs [90-100) 0.550, intervals do not overlap |
| Gender | no meaningful gap (0.680 vs 0.672) |
| Race | Caucasian vs African American no gap; Hispanic/Asian/Other (n = 114-392) inconclusive |
| If the top 20% by risk are flagged | catches 64% of readmissions at 20-30 but only 23% [13%, 33%] at 90-100 (random = 20%) |
| Mechanism | diagnosis count and Charlson index (diabetes excluded) tested directly: **no clean gradient**. Cause unexplained. |

## What "industrial" means here

| Concern | Implementation |
|---|---|
| Reproducibility | `config.yaml` drives everything; seeded; one command per stage; CI runs the whole chain |
| Correctness | 89 pytest tests; two bugs found and fixed by them (positional best/worst comparison; NaN breaking JSON) |
| Data leakage | patient-level split. 30% of encounters are repeat patients; a naive row split inflated AUC (0.688 to 0.676 once fixed) |
| Deployability | one sklearn Pipeline (impute, encode, model) so raw records can be scored; unseen categories map to -1, never crash |
| Independent verification | every subgroup AUC re-computed with Fairlearn `MetricFrame`; build fails if they disagree |
| Serving | FastAPI `/predict` returns probability, high-risk flag, and a per-age-band **reliability note** |
| Monitoring | PSI input drift + per-subgroup performance alerts on a simulated drifting stream |
| Documentation | auto-generated `MODEL_CARD.md` (numbers cannot drift from results) |

## Run it

```bash
pip install -r requirements.txt
make all              # pipeline -> reports/model card -> monitoring simulation -> tests
make dashboard        # streamlit run app/dashboard.py
make api              # uvicorn on :8000  (docs at /docs)
```
Example: `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"records":[{"age":"[90-100)","gender":"Female","number_inpatient":3}]}'`

```
config.yaml   Makefile   Dockerfile   MODEL_CARD.md
src/readmit_audit/  data, preprocessing, modeling, audit (bootstrap CI, calibration), fairness (Fairlearn + operating point),
                    mechanism (Charlson), monitoring (PSI, alerts), simulation, serving, api
scripts/            run_pipeline, build_reports, simulate_monitoring
app/dashboard.py    six-tab Streamlit explorer
tests/              89 tests (hermetic API tests use a tiny fitted model, not the full pipeline)
```

## Monitoring simulation: what it does and does not show
Batches 0-3 are clean; later batches get synthetic drift toward older patients plus growing label noise for age 70+.
* **Input drift is caught first and cleanly.** Age-mix PSI stays under 0.01 while clean, crosses "moderate" (0.10) at
  batch 6 and "major" (0.25) at batch 8. Clean-batch PSI on every feature stays under 0.04.
* **Outcome-based alerts lag.** Only the final batch triggers (trailing 3-batch window), for both the 70+ group and the
  aggregate. In reality readmission labels arrive 30 days late, so PSI is the early-warning signal and AUC alerts confirm it.
* **It does not demonstrate "aggregate stays quiet while a subgroup alerts".** The drifted group is ~60-75% of the
  stream, so the aggregate moves with it. Small subgroups lack the per-batch sample size to alert at all. Pooling
  batches (`window`) helps but delays detection.
* This is a stress test of the alerting logic on synthetic drift, not evidence about real-world drift.

## Known limitations
* Not validated on any external or recent cohort (data ends 2008); revalidate locally before any use.
* Single seeded split; no repeated-split variance estimate yet.
* `Dockerfile` and `.github/workflows/ci.yml` were written but **not executed** here (no Docker daemon, no GitHub runner).
  Everything else was run: pipeline, tests, API against a live uvicorn server, dashboard via Streamlit AppTest and a browser screenshot.
* PSI thresholds (0.10 / 0.25) are industry convention, not tuned to this data.
* Age is used both as a model feature and as the audited attribute; whether a model should use it is a policy question this project does not settle.

## Docs
* `docs/PROJECT_EXPLAINER.md`: the project start to end, component by component
* `docs/PRESENTATION_SCRIPT.md`: 5-minute presenter script with likely questions
* `docs/PUBLICATION_PLAN.md`: honest assessment of publishability, venues, and extension steps
* `MODEL_CARD.md`: auto-generated from the results

## Lineage
Course version (Lab 6): 10 numbered scripts, single row-level split, LightGBM. Kept the ideology, replaced the plumbing.
An earlier mechanism claim (diagnosis complexity, r = -0.80 over 9 age-band points) was retracted after direct testing
at full resolution; that rule ("test candidate mechanisms directly, never through a proxy") is now encoded in `mechanism.py`.
