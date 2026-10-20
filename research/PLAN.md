# Analysis plan (pre-specified)

Written and committed before any result from this repository was computed.
Any deviation is logged in the "Deviations" section at the bottom, with a date and reason.

## Question
On the UCI Diabetes 130-US Hospitals data, a standard readmission model's discrimination
(AUC for 30-day readmission) appeared to fall with age. Does that gap survive a
patient-level analysis, and if so, is it explained by base rates, label definition,
case mix, discharge destination, model capacity, or sample size, or does it reflect
information missing from the features (irreducible noise given these features)?

## Cohort (primary)
- Exclude encounters with discharge_disposition_id in {11, 19, 20, 21} (expired) and {13, 14} (hospice).
- Keep the first encounter per patient_nbr (ordered by encounter_id).
- Label: readmitted == "<30" -> 1, else 0.
- Age bands (fixed): <40, 40-59, 60-79, 80+.

## Primary estimand
delta = AUC(40-59) - AUC(80+), from out-of-fold predictions of 5-fold stratified
(label x band) grouped (patient) cross-validation, seed 42.
95% CI by paired patient-level (cluster) bootstrap, 2000 replicates.
Secondary: max - min across the four bands; AUC by the original 10-year buckets.

## Models
- LR: logistic regression, one-hot categoricals, standardised numerics, C = 1.
- GBM: LightGBM, 300 trees, learning rate 0.05, 31 leaves (the capstone's settings).

## Features (identical in all phases unless the phase changes them on purpose)
Numerics: time_in_hospital, num_lab_procedures, num_procedures, num_medications,
number_outpatient, number_emergency, number_inpatient, number_diagnoses, age (bucket midpoint).
Categoricals: race, gender, admission_type_id, discharge_disposition_id, admission_source_id,
medical_specialty (top 15, else "Other"), diag_1/2/3 grouped into the 9 ICD-9 groups of
Strack et al. (2014), max_glu_serum, A1Cresult, the 21 diabetes medication columns with
variation, change, diabetesMed. Dropped: weight, payer_code (mostly missing), IDs.

## Phase 1: leakage
Conditions: (A) all encounters, random row-level 80/20 split (the capstone's design);
(B) all encounters, patient-grouped 5-fold CV; (C) B with death/hospice exclusions;
(D) primary cohort. Report per-band AUC, delta and CI for each.
**Gate:** if the CI for delta includes 0 in D for both models, the headline becomes
"the gap was largely an artefact"; phases 2-4 still run but are framed as robustness checks.

## Phase 2: cheap explanations (primary cohort, LR and GBM)
1. Base rates: n, positives, prevalence per band.
2. Label definition: (a) <30 vs NO, dropping >30; (b) any readmission vs NO. Retrain, report delta.
3. Case mix: propensity model for P(80+ | features other than age), cross-fitted;
   reweight 40-59 by w = p/(1-p), weights truncated at the 99th percentile; recompute AUC(40-59).
   Gap "explained by case mix" if the reweighted delta's CI includes 0 and its point estimate
   is under half the unweighted delta.
4. Discharge destination: delta within home (1), home with health service (6), skilled nursing facility (3).
5. Calibration per band: calibration intercept and slope (logistic recalibration), ECE (10 bins).
   Spread of predicted risk per band (SD of OOF logit).

## Phase 3: decomposition (Chen, Johansson & Sontag, 2018)
1. Variance: learning curves. Training fractions {0.1, 0.2, 0.3, 0.5, 0.75, 1.0} of the training
   folds, 5 repeats, per-band test AUC, for LR and GBM. Plus a sample-size-matched test:
   band-specific models trained on equal n for 40-59 and 80+.
   "Variance-dominated" if the 80+ curve gains more than 0.01 AUC between fractions 0.5 and 1.0
   while 40-59 gains less.
2. Bias (model class): LR, LR + age-band x feature interactions, GBM, larger GBM (1000 trees,
   63 leaves, lr 0.03). "Model class not the bottleneck" if delta stays within each other's CIs.
3. Noise: k-NN (k in {5, 10, 25}) label separability within each band at matched n, on
   standardised features; report within-band k-NN AUC.
4. Per-band permutation importance (GBM) to see which signals carry discrimination where.

## Phase 4: interventions (primary cohort)
1. Band-specific models (train and test within band).
2. Reweighting: 80+ examples upweighted x2 and x5 in training.
3. Richer features: 3-digit ICD-9 codes for diag_1..3 (one-hot, min frequency 50),
   full medical_specialty, plus GBM.
Report AUC(80+) and delta for each, paired CIs against the baseline model.

## Deviations
All logged on 2026-10-17. None changes a pre-specified estimand, threshold or decision rule;
each addition is labelled "exploratory" wherever it is reported.

1. Phase 1: the 10-year-bucket table (secondary) uses a 500-replicate row bootstrap instead of 2000.
2. Phase 2.3: added the reverse reweighting (80+ reweighted to the 40-59 case mix). Exploratory.
   The pre-specified test (40-59 reweighted to the 80+ mix) is reported as the primary case-mix result.
3. Phase 3.1: 3 subsampling repeats per training fraction instead of 5 (compute: two CPU cores).
   The LightGBM power-law fits hit their parameter bounds and are not interpreted; the
   pre-specified gain criterion (fraction 0.5 -> 1.0) is used instead.
4. Phase 3.3: k in {10, 25, 50, 100} instead of {5, 10, 25}; with a 7-11% outcome rate, k = 5
   gives almost no positives per neighbourhood. Run at two matched sizes (12,871 and 4,500).
5. Phase 3.4: added single-feature within-band AUCs and a check of the diagnosis-count ceiling. Exploratory.
6. Phase 4.4: added geriatric-syndrome flags derived from the three recorded diagnosis codes. Exploratory.
