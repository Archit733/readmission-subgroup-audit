# Readmission models discriminate worse for the oldest patients, and neither more data nor a bigger model fixes it: a patient-level audit of the Diabetes 130-US Hospitals benchmark

Archit Bhatia, Plaksha University
Draft v0.1, 17 October 2026. Target: workshop or findings track (e.g. ML4H Findings, 4 pages) and an arXiv preprint.

## Abstract

Aggregate discrimination can hide subgroup failures. We ask whether a standard 30-day readmission model discriminates equally well across age, and if it does not, why. On the UCI Diabetes 130-US Hospitals data (one encounter per patient, deaths and hospice discharges excluded; 69,990 patients), logistic regression reaches an overall AUC of 0.648. The AUC is 0.647 for patients aged 40-59 and 0.603 for patients aged 80 and over, a gap of 0.044 (paired patient-level bootstrap 95% CI 0.021 to 0.066). LightGBM gives 0.036 (0.013 to 0.059). The common encounter-level analysis roughly doubles the gap. Most of that inflation comes from repeat admitters, who are easy to predict and concentrated among younger patients, not from train-test leakage.

The gap is not explained by outcome rarity, since readmission is more common in the oldest band, or by the outcome definition. Learning curves for the two bands run parallel, and the gap persists when band-specific models are trained on equal sample sizes, which argues against variance. Logistic regression with explicit age-by-feature interactions and a larger gradient-boosted model leave the gap unchanged, which argues against model capacity. The deficit is localised: once patients aged 80+ are reweighted to resemble the 40-59 case mix, their AUC matches the younger group (exploratory). Band-specific models, upweighting, finer ICD-9 codes and geriatric-syndrome flags do not raise AUC for the 80+ group. Calibration-in-the-large is preserved for all ages, but the calibration slope for 80+ is 0.72 (0.62 to 0.82). The model's risk differences between elderly patients are overstated. We read the gap as a property of the information these features contain about the oldest patients, not of the learner.

## 1. Question

Hospital readmission models are commonly reported with a single AUC. If that AUC is materially lower for an identifiable group, the model's ranking of patients is less trustworthy for exactly that group, and the aggregate does not show it. Older adults are the group with the highest readmission rate and the most at stake. We ask three questions:

1. Is discrimination lower for the oldest patients once the analysis is done at the patient level?
2. If so, is the gap explained by cheap factors (base rate, label definition, case mix, discharge destination), by variance (too little data), or by model capacity (bias)? Or does it persist after all of these, which would indicate that the features carry less information about outcomes in the oldest patients?
3. Does any straightforward intervention close the gap?

An earlier version of this analysis (encounter-level, single split) proposed that diagnosis complexity drives the gap. Direct tests did not support that claim, and it was retracted. This paper replaces that claim with a pre-specified decomposition.

## 2. Data and cohort

Data come from the UCI Diabetes 130-US Hospitals dataset (1999-2008; 101,766 encounters, 71,518 patients) [1]. We follow the cohort definition of the dataset's authors [1]. We exclude encounters ending in death (discharge codes 11, 19, 20, 21) or hospice discharge (13, 14), and keep the first encounter per patient. This yields 69,990 patients. Strack et al. report 69,984; we have not traced the six-patient difference. The outcome is readmission within 30 days (8.98%). Age is recorded in 10-year buckets, which we group into four bands fixed before analysis: <40, 40-59, 60-79 and 80+.

| Band | n | Readmitted <30 d | Rate |
|---|---|---|---|
| <40 | 4,500 | 300 | 6.7% |
| 40-59 | 19,179 | 1,386 | 7.2% |
| 60-79 | 33,440 | 3,232 | 9.7% |
| 80+ | 12,871 | 1,367 | 10.6% |

## 3. Methods

The analysis plan, including every threshold and decision rule below, was committed to the repository before any result was computed. Deviations are logged with reasons.

**Models.** We use two models. The first is logistic regression (LR) with one-hot categoricals and standardised numerics. The second is LightGBM (GBM) with 300 trees, learning rate 0.05 and 31 leaves. Features are the 9 numeric fields (including age as a bucket midpoint) and 34 categorical fields: demographics, admission and discharge codes, top-15 medical specialty, the three diagnosis codes grouped as in [1], laboratory flags and the diabetes medication fields.

**Evaluation.** We use out-of-fold predictions from 5-fold cross-validation, stratified by outcome and age band and grouped by patient. The primary estimand is

    delta = AUC(40-59) - AUC(80+)

Uncertainty comes from a paired patient-level bootstrap with 2,000 replicates. Each replicate resamples patients once and evaluates every band and every model on that same resample, so CIs for differences are valid even where the per-band intervals overlap.

**Decomposition.** Following Chen, Johansson and Sontag [2], we separate three sources of a subgroup gap:
- *Variance*: learning curves over training fractions 0.1 to 1.0, and band-specific models trained on equal n.
- *Bias*: four model classes, including LR with age-band-by-feature interactions.
- *Noise*: within-band k-NN separability at matched n, as a model-light proxy for how well the features separate outcomes.

## 4. Results

### 4.1 Patient-level analysis halves the gap but does not remove it

| Design | LR delta | GBM delta |
|---|---|---|
| A. All encounters, row-level 80/20 split | 0.076 (0.040, 0.109) | 0.068 (0.032, 0.100) |
| B. All encounters, patient-grouped CV | 0.069 (0.052, 0.087) | 0.063 (0.046, 0.082) |
| C. B, deaths and hospice excluded | 0.080 (0.061, 0.097) | 0.079 (0.061, 0.097) |
| D. First encounter per patient (primary) | **0.044 (0.021, 0.066)** | **0.036 (0.013, 0.059)** |

In design A, 42% of test encounters belong to a patient who also appears in training. Grouping by patient, however, changes the gap little (A to B). The reduction comes from keeping one encounter per patient (C to D). Repeat admitters have many prior inpatient visits, which makes them easy to predict, and they inflate AUC most in the young bands. In the earlier encounter-level analysis, AUC was 0.81 for ages 20-29. In the primary cohort it is 0.70 (0.63, 0.76), while ages 90-99 reach only 0.54 (0.50, 0.59). In the primary cohort the decline is concentrated after age 80: AUC is about 0.64-0.66 from 40 to 79 and then falls (Figure 1). Across five re-randomised CV splits, delta varied with standard deviation 0.002 (LR) and 0.004 (GBM).

### 4.2 Cheap explanations

- **Base rate.** Readmission is *more* common in 80+ (10.6% vs 7.2%), and the two bands have nearly equal positives (1,367 vs 1,386). Rarity does not explain lower AUC.
- **Label definition.** The gap persists for "<30 days vs never readmitted" (LR 0.063, CI 0.040 to 0.085) and for "any readmission vs never" (0.044, CI 0.031 to 0.055).
- **Case mix.** Pre-specified test: we reweight 40-59 patients by the cross-fitted odds of belonging to 80+, given every feature except age. This does not shrink the gap (LR 0.053, CI 0.018 to 0.087), so middle-aged patients with elderly-like profiles are still predicted well.
  Exploratory reverse test: reweighting 80+ patients to the 40-59 case mix removes the gap (-0.003, CI -0.039 to 0.034).
  The deficit is therefore confined to the part of feature space occupied mainly by the elderly. The two groups are well separated without age (propensity AUC 0.86, effective sample sizes 3,183-3,695), so these estimates concern the region of overlap.
- **Discharge destination.** 31.5% of 80+ patients go to a skilled nursing facility, compared with 3.6% of 40-59 patients. Within each destination the gap shrinks and its CI includes zero for home (0.021, CI -0.015 to 0.057) and skilled nursing (0.022, CI -0.051 to 0.092). These within-stratum tests are underpowered.
- **Calibration.** Calibration-in-the-large is close to zero in every band. The spread of predicted risk is not narrower for 80+ (SD of logit 0.52 vs 0.48). But the calibration slope is 0.72 (0.62 to 0.82) for 80+ against 1.00 (0.92 to 1.09) for 40-59. The model separates elderly patients as confidently as younger ones, but its risk differences between elderly patients are overstated.

### 4.3 Decomposition: not variance, not capacity

- **Variance.** The learning curves are parallel (Figure 2). Between half and all of the training data, AUC rises 0.008 for 80+ and 0.006 for 40-59 (LR), or 0.014 and 0.016 (GBM). This does not meet the pre-specified criterion for a variance-dominated gap. The fitted LR asymptotes are 0.611 for 80+ and 0.652 for 40-59. At equal training size (12,871 patients per band), band-specific LR reaches 0.589 for 80+ and 0.632 for 40-59.
- **Bias.** The four model classes give delta between 0.036 and 0.048. Every paired difference from LR has a CI that includes zero.

| Model | AUC 80+ | delta (95% CI) |
|---|---|---|
| LR | 0.603 | 0.044 (0.021, 0.066) |
| LR + age-band x feature interactions | 0.597 | 0.048 (0.024, 0.070) |
| LightGBM | 0.613 | 0.036 (0.013, 0.059) |
| LightGBM, 1,000 trees, 63 leaves | 0.596 | 0.047 (0.025, 0.069) |

- **Noise (weak evidence).** At matched n = 12,871, within-band k-NN AUC is lowest for 80+ at every k (k = 100: 0.556 vs 0.582 for 40-59). At n = 4,500 the 80+ and 60-79 bands are indistinguishable, so this signal is weak.
- **A recording ceiling, as a candidate mechanism.** The dataset stores at most three diagnosis codes, and the diagnosis count saturates at 9. 54% of 80+ patients are at the ceiling, compared with 37% of 40-59 patients. The diagnosis count carries some signal in 40-59 (single-feature AUC 0.56) and none in 80+ (0.50). Permutation importance shows the same pattern: discharge destination and prior inpatient visits carry discrimination in every band, while diagnosis-derived features lose importance with age.

### 4.4 Interventions: a null result

No intervention raised AUC for 80+ (Figure 3). We tried:
- band-specific models (LR -0.007, GBM -0.036);
- upweighting 80+ by 2 or by 5 (between -0.015 and +0.001);
- full 3-digit ICD-9 codes and full medical specialty (LR -0.002, GBM -0.028);
- eight geriatric-syndrome flags such as dementia, falls and fractures, UTI and aspiration pneumonia (LR +0.001, GBM -0.001).

One variant, LR with 80+ weighted by 5, reduced delta to 0.029. It did so by lowering AUC for 40-59 (0.647 to 0.631) while leaving 80+ unchanged. We report it as levelling down, not as a fix.

## 5. Related work and what is new

Fairness analyses of this dataset focus mostly on race. The Fairlearn version [3] works at the encounter level and bins age as 30 or younger, 30-60 and over 60, which merges the band where we find the deficit with the 60-79 band, where we find none. Recent methods papers on this data target cross-racial fairness, for example Lu and Xu [4]. Readmission models in general discriminate modestly; Kansagara et al. found c-statistics from 0.55 to 0.83, with few models above 0.70. That review notes that functional and social variables, rarely available in such data, improve discrimination [5]. In older inpatients, frailty and social support add discrimination over clinical variables [6, 7]. That is consistent with our reading that the 80+ deficit reflects missing information. Our design follows the bias, variance and noise decomposition of Chen, Johansson and Sontag [2] and is motivated by subgroup failures documented in deployed or published clinical models [8, 9].

As far as we can tell, our contributions are the following.
1. We give a patient-level, pre-specified estimate of the age gap on this benchmark, with paired intervals.
2. We show that about half of the gap in the usual encounter-level analysis comes from repeat admitters, not from leakage.
3. We decompose the remaining gap. It is not base rate, label definition, sample size or model capacity, and it is confined to the elderly-specific region of feature space.
4. We report a null result: standard interventions do not close it.

*To verify before submission:* a 2026 study on causal reweighting for calibration and subgroup equity on this dataset [10] could not be accessed here. Its treatment of age needs checking.

## 6. Limitations

- **Competing mortality.** In-hospital deaths and hospice discharges are excluded, but deaths after discharge are not recorded. Some "not readmitted" elderly patients may have died within 30 days, which adds label noise exactly where we see the deficit. This is the most important untested explanation.
- **Age resolution.** Age is coarse (10-year buckets) and partly confounded with care setting: the case-mix and destination results show that the deficit sits where age and nursing-home discharge coincide.
- **Interpretation of "noise".** The k-NN estimate depends on the feature representation. A noise interpretation is the residual after ruling out variance and capacity, not a direct measurement.
- **External validity.** The data are from 1999-2008, US only, diabetic encounters only. We have not replicated on a second dataset.
- **Scope of the bootstrap.** The CIs treat the out-of-fold predictions as fixed. Re-split variability is reported separately (SD of delta 0.002 to 0.004).

## 7. Conclusion

A readmission model with an overall AUC of 0.65 ranks patients aged 80 and over noticeably worse than middle-aged patients. Half of the commonly reported gap is an artefact of repeat admissions, but the remainder survives every check in our pre-specified plan. It is not fixed by more data, more capacity, band-specific models or finer diagnosis codes. The features in this dataset carry less information about who among the oldest patients will be readmitted. Fixing that likely requires different data (function, frailty, social support, post-discharge death) rather than a different model. Until then, aggregate AUCs on this benchmark should be reported alongside age-stratified AUCs and calibration slopes.

## Reproducibility

All code, the pre-specified plan, logged deviations and every results table are in the repository. `run_all.sh` regenerates every number and figure in about 25 minutes on two CPU cores.

## References

1. Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International, 2014:781670.
2. Chen I, Johansson FD, Sontag D. Why is my classifier discriminatory? NeurIPS 2018.
3. Fairlearn. Diabetes 130-Hospitals dataset (user guide). fairlearn.org.
4. Lu H, Xu J. Fair and robust early readmission risk prediction from electronic health records via diffusion-based data augmentation and causal-invariant representation learning. Scientific Reports 16:22277, 2026.
5. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission: a systematic review. JAMA 306(15):1688-1698, 2011.
6. Sokoreli I, Cleland JG, Pauws SC, et al. Added value of frailty and social support in predicting risk of 30-day unplanned re-admission or death for patients with heart failure: an analysis from OPERA-HF. International Journal of Cardiology 278:167-172, 2019.
7. Kahlon S, Pederson J, Majumdar SR, et al. Association between frailty and 30-day outcomes after discharge from hospital. CMAJ 187(11):799-804, 2015.
8. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science 366(6464):447-453, 2019.
9. Seyyed-Kalantari L, Zhang H, McDermott MBA, Chen IY, Ghassemi M. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature Medicine 27:2176-2182, 2021.
10. Improving calibration and subgroup equity in diabetes readmission prediction using causal reweighting of LSTM models. 2026 (ScienceDirect, S235291482600050X). Authors and venue to be confirmed.
