# Age gap in readmission-model discrimination: a patient-level audit

**Question.** On the UCI Diabetes 130-US Hospitals data, does a 30-day readmission model discriminate worse for the oldest patients? If it does, why, and can the gap be closed?

**Answer, in one paragraph.** Yes. In the standard patient-level cohort (first encounter per patient, deaths and hospice excluded; 69,990 patients), AUC is 0.647 for ages 40-59 and 0.603 for 80+, a gap of 0.044 (paired 95% CI 0.021 to 0.066). The usual encounter-level analysis roughly doubles the gap, mainly because of repeat admitters, not leakage. The gap is not explained by outcome rarity, label definition, sample size or model capacity. It sits in the part of feature space occupied by elderly patients, and band-specific models, reweighting, finer ICD codes and geriatric flags do not close it. Calibration-in-the-large is fine for every age, but the calibration slope for 80+ is 0.72.

The full write-up is in `paper/draft.md`. The analysis plan was committed before any result (`PLAN.md`, first commit), and deviations are logged at its end.

## Run

```bash
python3 -m pip install -r requirements.txt
# place diabetic_data.csv (UCI dataset 296) in data/
bash run_all.sh        # about 25 minutes on 2 CPU cores
```

| Script | Phase | Main output |
|---|---|---|
| `01_leakage.py` | 1. Patient-level cohort and splits; gap under four designs | `results/p1_leakage.csv` |
| `02_cheap_explanations.py` | 2. Base rates, label definitions, case mix, discharge destination, calibration | `results/p2_*.csv` |
| `03_decomposition.py` | 3. Learning curves, matched-n models, model class, k-NN noise, importance | `results/p3_*.csv` |
| `04_interventions.py` | 4. Band-specific models, reweighting, richer features, geriatric flags | `results/p4_*.csv` |
| `05_figures.py` | Figures 1-3 | `figures/` |

All shared choices (cohort, bands, features, models, folds, bootstrap) live in `src/common.py`.

## Relation to the capstone

This repository extends [Python-Capstone-Project](https://github.com/Archit733/Python-Capstone-Project). The capstone used an encounter-level single split and reported AUC 0.81 for ages 20-29 against 0.58 for ages 90+. Under the patient-level design here, those figures are 0.70 and 0.54. The capstone's retracted "diagnosis complexity" explanation is replaced by the decomposition above.
