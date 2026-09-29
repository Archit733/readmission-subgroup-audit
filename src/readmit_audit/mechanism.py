"""Candidate-mechanism testing tools: Charlson Comorbidity Index (diabetes
component deliberately excluded -- see charlson_category docstring) and
direct diagnosis-count binning.

Design rule this module enforces: a candidate mechanism must be tested
directly, at full resolution, against the outcome -- never accepted on the
strength of a correlation computed through a coarser proxy variable (e.g.
age bands). This rule exists because of a real mistake caught during this
project's development: an age-band-routed correlation (r=-0.80, n=9 points)
looked like strong evidence for a diagnosis-complexity mechanism, but did not
replicate when tested directly on ~20,000 patients at full resolution. See
reports/results/diagnosis_binning_results.csv and charlson_results.csv.
"""
import numpy as np
import pandas as pd

# Charlson Comorbidity Index categories (Deyo 1992 ICD-9-CM adaptation).
# Diabetes categories are intentionally omitted: this dataset's inclusion
# criterion is that every encounter has a diabetes diagnosis somewhere in the
# chart, so diabetes is not an independent, between-patient-varying
# comorbidity here -- it's the population-defining condition. Including it
# would encode diagnosis-ORDERING artifacts (which of the top-3 recorded
# diagnoses happened to capture it) rather than real signal.
CHARLSON_RULES = [
    ("MI", 1, [(410, 410.99), (412, 412.99)]),
    ("CHF", 1, [(428, 428.99)]),
    ("PVD", 1, [(441, 441.99), (443.9, 443.9), (785.4, 785.4)]),
    ("CVD", 1, [(430, 438.99)]),
    ("Dementia", 1, [(290, 290.99)]),
    ("COPD", 1, [(490, 496.99), (500, 505.99), (506.4, 506.4)]),
    ("Rheumatic", 1, [(710.0, 710.1), (710.4, 710.4), (714.0, 714.2), (714.81, 714.81), (725, 725.99)]),
    ("PepticUlcer", 1, [(531, 534.99)]),
    ("MildLiver", 1, [(571.2, 571.2), (571.5, 571.6), (571.4, 571.49)]),
    ("Hemiplegia", 2, [(342, 342.99), (344.1, 344.1)]),
    ("Renal", 2, [(582, 582.99), (583, 583.99), (585, 585.99), (586, 586.99), (588, 588.99)]),
    ("Malignancy", 2, [(140, 172.99), (174, 195.99), (200, 208.99)]),
    ("ModSevereLiver", 3, [(572.2, 572.8), (456.0, 456.2)]),
    ("MetastaticTumor", 6, [(196, 199.99)]),
    ("AIDS", 6, [(42, 44.99)]),
]


def charlson_category(code) -> list:
    """Map one ICD-9 code (string) to a list of (category_name, weight) matches.
    Returns [] for missing values, V-codes, E-codes, or codes in no category.
    """
    if pd.isna(code):
        return []
    code = str(code).strip()
    if code in ("", "?") or code.startswith("V") or code.startswith("E"):
        return []
    try:
        num = float(code)
    except ValueError:
        return []

    matched = []
    for name, weight, ranges in CHARLSON_RULES:
        for lo, hi in ranges:
            if lo <= num <= hi:
                matched.append((name, weight))
                break
    return matched


def charlson_score_for_row(diag1, diag2, diag3) -> int:
    """Sum Charlson weights across up to 3 diagnosis fields, de-duplicated by
    category (a condition appearing in both diag_1 and diag_2 counts once)."""
    seen = {}
    for code in (diag1, diag2, diag3):
        for name, weight in charlson_category(code):
            seen[name] = weight
    return sum(seen.values())


def compute_charlson_scores(df: pd.DataFrame) -> pd.Series:
    """Vectorized-ish wrapper: compute Charlson score per row from diag_1/2/3."""
    return pd.Series(
        [charlson_score_for_row(d1, d2, d3) for d1, d2, d3 in zip(df["diag_1"], df["diag_2"], df["diag_3"])],
        index=df.index, name="charlson",
    )


def bin_diagnosis_count(n, max_individual: int = 9) -> str:
    """Bin a raw diagnosis count: 1..max_individual get their own bin, above
    that is grouped (sparse tail, individually unreliable)."""
    if pd.isna(n):
        return None
    n = int(n)
    if n <= max_individual:
        return str(n)
    return f"{max_individual + 1}+"


def bin_charlson_score(score) -> str:
    """Bin a Charlson score into 4 clinically-spaced buckets."""
    if pd.isna(score):
        return None
    score = int(score)
    if score == 0:
        return "0"
    elif score <= 2:
        return "1-2"
    elif score <= 4:
        return "3-4"
    return "5+"
