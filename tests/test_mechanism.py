"""Tests for readmit_audit.mechanism -- Charlson mapping and diagnosis binning."""
import numpy as np
import pandas as pd
import pytest
from readmit_audit.mechanism import (
    charlson_category, charlson_score_for_row, compute_charlson_scores,
    bin_diagnosis_count, bin_charlson_score,
)


class TestCharlsonCategory:
    def test_known_mi_code(self):
        assert charlson_category("410")[0] == ("MI", 1)

    def test_known_malignancy_code(self):
        assert charlson_category("150")[0] == ("Malignancy", 2)  # 150 = esophageal cancer, in 140-172.99

    def test_known_metastatic_code_higher_weight(self):
        assert charlson_category("197")[0] == ("MetastaticTumor", 6)

    def test_diabetes_code_returns_empty_by_design(self):
        """Diabetes is deliberately excluded -- this is the project's key
        methodological decision and must not silently regress."""
        assert charlson_category("250.83") == []
        assert charlson_category("250") == []

    def test_v_code_returns_empty(self):
        assert charlson_category("V27") == []

    def test_e_code_returns_empty(self):
        assert charlson_category("E819") == []

    def test_missing_value_returns_empty(self):
        assert charlson_category(np.nan) == []
        assert charlson_category("?") == []

    def test_unmapped_code_returns_empty(self):
        assert charlson_category("999") == []  # not in any Charlson category

    def test_decimal_code_in_range(self):
        # 428.0 - 428.99 is CHF
        assert charlson_category("428.0")[0] == ("CHF", 1)


class TestCharlsonScoreForRow:
    def test_dedupes_same_category_across_fields(self):
        """Same category appearing in diag_1 and diag_2 should count ONCE, not twice."""
        score = charlson_score_for_row("410", "412", None)  # both map to MI, weight 1
        assert score == 1  # not 2

    def test_sums_distinct_categories(self):
        score = charlson_score_for_row("410", "150", None)  # MI (1) + Malignancy (2)
        assert score == 3

    def test_all_missing_gives_zero(self):
        assert charlson_score_for_row(np.nan, np.nan, np.nan) == 0

    def test_all_diabetes_gives_zero(self):
        """A patient whose only recorded diagnoses are diabetes variants should
        score 0 -- confirms the exclusion works end-to-end, not just per-code."""
        assert charlson_score_for_row("250", "250.01", "250.83") == 0


class TestComputeCharlsonScores:
    def test_matches_row_level_function(self):
        df = pd.DataFrame({
            "diag_1": ["410", "250", np.nan],
            "diag_2": [np.nan, "150", "999"],
            "diag_3": [np.nan, np.nan, np.nan],
        })
        scores = compute_charlson_scores(df)
        assert scores.tolist() == [1, 2, 0]


class TestBinning:
    def test_bin_diagnosis_count_individual(self):
        assert bin_diagnosis_count(5) == "5"

    def test_bin_diagnosis_count_tail(self):
        assert bin_diagnosis_count(12, max_individual=9) == "10+"

    def test_bin_diagnosis_count_missing(self):
        assert bin_diagnosis_count(np.nan) is None

    def test_bin_charlson_zero(self):
        assert bin_charlson_score(0) == "0"

    def test_bin_charlson_boundaries(self):
        assert bin_charlson_score(2) == "1-2"
        assert bin_charlson_score(3) == "3-4"
        assert bin_charlson_score(5) == "5+"
