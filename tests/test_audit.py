"""Tests for readmit_audit.audit -- the module the whole project's validity rests on."""
import numpy as np
import pandas as pd
import pytest
from readmit_audit.audit import (
    bootstrap_auc_ci, subgroup_audit, groups_differ,
    calibration_table, expected_calibration_error,
)


class TestBootstrapAucCi:
    def test_perfect_separation_gives_auc_one(self):
        """A model that perfectly ranks positives above negatives should get AUC=1.0
        and a tight CI (not necessarily exactly [1,1] due to bootstrap resampling
        of a finite, perfectly-separable sample -- but the point estimate must be 1.0)."""
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 100 + [1] * 100)
        y_prob = np.concatenate([rng.uniform(0, 0.4, 100), rng.uniform(0.6, 1.0, 100)])
        point, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_boot=200, random_state=1)
        assert point == pytest.approx(1.0)
        assert lo > 0.9  # CI should stay high given clean separation

    def test_random_predictions_give_ci_spanning_half(self):
        """Predictions with no real relationship to the label should give an AUC
        near 0.5, with a CI that comfortably contains 0.5."""
        rng = np.random.default_rng(0)
        n = 2000
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)  # unrelated to y_true by construction
        point, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_boot=300, random_state=2)
        assert 0.45 < point < 0.55
        assert lo < 0.5 < hi

    def test_returns_nan_for_single_class(self):
        """AUC is undefined with only one class present -- must fail loudly (NaN),
        never silently return a fabricated number."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        point, lo, hi = bootstrap_auc_ci(y_true, y_prob, min_n=1)
        assert np.isnan(point)

    def test_returns_nan_below_min_n(self):
        """A sample smaller than min_n should be refused, not computed anyway."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        point, lo, hi = bootstrap_auc_ci(y_true, y_prob, min_n=20)
        assert np.isnan(point)

    def test_is_reproducible_given_same_seed(self):
        """Same inputs + same random_state must give bit-identical output --
        required for any result in this project to be independently verifiable."""
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        r1 = bootstrap_auc_ci(y_true, y_prob, n_boot=100, random_state=42)
        r2 = bootstrap_auc_ci(y_true, y_prob, n_boot=100, random_state=42)
        assert r1 == r2


class TestSubgroupAudit:
    def _cfg(self, min_n=20):
        return {"audit": {"n_bootstrap": 200, "confidence_level": 0.95, "min_subgroup_n": min_n},
                "split": {"random_state": 42}}

    def test_skips_groups_below_min_n(self):
        y_true = pd.Series([0, 1] * 15)
        y_prob = pd.Series(np.linspace(0, 1, 30))
        groups = pd.Series(["A"] * 25 + ["B"] * 5)  # B has only 5, below min_n=20
        result = subgroup_audit(y_true, y_prob, groups, self._cfg(min_n=20))
        assert set(result["group"]) == {"A"}

    def test_includes_groups_at_or_above_min_n(self):
        rng = np.random.default_rng(3)
        y_true = pd.Series(rng.integers(0, 2, 60))
        y_prob = pd.Series(rng.uniform(0, 1, 60))
        groups = pd.Series(["A"] * 30 + ["B"] * 30)
        result = subgroup_audit(y_true, y_prob, groups, self._cfg(min_n=20))
        assert set(result["group"]) == {"A", "B"}
        assert (result["n"] == 30).all()

    def test_respects_explicit_group_order(self):
        """A caller-supplied group_order must be preserved in the output row
        order -- this matters because downstream code (e.g. 'compare first vs
        last row for youngest-vs-oldest') silently breaks if order isn't
        guaranteed. Caught as a real bug during pipeline integration."""
        rng = np.random.default_rng(4)
        n_per_group = 40
        y_true = pd.Series(rng.integers(0, 2, n_per_group * 3))
        y_prob = pd.Series(rng.uniform(0, 1, n_per_group * 3))
        # Data appears in reverse-alphabetical order, but we request A, B, C order
        groups = pd.Series(["C"] * n_per_group + ["B"] * n_per_group + ["A"] * n_per_group)
        result = subgroup_audit(y_true, y_prob, groups, self._cfg(min_n=20),
                                 group_order=["A", "B", "C"])
        assert result["group"].tolist() == ["A", "B", "C"]


class TestGroupsDiffer:
    def test_non_overlapping_cis_are_different(self):
        row_a = pd.Series({"ci_lo": 0.8, "ci_hi": 0.9})
        row_b = pd.Series({"ci_lo": 0.5, "ci_hi": 0.6})
        assert groups_differ(row_a, row_b) is True

    def test_overlapping_cis_are_not_different(self):
        row_a = pd.Series({"ci_lo": 0.6, "ci_hi": 0.8})
        row_b = pd.Series({"ci_lo": 0.7, "ci_hi": 0.9})
        assert groups_differ(row_a, row_b) is False

    def test_nan_ci_is_never_claimed_different(self):
        """An untrustworthy (NaN) CI must never be reported as evidence of a gap."""
        row_a = pd.Series({"ci_lo": np.nan, "ci_hi": np.nan})
        row_b = pd.Series({"ci_lo": 0.5, "ci_hi": 0.6})
        assert groups_differ(row_a, row_b) is False

    def test_best_vs_worst_by_point_estimate_not_position(self):
        """Regression test for a real pipeline bug: comparing the first/last
        rows of a subgroup table (by position) instead of the best/worst by
        AUC point estimate can pick a small, wide-CI extreme group and wrongly
        conclude 'no real gap' even when one clearly exists elsewhere in the
        table. idxmax/idxmin must be used, not iloc[0]/iloc[-1]."""
        table = pd.DataFrame([
            {"group": "tiny_extreme_1", "auc": 0.70, "ci_lo": 0.40, "ci_hi": 0.95},  # small n, wide CI
            {"group": "best", "auc": 0.85, "ci_lo": 0.80, "ci_hi": 0.90},
            {"group": "worst", "auc": 0.60, "ci_lo": 0.55, "ci_hi": 0.65},
            {"group": "tiny_extreme_2", "auc": 0.65, "ci_lo": 0.35, "ci_hi": 0.92},  # small n, wide CI
        ])
        best = table.loc[table["auc"].idxmax()]
        worst = table.loc[table["auc"].idxmin()]
        assert best["group"] == "best"
        assert worst["group"] == "worst"
        assert groups_differ(best, worst) is True
        # the naive positional version would have compared the two wide-CI
        # extremes and found overlapping CIs -- confirm that WOULD be wrong:
        positional_first, positional_last = table.iloc[0], table.iloc[-1]
        assert groups_differ(positional_first, positional_last) is False


class TestCalibration:
    def test_perfectly_calibrated_predictions_give_zero_ece(self):
        """If predicted probability exactly matches observed frequency in every
        bin, ECE must be (approximately) zero."""
        rng = np.random.default_rng(0)
        n = 5000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)  # true rate = predicted rate
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.05  # small tolerance for finite-sample noise

    def test_badly_miscalibrated_predictions_give_high_ece(self):
        """Predictions that are the OPPOSITE of the true rate should score a high ECE."""
        n = 2000
        y_prob = np.full(n, 0.9)
        y_true = np.zeros(n, dtype=int)  # predicted 0.9, true rate 0.0
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.8

    def test_calibration_table_bin_sizes_sum_to_n(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 300)
        y_prob = rng.uniform(0, 1, 300)
        tbl = calibration_table(y_true, y_prob, n_bins=10)
        assert tbl["n"].sum() == 300
