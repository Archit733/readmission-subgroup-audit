import numpy as np
import pandas as pd
import pytest
from readmit_audit.monitoring import (psi, psi_status, feature_drift_report, degradation_alert,
                                      batch_subgroup_monitor, PSI_MAJOR)


def test_psi_identical_distributions_is_near_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0, abs=1e-6)


def test_psi_detects_shift_and_ranks_by_severity():
    rng = np.random.default_rng(1); ref = rng.normal(size=5000)
    small, big = psi(ref, rng.normal(0.2, 1, 5000)), psi(ref, rng.normal(1.5, 1, 5000))
    assert small < big and psi_status(big) == "major"


def test_psi_same_sample_size_noise_stays_stable():
    rng = np.random.default_rng(2)
    assert psi_status(psi(rng.normal(size=5000), rng.normal(size=5000))) == "stable"


def test_psi_categorical_and_unseen_category():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    assert psi(ref, pd.Series(["a"] * 500 + ["b"] * 500)) == pytest.approx(0, abs=1e-6)
    assert psi(ref, pd.Series(["a"] * 300 + ["b"] * 300 + ["NEW"] * 400)) > PSI_MAJOR


def test_psi_empty_returns_nan():
    assert np.isnan(psi([], [1, 2, 3]))


def test_feature_drift_report_sorted_and_labelled():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"x": rng.normal(size=3000), "y": rng.normal(size=3000)})
    cur = pd.DataFrame({"x": rng.normal(size=3000), "y": rng.normal(2, 1, 3000)})
    rep = feature_drift_report(ref, cur)
    assert rep.iloc[0]["feature"] == "y" and rep.iloc[0]["status"] == "major"


def test_degradation_alert_rules():
    assert degradation_alert(0.60, 0.70) is True
    assert degradation_alert(0.72, 0.70) is False      # interval still reaches the reference
    assert degradation_alert(np.nan, 0.70) is False    # untrustworthy CI never alerts


def _scored(n, signal, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, np.clip(signal * y + rng.normal(0.4, 0.25, n), 0, 1)


def test_monitor_alerts_on_degraded_subgroup_only():
    yA, pA = _scored(2000, 0.5, 4); yB, pB = _scored(2000, 0.5, 5)
    yB2, pB2 = _scored(2000, 0.0, 6)          # subgroup B stops carrying signal in batch 1
    y = np.r_[yA, yB, yA, yB2]; p = np.r_[pA, pB, pA, pB2]
    g = np.array(["A"] * 2000 + ["B"] * 2000 + ["A"] * 2000 + ["B"] * 2000)
    b = np.array([0] * 4000 + [1] * 4000)
    from sklearn.metrics import roc_auc_score
    ref = {"A": roc_auc_score(yA, pA), "B": roc_auc_score(yB, pB), "__overall__": 0.7}
    out = batch_subgroup_monitor(y, p, g, b, ref, min_n=100, n_boot=100)
    alerts = out[out["alert"]]
    assert set(alerts["group"]) == {"B"} and set(alerts["batch"]) == {1}


def test_monitor_skips_small_groups():
    y, p = _scored(100, 0.5, 7)
    out = batch_subgroup_monitor(y, p, np.array(["A"] * 100), np.zeros(100), {"A": 0.7, "__overall__": 0.7}, min_n=150)
    assert out.empty


def test_window_one_matches_single_batch_behaviour_and_window_pools_batches():
    y, p = _scored(3000, 0.5, 8)
    g = np.array(["A"] * 3000); b = np.repeat([0, 1, 2], 1000)
    ref = {"A": 0.7, "__overall__": 0.7}
    w1 = batch_subgroup_monitor(y, p, g, b, ref, min_n=100, n_boot=50, window=1)
    w3 = batch_subgroup_monitor(y, p, g, b, ref, min_n=100, n_boot=50, window=3)
    assert w1[w1.group == "A"]["n"].tolist() == [1000, 1000, 1000]
    assert w3[w3.group == "A"]["n"].tolist() == [1000, 2000, 3000]


def test_rare_categories_pooled_so_high_cardinality_noise_is_stable():
    rng = np.random.default_rng(9)
    codes = [f"c{i}" for i in range(500)]
    a = pd.Series(rng.choice(codes, 2000)); b = pd.Series(rng.choice(codes, 2000))
    assert psi_status(psi(a, b)) == "stable"
