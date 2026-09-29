import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from readmit_audit.fairness import (fairlearn_auc_by_group, crosscheck_auc, top_fraction_threshold,
                                    threshold_metrics_by_group, max_gap)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 1200
    g = np.array(["a"] * 600 + ["b"] * 400 + ["c"] * 200)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.25, n), 0, 1)
    return y, p, g


def test_fairlearn_matches_sklearn(data):
    y, p, g = data
    fl = fairlearn_auc_by_group(y, p, g)
    for grp in ["a", "b", "c"]:
        assert fl[grp] == pytest.approx(roc_auc_score(y[g == grp], p[g == grp]))


def test_crosscheck_agrees_with_own_audit(data):
    y, p, g = data
    tbl = pd.DataFrame([{"group": k, "auc": roc_auc_score(y[g == k], p[g == k])} for k in "abc"])
    assert crosscheck_auc(tbl, y, p, g)["agrees"].all()


def test_crosscheck_flags_disagreement(data):
    y, p, g = data
    tbl = pd.DataFrame([{"group": "a", "auc": 0.99}])
    assert not crosscheck_auc(tbl, y, p, g)["agrees"].iloc[0]


def test_top_fraction_threshold_flags_expected_share(data):
    _, p, _ = data
    thr = top_fraction_threshold(p, 0.2)
    assert (p >= thr).mean() == pytest.approx(0.2, abs=0.01)


def test_threshold_metrics_respects_min_n(data):
    y, p, g = data
    t = threshold_metrics_by_group(y, p, g, 0.5, min_n=300)
    assert set(t["group"]) == {"a", "b"}


def test_max_gap():
    assert max_gap(pd.DataFrame({"x": [0.1, 0.4, 0.2]}), "x") == pytest.approx(0.3)


def test_bootstrap_tpr_ci_contains_point_and_is_reproducible():
    from readmit_audit.fairness import bootstrap_tpr_ci
    rng = np.random.default_rng(1)
    y = np.ones(200, dtype=int); pred = (rng.uniform(size=200) < 0.4).astype(int)
    r1 = bootstrap_tpr_ci(y, pred); r2 = bootstrap_tpr_ci(y, pred)
    assert r1 == r2 and r1[1] <= r1[0] <= r1[2] and r1[3] == 200


def test_bootstrap_tpr_ci_refuses_tiny_positive_count():
    from readmit_audit.fairness import bootstrap_tpr_ci
    tpr, lo, hi, n = bootstrap_tpr_ci(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert np.isnan(tpr) and n == 2
