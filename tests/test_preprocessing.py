import numpy as np
import pandas as pd
import pytest
from readmit_audit.preprocessing import build_preprocessor, infer_column_types


def test_splits_types():
    X = pd.DataFrame({"c": ["a", "b"], "n": [1, 2]})
    assert infer_column_types(X) == (["c"], ["n"])


def test_no_nans_after_imputation():
    X = pd.DataFrame({"race": ["A", "B", "A", np.nan], "n": [1.0, 2.0, np.nan, 4.0]})
    out = build_preprocessor(X).fit_transform(X)
    assert out.shape == (4, 2) and np.isfinite(out).all()


def test_unseen_category_does_not_crash():
    pre = build_preprocessor(pd.DataFrame({"race": ["A", "B", "A"]}))
    pre.fit(pd.DataFrame({"race": ["A", "B", "A"]}))
    assert pre.transform(pd.DataFrame({"race": ["Zzz"]}))[0, 0] == -1


def test_same_category_same_code():
    X = pd.DataFrame({"race": ["A", "B", "C"]})
    pre = build_preprocessor(X).fit(X)
    assert pre.transform(pd.DataFrame({"race": ["A"]}))[0, 0] == pre.transform(pd.DataFrame({"race": ["A"]}))[0, 0]


def test_numeric_median_imputed():
    X = pd.DataFrame({"n": [1.0, np.nan, 3.0]})
    assert build_preprocessor(X).fit_transform(X)[1, 0] == pytest.approx(2.0)
