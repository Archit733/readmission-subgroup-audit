import numpy as np
import pandas as pd
from readmit_audit.data import split_data


def _df(n_patients=400, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_patients), rng.integers(1, 5, n_patients))
    return pd.DataFrame({"patient_nbr": pid, "target": rng.integers(0, 2, len(pid)),
                         "f": rng.normal(size=len(pid))})


def _cfg(group="patient_nbr"):
    return {"split": {"test_size": 0.2, "random_state": 42, "group_column": group}}


def test_no_patient_appears_in_both_sides():
    df = _df(); X = df[["f"]]
    _, _, _, _, tr, te = split_data(df, X, _cfg())
    assert set(df.loc[tr, "patient_nbr"]).isdisjoint(set(df.loc[te, "patient_nbr"]))


def test_test_share_is_about_twenty_percent():
    df = _df(); _, Xte, _, _, _, _ = split_data(df, df[["f"]], _cfg())
    assert 0.15 < len(Xte) / len(df) < 0.25


def test_positive_rate_roughly_preserved():
    df = _df(800); _, _, ytr, yte, _, _ = split_data(df, df[["f"]], _cfg())
    assert abs(ytr.mean() - yte.mean()) < 0.05


def test_group_split_can_be_disabled():
    df = _df(); out = split_data(df, df[["f"]], _cfg(group=None))
    assert abs(len(out[1]) - len(df) * 0.2) <= 1


def test_split_is_reproducible():
    df = _df(); a = split_data(df, df[["f"]], _cfg())[5]; b = split_data(df, df[["f"]], _cfg())[5]
    assert list(a) == list(b)
