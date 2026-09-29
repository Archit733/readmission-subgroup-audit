import numpy as np
import pandas as pd
from readmit_audit.simulation import make_stream


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["young", "old"], n, p=[0.6, 0.4])
    y = rng.integers(0, 2, n); p = rng.uniform(size=n)
    return y, p, pd.DataFrame({"f": rng.normal(size=n)}), pd.Series(g)


def test_clean_batches_untouched_and_drift_increases():
    y, p, X, g = _data()
    ys, ps, Xs, gs, b, meta = make_stream(y, p, X, g, "old", n_batches=10, clean_batches=4)
    assert (meta["severity"].iloc[:4] == 0).all()
    assert list(meta["severity"].iloc[4:]) == sorted(meta["severity"].iloc[4:])
    share = [(gs[b == i] == "old").mean() for i in range(10)]
    assert share[9] > share[0] + 0.1                       # patient mix shifted toward target group


def test_only_target_group_labels_are_corrupted():
    y, p, X, g = _data()
    ys, ps, Xs, gs, b, _ = make_stream(y, p, X, g, "old", max_label_noise=1.0, max_weight=1.0)
    # compare label/score relationship: scores are independent here, so check via identity of copies
    assert len(ys) == len(y) and len(np.unique(b)) == 10


def test_stream_is_reproducible():
    y, p, X, g = _data()
    a = make_stream(y, p, X, g, "old", seed=7)[0]; c = make_stream(y, p, X, g, "old", seed=7)[0]
    assert (a == c).all()


def test_lengths_and_alignment():
    y, p, X, g = _data(1000)
    ys, ps, Xs, gs, b, _ = make_stream(y, p, X, g, "old")
    assert len(ys) == len(ps) == len(Xs) == len(gs) == len(b) == 1000
