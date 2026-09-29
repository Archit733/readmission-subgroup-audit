"""Simulated post-deployment stream for exercising the monitoring layer.

The held-out test set is cut into sequential batches. The first `clean_batches`
are left alone; later batches get progressively stronger, clearly-labelled
synthetic drift of two kinds:
  * covariate drift: the patient mix shifts toward `target_group` (resampling)
  * outcome drift: for `target_group` only, a growing share of labels is replaced
    by random draws (as if the outcome definition or follow-up process changed)
This is a stress test of the alerting logic, NOT a claim about real-world drift.
"""
import numpy as np
import pandas as pd


def make_stream(y, p, X: pd.DataFrame, group: pd.Series, target_group: str, n_batches: int = 10,
                clean_batches: int = 4, max_weight: float = 3.0, max_label_noise: float = 0.5,
                seed: int = 42):
    """Returns (y, p, X, group, batch_ids, drift_meta) for the stream."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); group = np.asarray(group)
    order = rng.permutation(len(y))
    chunks = np.array_split(order, n_batches)
    base_rate = float(y.mean())
    ys, ps, gs, bs, idxs, meta = [], [], [], [], [], []
    n_drift = max(1, n_batches - clean_batches)
    for b, idx in enumerate(chunks):
        severity = 0.0 if b < clean_batches else (b - clean_batches + 1) / n_drift
        if severity > 0:
            w = np.where(group[idx] == target_group, 1 + (max_weight - 1) * severity, 1.0)
            idx = rng.choice(idx, size=len(idx), replace=True, p=w / w.sum())
        yb = y[idx].copy()
        if severity > 0:
            hit = (group[idx] == target_group) & (rng.uniform(size=len(idx)) < max_label_noise * severity)
            yb[hit] = (rng.uniform(size=hit.sum()) < base_rate).astype(int)
        ys.append(yb); ps.append(p[idx]); gs.append(group[idx]); bs.append(np.full(len(idx), b)); idxs.append(idx)
        meta.append({"batch": b, "severity": round(severity, 3)})
    idx_all = np.concatenate(idxs)
    return (np.concatenate(ys), np.concatenate(ps), X.iloc[idx_all].reset_index(drop=True),
            np.concatenate(gs), np.concatenate(bs), pd.DataFrame(meta))
