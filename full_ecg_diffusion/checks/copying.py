"""Copying check: nearest-neighbour correlation to the generator's training records.

The same statistic for held-out real records (never seen by the generator) is the baseline,
since real ECGs of the same class are naturally somewhat alike.
"""
from __future__ import annotations

import numpy as np


def _unit_rows(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    z = z - z.mean(axis=1, keepdims=True)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def nn_corr(x: np.ndarray, ref: np.ndarray, chunk: int = 512) -> np.ndarray:
    r = _unit_rows(ref)
    out = []
    for s in range(0, len(x), chunk):
        out.append((_unit_rows(x[s:s + chunk]) @ r.T).max(axis=1))
    return np.concatenate(out) if out else np.zeros(0)


def copying_check(x_syn: np.ndarray, x_heldout_real: np.ndarray, x_ref: np.ndarray, th) -> dict:
    s = nn_corr(x_syn, x_ref)
    r = nn_corr(x_heldout_real, x_ref)
    frac_s, frac_r = float((s > th.copy_corr).mean()), float((r > th.copy_corr).mean())
    return {
        "n_ref": int(len(x_ref)),
        "syn": {"mean_nn_corr": float(s.mean()), "p99_nn_corr": float(np.percentile(s, 99)),
                "frac_gt_thr": frac_s, "n_flagged": int((s > th.copy_corr).sum())},
        "heldout_real": {"mean_nn_corr": float(r.mean()), "p99_nn_corr": float(np.percentile(r, 99)),
                         "frac_gt_thr": frac_r},
        "flagged_idx": np.flatnonzero(s > th.copy_corr)[:50].tolist(),
        "pass": bool(frac_s <= frac_r + th.copy_frac_margin and s.mean() <= r.mean() + th.copy_mean_margin),
    }
