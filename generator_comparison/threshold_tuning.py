"""Decision-threshold tuning for the 4-class CNN-LSTM (report/PREREGISTRATION_3.md, part A).

Multiclass "thresholds" are per-class multiplicative weights on the softmax: predict argmax_c(p_c * w_c), with
w_SR = 1. Weights are fitted on VALIDATION probabilities by coordinate ascent of macro-F1 over a fixed log grid and
then applied unchanged to test probabilities. Everything here is a pure function of arrays.
"""
from __future__ import annotations

import numpy as np

from generator_comparison.stats import fast_f1

LOG_GRID = np.linspace(-3.0, 3.0, 61)   # w in [e^-3, e^3], step 0.1 in log space; contains log w = 0
SWEEPS = 3
TUNED_CLASSES = (1, 2, 3)               # SR (0) is the reference, w_SR = 1


def apply_weights(prob: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reweighted, renormalised probabilities; argmax gives the tuned prediction."""
    q = prob * np.asarray(w, dtype=np.float64)[None, :]
    return q / q.sum(axis=1, keepdims=True)


def macro_f1(y: np.ndarray, prob: np.ndarray, w=None) -> float:
    pred = (prob if w is None else prob * np.asarray(w)[None, :]).argmax(1)
    return float(fast_f1(y, pred)[0])


def fit_class_weights(prob: np.ndarray, y: np.ndarray, grid=LOG_GRID, sweeps: int = SWEEPS) -> np.ndarray:
    """Coordinate ascent of macro-F1 over log w_c for c in AFIB, STACH, SBRAD, starting at w = 1.
    Ties go to the grid value closest to log w = 0, so no evidence means no change."""
    logw = np.zeros(prob.shape[1])
    order = np.argsort(np.abs(grid), kind="stable")          # candidates nearest to 0 first -> wins ties
    for _ in range(sweeps):
        for c in TUNED_CLASSES:
            best_v, best_f = logw[c], -1.0
            for v in grid[order]:
                logw[c] = v
                f = macro_f1(y, prob, np.exp(logw))
                if f > best_f + 1e-12:
                    best_v, best_f = v, f
            logw[c] = best_v
    return np.exp(logw)


def half_split_optimism(prob: np.ndarray, y: np.ndarray, pid: np.ndarray, seed: int = 0) -> dict:
    """How much of the validation gain from tuning is overfitting: split val patients into two halves, tune on
    one, score on the other (both ways). Returns the in-sample gain and the held-out gain (macro-F1)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(pid)
    half = set(rng.permutation(pats)[: len(pats) // 2].tolist())
    a = np.array([p in half for p in pid])
    held, insample = [], []
    for tr, te in ((a, ~a), (~a, a)):
        w = fit_class_weights(prob[tr], y[tr])
        held.append(macro_f1(y[te], prob[te], w) - macro_f1(y[te], prob[te]))
        insample.append(macro_f1(y[tr], prob[tr], w) - macro_f1(y[tr], prob[tr]))
    return {"insample_gain": float(np.mean(insample)), "heldout_gain": float(np.mean(held))}
