"""Regression tests for the 2026-10-28 review fixes. Touch no val/test data.

Run from repo root: py -3.14 -m pytest tests -q
"""
from pathlib import Path

import numpy as np
import pytest

from generator_comparison.augment_experiment import pick
from generator_comparison.quality_filter import realism_within_fold, select
from synthetic_generation.assembly import MEAN_RR_BOUNDS, RhythmModel

POOL_SCORES = Path(__file__).resolve().parent.parent / "generator_comparison" / "final_pool" / "scores.npz"


def true_top_k(scores, y, c, k):
    """Top-k with ties at the cut-off broken by pool order, the documented behaviour of select()."""
    ic = np.where(y == c)[0]
    return set(ic[np.argsort(-scores[ic], kind="stable")[:k]])


def test_select_breaks_ties_by_pool_order():
    y = np.array([1] * 5)
    s = np.array([0.9, 0.5, 0.7, 0.5, 0.5])                 # three-way tie for places 3-5
    assert list(select(s, y, {1: 3})) == [0, 1, 2]           # the earliest tied sample wins, every time


@pytest.mark.parametrize("cond", ["filtered", "filtered_nolabel"])
def test_pick_filtered_is_per_class_top_k(cond):
    # Scores fall with pool position, so slicing by pool order (the old bug) would pick the best by luck;
    # make them RISE with position instead, so pool order and score order disagree.
    y = np.repeat([1, 2, 3], [300, 200, 100])
    rng = np.random.default_rng(0)
    s = np.arange(len(y)) + rng.normal(0, 5, len(y))
    scores = {"score": s, "score_nolabel": s}
    counts = {1: 300, 2: 50, 3: 20}          # k differs per class and is below max(k) for 2 and 3
    sel = pick(cond, y, scores, counts, seed=0)
    for c, k in counts.items():
        got = sel[y[sel] == c]
        assert len(got) == k
        assert set(got) == true_top_k(s, y, c, k)


def test_select_int_and_dict_agree():
    y = np.repeat([0, 1, 2, 3], 50)
    s = np.random.default_rng(1).random(len(y))
    assert np.array_equal(select(s, y, 10), select(s, y, dict.fromkeys(range(4), 10)))


@pytest.mark.skipif(not POOL_SCORES.exists(), reason="final_pool/scores.npz not built")
def test_pick_on_final_pool_scores():
    d = dict(np.load(POOL_SCORES))
    counts = {1: 1056, 2: 568, 3: 446}
    sel = pick("filtered", d["y"], d, counts, seed=0)
    for c, k in counts.items():
        assert set(sel[d["y"][sel] == c]) == true_top_k(d["score"], d["y"], c, k)


def test_realism_within_fold_removes_fold_offset():
    rng = np.random.default_rng(0)
    n, folds = 5000, 5
    y = rng.integers(1, 4, n)
    fold = rng.permutation(n) % folds
    quality = rng.normal(size=n)
    logodds = quality + 3.0 * fold                       # each fold detector on its own scale
    r = realism_within_fold(logodds, fold, y)
    for c in (1, 2, 3):
        m = y == c
        thr = np.quantile(r[m], 2 / 3)
        shares = [(r[m & (fold == k)] > thr).mean() for k in range(folds)]
        assert max(shares) - min(shares) < 0.1, shares
    raw_top = logodds > np.quantile(logodds, 2 / 3)
    assert raw_top[fold == 4].mean() > 0.9               # sanity: the raw values were fold-driven


def _kde_straddling(lo, hi):
    rng = np.random.default_rng(0)
    rr = np.exp(rng.uniform(np.log(lo) - 0.4, np.log(hi) + 0.4, 400))
    return RhythmModel(rr, np.full(400, 0.05), irregular=False)


@pytest.mark.parametrize("cls", [0, 2, 3])
def test_rhythm_mean_rr_within_class_bounds(cls):
    lo, hi = MEAN_RR_BOUNDS[cls]
    m = _kde_straddling(lo, hi)
    m.mean_rr_bounds = (lo, hi)
    rng = np.random.default_rng(1)
    means = np.array([m.sample_mean_cv(rng)[0] for _ in range(2000)])
    assert means.min() >= lo and means.max() <= hi


@pytest.mark.parametrize("cls", [0, 2, 3])
def test_realised_rate_of_placed_beats_within_class(cls):
    lo, hi = MEAN_RR_BOUNDS[cls]
    m = _kde_straddling(lo, hi)
    m.mean_rr_bounds = (lo, hi)
    m.kde.set_bandwidth(m.kde.factor * 3)                 # wider KDE: more draws near and over the boundary
    rng = np.random.default_rng(2)
    realised = np.array([np.diff(m.sample_r_positions(rng)).mean() / 100 for _ in range(1000)])
    assert realised.min() > lo and realised.max() < hi


def test_rhythm_without_bounds_matches_legacy_rng_use():
    m = _kde_straddling(0.3, 2.0)
    m.mean_rr_bounds = None
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(200):
        lm, lc = m.kde.resample(1, seed=b)[:, 0]                     # pre-fix code: one draw, clip
        assert m.sample_mean_cv(a) == (float(np.clip(np.exp(lm), 0.3, 2.0)), float(np.exp(lc)))
