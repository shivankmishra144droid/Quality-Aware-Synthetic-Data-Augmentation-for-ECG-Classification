"""Part A of report/PREREGISTRATION_3.md (threshold-tuned real-only control). Fake data only.

Run from repo root: py -3.14 -m pytest tests -q
"""
import json

import numpy as np
import pytest

import _run_threshold_control as R
import _threshold_analysis as A
from generator_comparison import augment_experiment as E
from generator_comparison.threshold_tuning import apply_weights, fit_class_weights, half_split_optimism, macro_f1


def shifted_probs(n, rng, bias):
    """Well-separated true class scores, then a prior shift `bias` (log space) that a tuned weight should undo."""
    y = rng.integers(0, 4, n)
    logits = rng.normal(0, 1, (n, 4))
    logits[np.arange(n), y] += 3.0
    logits += np.asarray(bias)[None, :]
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), y


def test_weights_undo_a_known_prior_shift():
    rng = np.random.default_rng(0)
    prob, y = shifted_probs(4000, rng, [2.5, 0, 0, -2.0])      # SR over-predicted, SBRAD under-predicted
    w = fit_class_weights(prob, y)
    assert w[0] == 1.0
    assert macro_f1(y, prob, w) > macro_f1(y, prob) + 0.05
    assert w[3] > w[1] and w[3] > w[2]                           # SBRAD gets the biggest boost


def test_no_shift_means_no_change():
    rng = np.random.default_rng(1)
    prob, y = shifted_probs(2000, rng, [0, 0, 0, 0])
    w = fit_class_weights(prob, y)
    assert macro_f1(y, prob, w) >= macro_f1(y, prob)
    assert np.allclose(apply_weights(prob, np.ones(4)), prob)


def test_half_split_reports_both_gains():
    rng = np.random.default_rng(2)
    prob, y = shifted_probs(1000, rng, [2.0, 0, 0, -1.5])
    o = half_split_optimism(prob, y, np.repeat(np.arange(500), 2))
    assert o["insample_gain"] >= o["heldout_gain"] - 0.02 and o["heldout_gain"] > 0


def fake_data(monkeypatch, tmp_path, n_pool_mult=3):
    rng = np.random.default_rng(3)
    data, exp = tmp_path / "data", tmp_path / "exp"
    data.mkdir()

    def split(n):
        y = np.concatenate([np.arange(4), rng.integers(0, 4, n - 4)]).astype(np.int64)
        return rng.normal(size=(n, 200, 12)).astype(np.float32), y

    for name, n in (("train", 24), ("val", 16), ("test", 16)):
        X, y = split(n)
        np.save(data / f"X_{name}.npy", X)
        np.save(data / f"y_{name}.npy", y)
    import pandas as pd
    pd.DataFrame({"patient_id": np.repeat(np.arange(8), 2)}).to_csv(data / "val_metadata.csv", index=False)
    ytr = np.load(data / "y_train.npy")
    py = np.repeat([1, 2, 3], [n_pool_mult * int((ytr == c).sum()) for c in (1, 2, 3)])
    pools = tmp_path / "pools"
    pools.mkdir()
    for k in (1, 2):
        np.savez(pools / f"pool_{k}.npz", X=rng.normal(size=(len(py), 200, 12)).astype(np.float32), y=py)
    monkeypatch.setattr(E, "DATA", data)
    monkeypatch.setattr(E, "OUT", exp)
    monkeypatch.setattr(R, "POOL_DIR", pools)
    monkeypatch.setattr(R, "STUDY2", tmp_path / "no_study2")
    return exp / R.TAG, pools


def test_runner_saves_val_and_test_probs_and_resumes(monkeypatch, tmp_path):
    out, _ = fake_data(monkeypatch, tmp_path)
    args = ["--pools", "1", "--steps", "2", "--eval-every", "1", "--no-deterministic", "--seeds-per-pool", "1"]
    R.main(args)
    res = json.loads((out / "pool1" / "results.json").read_text())
    assert set(res) == {"real/seed200", "random/seed200"}
    for key, r in res.items():
        cond = key.split("/")[0]
        assert (out / "pool1" / f"val_prob_{cond}_seed200.npy").exists()
        assert (out / "pool1" / f"test_prob_{cond}_seed200.npy").exists()
        assert {"test", "test_tuned", "val_tuned", "class_weights_tuned", "val_optimism"} <= set(r)
    assert res["random/seed200"]["n_synthetic"] > 0 and res["real/seed200"]["n_synthetic"] == 0
    R.main(args)                                               # same recipe: skips finished runs
    with pytest.raises(SystemExit, match="different"):
        R.main(["--pools", "1", "--steps", "3", "--eval-every", "1", "--no-deterministic", "--seeds-per-pool", "1"])


def write_fake_runs(d, f1_shift, rng, n_test=120):
    """Writes results + test probabilities for all 80 runs. Real-only models are biased towards SR; `f1_shift`
    controls how much better the random arm separates the classes."""
    y = rng.integers(0, 4, n_test)
    for k in A.POOLS:
        (d / f"pool{k}").mkdir(parents=True)
        res = {}
        for c in A.BASE:
            for s in A.SEEDS[k]:
                logits = rng.normal(0, 1, (n_test, 4))
                logits[np.arange(n_test), y] += 2.0 + (f1_shift if c == "random" else 0.0)
                logits[:, 0] += 1.5 if c == "real" else 0.0
                p = np.exp(logits)
                p /= p.sum(1, keepdims=True)
                np.save(d / f"pool{k}" / f"test_prob_{c}_seed{s}.npy", p)
                w = fit_class_weights(p, y)                    # stands in for the val fit
                res[f"{c}/seed{s}"] = {"class_weights_tuned": dict(zip(A.CLASS_ORDER, map(float, w))),
                                       "val_optimism": {"insample_gain": 0.01, "heldout_gain": 0.005},
                                       "reproduction": {"reference": None}}
        (d / f"pool{k}" / "results.json").write_text(json.dumps(res))
    return y, np.repeat(np.arange(n_test // 2), 2)


def test_analysis_runs_and_decides(tmp_path):
    rng = np.random.default_rng(4)
    y, pid = write_fake_runs(tmp_path, f1_shift=1.5, rng=rng)
    out = A.analyse(tmp_path, y, pid, boot=10)
    assert set(out["condition_means"]) == set(A.CONDS)
    assert out["hypotheses"]["T2"]["mean_diff"] > 0          # tuning helps the SR-biased real-only model
    assert out["decision"].startswith("AUGMENTATION DOES REAL WORK")
    assert "Decision" in A.to_md(out)


def test_analysis_refuses_incomplete(tmp_path):
    (tmp_path / "pool1").mkdir()
    with pytest.raises(SystemExit, match="Incomplete"):
        A.load(tmp_path)
