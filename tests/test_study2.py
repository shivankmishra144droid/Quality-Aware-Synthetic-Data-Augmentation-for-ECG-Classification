"""Study 2 code (Phase 1 of report/WORKFLOW_2.md). Fake data only: no real val/test array is read.

Run from repo root: py -3.14 -m pytest tests -q
"""
import json

import numpy as np
import pytest
import torch

import _study2_analysis as A
from generator_comparison import augment_experiment as E


def fake_split(n, rng, length=200):
    y = np.concatenate([np.arange(4), rng.integers(0, 4, n - 4)])        # every class present
    return rng.normal(size=(n, length, 12)).astype(np.float32), y.astype(np.int64)


@pytest.mark.parametrize("n_train", [20, 50])
def test_train_steps_uses_exactly_the_budget(monkeypatch, n_train):
    rng = np.random.default_rng(0)
    Xtr, ytr = fake_split(n_train, rng)
    Xva, yva = fake_split(12, rng)
    calls = []
    orig = torch.optim.Adam.step
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1])
    _, info = E.train_steps(Xtr, ytr, Xva, yva, seed=0, dev=torch.device("cpu"), steps=7, eval_every=3, batch=8)
    assert len(calls) == 7 and info["steps"] == 7
    assert len(info["val_history"]) == 3                 # steps 3, 6 and the last step
    assert info["best_step"] in (3, 6, 7)


def test_val_only_never_touches_test(monkeypatch, tmp_path):
    rng = np.random.default_rng(1)
    data, out = tmp_path / "data", tmp_path / "exp"
    data.mkdir()
    Xtr, ytr = fake_split(24, rng)
    Xva, yva = fake_split(12, rng)
    for name, arr in [("X_train", Xtr), ("y_train", ytr), ("X_val", Xva), ("y_val", yva)]:
        np.save(data / f"{name}.npy", arr)                 # no X_test / y_test: loading them would fail
    counts = {c: int((ytr == c).sum()) for c in (1, 2, 3)}
    py = np.repeat([1, 2, 3], [3 * counts[1], 3 * counts[2], 3 * counts[3]])
    np.savez(tmp_path / "pool.npz", X=rng.normal(size=(len(py), 200, 12)).astype(np.float32), y=py)
    s = rng.random(len(py))
    np.savez(tmp_path / "scores.npz", y=py, score=s, score_nolabel=s)
    monkeypatch.setattr(E, "DATA", data)
    monkeypatch.setattr(E, "OUT", out)
    E.main([str(tmp_path / "pool.npz"), str(tmp_path / "scores.npz"), "--seeds", "1", "--seed-start", "200",
            "--conditions", "real,oversample,random,filtered", "--steps", "2", "--eval-every", "1", "--val-only",
            "--tag", "check"])
    res = json.loads((out / "check" / "results.json").read_text())
    assert set(res) == {f"{c}/seed200" for c in ("real", "oversample", "random", "filtered")}
    assert all("test" not in r and r["steps"] == 2 for r in res.values())
    assert not list((out / "check").glob("test_prob_*"))
    assert json.loads((out / "check" / "config.json").read_text())["val_only"] is True


def write_fake_study(d, f1_of, rng, n_test=120):
    """f1_of(cond, pool, seed) -> test macro-F1 written into results.json; probabilities are random."""
    for k in A.POOLS:
        res = {}
        (d / f"pool{k}").mkdir(parents=True)
        for c in A.CONDS:
            for s in A.SEEDS[k]:
                res[f"{c}/seed{s}"] = {"test": {"macro_f1": f1_of(c, k, s)}}
                np.save(d / f"pool{k}" / f"test_prob_{c}_seed{s}.npy", rng.random((n_test, 4)))
        (d / f"pool{k}" / "results.json").write_text(json.dumps(res))
    return rng.integers(0, 4, n_test), np.repeat(np.arange(n_test // 2), 2)


def test_analysis_detects_a_clear_effect(tmp_path):
    rng = np.random.default_rng(2)
    noise = {(c, s): rng.normal(0, 0.003) for c in A.CONDS for k in A.POOLS for s in A.SEEDS[k]}
    base = {"real": 0.79, "oversample": 0.81, "random": 0.83, "filtered": 0.86}
    y, pid = write_fake_study(tmp_path, lambda c, k, s: base[c] + noise[c, s], rng)
    out = A.analyse(tmp_path, y, pid, boot=20)
    h = out["hypotheses"]
    assert h["H1"]["decision"].startswith("SUPPORTED") and h["H1"]["mean_diff"] > 0.02
    assert h["H2"]["decision"].startswith("SUPPORTED") and h["H3"]["decision"].startswith("SUPPORTED")
    assert all(h[x]["holm_p"] >= h[x]["wilcoxon_p"] for x in h)
    assert "H1" in A.to_md(out)


def test_analysis_null_with_equivalence(tmp_path):
    rng = np.random.default_rng(3)
    noise = {(c, s): rng.normal(0, 0.002) for c in A.CONDS for k in A.POOLS for s in A.SEEDS[k]}
    y, pid = write_fake_study(tmp_path, lambda c, k, s: 0.83 + noise[c, s], rng)
    h1 = A.analyse(tmp_path, y, pid, boot=20)["hypotheses"]["H1"]
    assert h1["holm_p"] >= A.ALPHA
    assert h1["tost"]["equivalent"] and "equivalent" in h1["decision"]


def test_analysis_refuses_incomplete_results(tmp_path):
    rng = np.random.default_rng(4)
    write_fake_study(tmp_path, lambda c, k, s: 0.8, rng)
    res = json.loads((tmp_path / "pool3" / "results.json").read_text())
    del res["filtered/seed221"]
    (tmp_path / "pool3" / "results.json").write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="Incomplete"):
        A.load(tmp_path)
