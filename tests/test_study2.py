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


def fake_setup(monkeypatch, tmp_path, seed=1):
    """Fake train/val split (no test files at all), pool and scores; points the experiment at them."""
    rng = np.random.default_rng(seed)
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
    return out, [str(tmp_path / "pool.npz"), str(tmp_path / "scores.npz")]


def test_val_only_never_touches_test(monkeypatch, tmp_path):
    out, files = fake_setup(monkeypatch, tmp_path)
    E.main(files + ["--seeds", "1", "--seed-start", "200", "--conditions", "real,oversample,random,filtered",
                    "--steps", "2", "--eval-every", "1", "--val-only", "--tag", "check"])
    res = json.loads((out / "check" / "results.json").read_text())
    assert set(res) == {f"{c}/seed200" for c in ("real", "oversample", "random", "filtered")}
    assert all("test" not in r and r["steps"] == 2 for r in res.values())
    assert not list((out / "check").glob("test_prob_*"))
    assert json.loads((out / "check" / "config.json").read_text())["val_only"] is True


def test_deterministic_refuses_without_cuda(monkeypatch, tmp_path):
    out, files = fake_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(E, "set_deterministic", lambda: None)          # keep the test process's torch settings
    with pytest.raises(SystemExit, match="needs CUDA"):
        E.main(files + ["--seeds", "1", "--conditions", "real", "--steps", "2", "--deterministic", "--val-only",
                        "--tag", "nocuda"])
    assert not (out / "nocuda" / "results.json").exists()


def test_resume_refuses_changed_recipe(monkeypatch, tmp_path):
    out, files = fake_setup(monkeypatch, tmp_path)
    base = files + ["--seeds", "1", "--conditions", "real", "--eval-every", "1", "--val-only", "--tag", "resume"]
    E.main(base + ["--steps", "2"])
    E.main(base + ["--steps", "2"])                                     # same recipe: resumes (skips) fine
    with pytest.raises(SystemExit, match="different"):
        E.main(base + ["--steps", "3"])
    p = dict(np.load(tmp_path / "pool.npz"))
    p["X"] = p["X"] + 1                                                 # a rebuilt pool
    np.savez(tmp_path / "pool.npz", **p)
    with pytest.raises(SystemExit, match="pool_sha256"):
        E.main(base + ["--steps", "2"])


@pytest.mark.parametrize("diffs, ok", [
    ([0.02, 0.01, 0.03, -0.01], True),          # 3 of 4, mean > 0
    ([0.05, 0.05, -0.01, -0.01], False),        # only 2 of 4, even though the mean is positive
    ([0.001, 0.001, 0.001, -0.02], False),      # 3 of 4 but mean < 0
    ([0.0, 0.01, 0.01, 0.01], True),            # a tie does not count as a win
    ([0.0, 0.0, 0.01, 0.01], False),
])
def test_gate_g1_rule(diffs, ok):
    import _gate_study2 as G
    assert G.g1_decision(diffs)["pass"] is ok


def write_fake_study(d, f1_of, rng, n_test=120):
    """f1_of(cond, pool, seed) -> test macro-F1 written into results.json; probabilities are random."""
    for k in A.POOLS:
        res = {}
        (d / f"pool{k}").mkdir(parents=True)
        for c in A.CONDS:
            for s in A.SEEDS[k]:
                m = f1_of(c, k, s)
                res[f"{c}/seed{s}"] = {"test": {"macro_f1": m, "f1": {cl: m for cl in ("SR", "AFIB", "STACH", "SBRAD")},
                                                "recall": {cl: m for cl in ("SR", "AFIB", "STACH", "SBRAD")}}}
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
    assert h["H1"]["adjusted_p"] == h["H1"]["wilcoxon_p"]                  # primary: unadjusted
    assert all(h[x]["adjusted_p"] >= h[x]["wilcoxon_p"] for x in ("H2", "H3"))
    md = A.to_md(out)
    assert "H1" in md and "Secondary (descriptive only" in md
    ds = out["descriptive"]
    assert set(ds["per_condition"]) == set(A.CONDS)
    assert all(0 <= x <= 1 for v in ds["per_condition"].values() for x in v["auprc"].values())
    assert ds["sbrad_f1_diff"]["H1"]["mean"] > 0.02 and ds["sbrad_f1_diff"]["H1"]["n"] == 40


def test_analysis_null_with_equivalence(tmp_path):
    rng = np.random.default_rng(3)
    noise = {(c, s): rng.normal(0, 0.002) for c in A.CONDS for k in A.POOLS for s in A.SEEDS[k]}
    y, pid = write_fake_study(tmp_path, lambda c, k, s: 0.83 + noise[c, s], rng)
    h1 = A.analyse(tmp_path, y, pid, boot=20)["hypotheses"]["H1"]
    assert h1["adjusted_p"] >= A.ALPHA
    assert h1["tost"]["equivalent"] and "equivalent" in h1["decision"]


def test_analysis_refuses_incomplete_results(tmp_path):
    rng = np.random.default_rng(4)
    write_fake_study(tmp_path, lambda c, k, s: 0.8, rng)
    res = json.loads((tmp_path / "pool3" / "results.json").read_text())
    del res["filtered/seed221"]
    (tmp_path / "pool3" / "results.json").write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="Incomplete"):
        A.load(tmp_path)


def test_fast_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    from generator_comparison.stats import fast_f1
    rng = np.random.default_rng(5)
    for i in range(20):
        y = rng.integers(0, 4, 300)
        p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 4, 300))   # ~70 % right, like a weak classifier
        if i == 0:
            p[p == 3] = 0                                                  # a class never predicted (F1 0, no NaN)
        macro, sbrad = fast_f1(y, p)
        assert np.isclose(macro, f1_score(y, p, average="macro", labels=range(4), zero_division=0))
        assert np.isclose(sbrad, f1_score(y, p, average=None, labels=range(4), zero_division=0)[3])


def test_holm_matches_reference():
    from generator_comparison.stats import holm
    # Reference values computed by hand: sorted p (0.01, 0.03, 0.04) -> 3*0.01, max(0.03, 2*0.03), max(0.06, 0.04)
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_launcher_seeds_and_tags(monkeypatch):
    import runpy
    calls = []
    monkeypatch.setattr(E, "main", lambda argv=None: calls.append(argv))
    monkeypatch.setattr("sys.argv", ["_run_study2.py"])
    runpy.run_path("_run_study2.py", run_name="__main__")
    assert len(calls) == 4
    for k, argv in enumerate(calls, start=1):
        val = lambda flag: argv[argv.index(flag) + 1]
        assert argv[:2] == [f"generator_comparison/pools2/pool_{k}.npz", f"generator_comparison/pools2/scores_{k}.npz"]
        assert int(val("--seed-start")) == 200 + 10 * (k - 1) and val("--seeds") == "10"
        assert val("--conditions") == "real,oversample,random,filtered" and val("--steps") == "6250"
        assert val("--tag") == f"study2/pool{k}" and "--deterministic" in argv and "--val-only" not in argv
    seeds = sorted(int(c[c.index("--seed-start") + 1]) + i for c in calls for i in range(10))
    assert seeds == list(range(200, 240))                                  # 40 distinct, never-used seeds


def test_gate_g2_reads_what_the_runs_write(monkeypatch, tmp_path):
    import _gate_study2 as G
    monkeypatch.chdir(tmp_path)
    history = iter([[0.5, 0.6], [0.5, 0.6], [0.5, 0.6], [0.5, 0.61]])

    def fake_main(argv):
        tag = argv[argv.index("--tag") + 1]
        assert "--val-only" in argv and tag.startswith("study2_check/")
        d = tmp_path / "generator_comparison" / "experiment" / tag
        d.mkdir(parents=True)
        h = next(history)
        (d / "results.json").write_text(json.dumps({"filtered/seed200": {
            "val": {"macro_f1": h[-1]}, "val_history": h, "best_step": 500}}))

    monkeypatch.setattr(E, "main", fake_main)
    assert G.g2()["pass"] is True                                          # identical histories
    assert G.g2()["pass"] is False                                         # differs in the last evaluation
