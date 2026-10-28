"""Phase 3 of report/WORKFLOW_2.md: the validation gate for study 2. Validation split only; never loads test data.

G1 (filter is useful on val): per pool, _filter_check.py TSTR balanced accuracy on real val, top-440 vs random-440
    (3 seeds). PASS if top > random in at least 3 of 4 pools AND the mean (top - random) over the 4 pools is > 0.
G2 (runs repeat exactly): the study-2 recipe (pool 1, seed 200, filtered, --val-only) run twice in separate output
    folders gives identical validation metrics and validation history.
Writes generator_comparison/pools2/gate.json. Filter checks already written are reused, not rerun.
Fixed before any gate result was seen (2026-10-29).

Usage: py -3.14 _gate_study2.py"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

D = Path("generator_comparison/pools2")
POOLS = (1, 2, 3, 4)
K = 440
MIN_POOLS_WON = 3


def g1_decision(diffs):
    """diffs: per-pool (top - random) balanced accuracy."""
    won = sum(d > 0 for d in diffs)
    mean = sum(diffs) / len(diffs)
    return {"per_pool_top_minus_random": list(diffs), "pools_won": won, "mean": mean,
            "pass": bool(won >= MIN_POOLS_WON and mean > 0)}


def g1():
    diffs = []
    for k in POOLS:
        out = D / f"filter_check_{k}"
        if not out.with_suffix(".json").exists():
            subprocess.run([sys.executable, "_filter_check.py", str(D / f"pool_{k}.npz"), str(D / f"scores_{k}.npz"),
                            str(K), str(out)], check=True)
        r = json.loads(out.with_suffix(".json").read_text())["results"]
        diffs.append(r["top"]["balanced_acc_mean"] - r["random"]["balanced_acc_mean"])
    return g1_decision(diffs)


def g2():
    from _run_study2 import args_for
    from generator_comparison import augment_experiment
    runs = {}
    for name in ("a", "b"):
        folder = Path("generator_comparison/experiment/study2_check") / name
        if folder.exists():
            shutil.rmtree(folder)                       # always a fresh run, never a skipped one
        augment_experiment.main(args_for(1, 200, 1, "filtered", f"study2_check/{name}", val_only=True))
        r = json.loads((folder / "pool1" / "results.json").read_text())["filtered/seed200"]
        assert "test" not in r and not list((folder / "pool1").glob("test_prob_*")), "val-only run touched test"
        runs[name] = {"val": r["val"], "val_history": r["val_history"], "best_step": r["best_step"]}
    return {"run_a": runs["a"]["val_history"], "run_b": runs["b"]["val_history"],
            "pass": runs["a"] == runs["b"]}


if __name__ == "__main__":
    res = {"G1": g1(), "G2": g2()}
    res["GO"] = res["G1"]["pass"] and res["G2"]["pass"]
    (D / "gate.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
