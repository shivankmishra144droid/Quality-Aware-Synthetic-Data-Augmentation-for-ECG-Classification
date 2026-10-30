"""Part A of report/PREREGISTRATION_3.md: retrain the study-2 `real` and `random` runs and keep VALIDATION
probabilities, so decision thresholds can be tuned on validation and applied to test.

Same 40 units as study 2 (pools 1-4, seeds 200-239), same recipe (augment_experiment.train_steps: 6,250 steps,
batch 64, val every 250, best-val state, deterministic CUDA), same training sets (augment_experiment.pick, real
rows first). Per run it saves val_prob_*.npy and test_prob_*.npy, fits the class weights on validation
(threshold_tuning.fit_class_weights) and records untuned and tuned metrics. Test numbers are saved, not printed.

Reproduction check: each retrained test_prob is compared with the committed study-2 file of the same run.

Output: generator_comparison/experiment/threshold_control/pool<k>/{config.json, results.json, *_prob_*.npy}
Usage:  py -3.14 -u _run_threshold_control.py [--pools 1,2,3,4] [--conditions real,random]
        bash run_retry.sh _run_threshold_control.py generator_comparison/threshold_control_run.log
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _study2_analysis import SEEDS
from generator_comparison import augment_experiment as E
from generator_comparison.threshold_tuning import apply_weights, fit_class_weights, half_split_optimism

ROOT = Path(__file__).resolve().parent
POOL_DIR = ROOT / "generator_comparison" / "pools2"
STUDY2 = ROOT / "generator_comparison" / "experiment" / "study2"
TAG = "threshold_control"
STEPS, EVAL_EVERY = 6250, 250
RECIPE_KEYS = ("steps", "eval_every", "deterministic", "pool_sha256")


def reproduction(out_prob: np.ndarray, ref_path: Path, y: np.ndarray) -> dict:
    if not ref_path.exists():
        return {"reference": None}
    ref = np.load(ref_path)
    same_pred = float((ref.argmax(1) == out_prob.argmax(1)).mean())
    return {"reference": str(ref_path.relative_to(ROOT)), "max_abs_diff": float(np.abs(ref - out_prob).max()),
            "same_pred_frac": same_pred, "identical": bool(np.array_equal(ref, out_prob)),
            "macro_f1_equal": bool(E.metrics(y, ref)["macro_f1"] == E.metrics(y, out_prob)["macro_f1"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="1,2,3,4")
    ap.add_argument("--conditions", default="real,random")
    ap.add_argument("--steps", type=int, default=STEPS, help="registered value 6250; other values only in tests")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--no-deterministic", action="store_true", help="tests only (CPU)")
    ap.add_argument("--seeds-per-pool", type=int, default=None, help="tests only")
    a = ap.parse_args(argv)
    det = not a.no_deterministic
    if det:
        E.set_deterministic()
        if not torch.cuda.is_available():
            raise SystemExit("deterministic run needs CUDA, and torch.cuda.is_available() is False. Not falling back to CPU.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = E.DATA
    Xtr, ytr = np.load(data / "X_train.npy").astype(np.float32), np.load(data / "y_train.npy").astype(np.int64)
    Xva, yva = np.load(data / "X_val.npy").astype(np.float32), np.load(data / "y_val.npy").astype(np.int64)
    Xte, yte = np.load(data / "X_test.npy").astype(np.float32), np.load(data / "y_test.npy").astype(np.int64)
    pid_va = pd.read_csv(data / "val_metadata.csv")["patient_id"].to_numpy()
    counts = E.synthetic_counts(ytr, 1.0)
    conds = a.conditions.split(",")

    for k in map(int, a.pools.split(",")):
        out = E.OUT / TAG / f"pool{k}"
        out.mkdir(parents=True, exist_ok=True)
        pool_path = POOL_DIR / f"pool_{k}.npz"
        pX = py = None
        if "random" in conds:
            if not pool_path.exists():
                raise SystemExit(f"{pool_path} missing: rebuild with _build_pools2.sh and check its SHA-256 "
                                 "against report/PREREGISTRATION_2.md")
            pool = np.load(pool_path)
            pX, py = pool["X"].astype(np.float32), pool["y"].astype(np.int64)
        cfg = {"pool": str(pool_path.relative_to(ROOT)) if pool_path.is_relative_to(ROOT) else str(pool_path),
               "pool_sha256": E.sha256(pool_path) if pool_path.exists() else None,
               "steps": a.steps, "eval_every": a.eval_every, "deterministic": det, "device": str(dev),
               "synthetic_per_class": {E.CLASS_ORDER[c]: n for c, n in counts.items()}}
        results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
        if results and (out / "config.json").exists():
            old = json.loads((out / "config.json").read_text())
            changed = [key for key in RECIPE_KEYS if old.get(key) is not None and cfg.get(key) is not None
                       and old[key] != cfg[key]]
            if changed:
                raise SystemExit(f"{out} has finished runs made with a different {changed}; resuming would mix them.")
        (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        seeds = SEEDS[k][: a.seeds_per_pool] if a.seeds_per_pool else SEEDS[k]
        for cond in conds:
            for seed in seeds:
                key = f"{cond}/seed{seed}"
                if key in results:
                    print("skip (done)", key)
                    continue
                sel = E.pick(cond, py, None, counts, seed) if cond == "random" else np.array([], dtype=np.int64)
                Xa = np.concatenate([Xtr, pX[sel]]) if len(sel) else Xtr
                ya = np.concatenate([ytr, py[sel]]) if len(sel) else ytr
                t0 = time.time()
                net, info = E.train_steps(Xa, ya, Xva, yva, seed, dev, a.steps, a.eval_every)
                pva, pte = E.predict(net, Xva, dev), E.predict(net, Xte, dev)
                np.save(out / f"val_prob_{cond}_seed{seed}.npy", pva)
                np.save(out / f"test_prob_{cond}_seed{seed}.npy", pte)
                w = fit_class_weights(pva, yva)
                results[key] = {
                    **info, "n_synthetic": int(len(sel)),
                    "class_weights_tuned": dict(zip(E.CLASS_ORDER, map(float, w))),
                    "val": E.metrics(yva, pva), "val_tuned": E.metrics(yva, apply_weights(pva, w)),
                    "val_optimism": half_split_optimism(pva, yva, pid_va, seed=0),
                    "test": E.metrics(yte, pte), "test_tuned": E.metrics(yte, apply_weights(pte, w)),
                    "reproduction": reproduction(pte, STUDY2 / f"pool{k}" / f"test_prob_{cond}_seed{seed}.npy", yte),
                    "sec": round(time.time() - t0, 1)}
                (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                r = results[key]
                print(f"pool{k} {key}: val mF1 {r['val']['macro_f1']:.3f} -> tuned {r['val_tuned']['macro_f1']:.3f} "
                      f"| w {[round(x, 2) for x in w]} | repro identical={r['reproduction'].get('identical')} "
                      f"| {r['sec']}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
