"""Phase 2 sanity check: does the quality score pick more useful synthetic ECGs?
TSTR (train on synthetic only, test on real val) for three same-size subsets of the final pool:
top-k by quality score, random k, bottom-k. A useful filter gives top > random > bottom.
The final pool has minority classes only, so the SR training examples come from a separate
generated SR set (pool_beatgen.npz) and are identical in all three subsets.
Writes <out>.json / .md (default generator_comparison/tstr/filter_check).

Usage: py -3.14 _filter_check.py <final_pool.npz> <scores.npz> [k_per_class=440] [out]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from generator_comparison.eval_metrics import CLASS_ORDER
from generator_comparison.tstr import DATA, train_eval

pool, scores_path = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 440
OUT = Path("generator_comparison/tstr")
RESULT = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT / "filter_check"
RESULT.parent.mkdir(parents=True, exist_ok=True)
d, s = np.load(pool), np.load(scores_path)
X, y, sc = d["X"].astype(np.float32), d["y"], s["score"]
assert len(sc) == len(y)
sr = np.load(OUT / "pool_beatgen.npz")
X_sr = sr["X"][sr["y"] == 0][:K].astype(np.float32)
Xva, yva = np.load(DATA / "X_val.npy").astype(np.float32), np.load(DATA / "y_val.npy")

results = {}
for name in ("top", "random", "bottom"):
    runs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        idx = []
        for c in (1, 2, 3):
            ic = np.where(y == c)[0]
            order = ic[np.argsort(-sc[ic])]
            idx.append({"top": order[:K], "bottom": order[-K:], "random": rng.choice(ic, K, replace=False)}[name])
        idx = np.concatenate(idx)
        Xtr = np.concatenate([X_sr, X[idx]])
        ytr = np.concatenate([np.zeros(len(X_sr), np.int64), y[idx]])
        runs.append(train_eval(Xtr, ytr, Xva, yva, seed))
        print(name, "seed", seed, "bal acc %.3f" % runs[-1]["balanced_acc"], flush=True)
    results[name] = {"balanced_acc_mean": float(np.mean([r["balanced_acc"] for r in runs])),
                     "balanced_acc_std": float(np.std([r["balanced_acc"] for r in runs])),
                     "recall_mean": {c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASS_ORDER},
                     "mean_score": float(np.mean([sc[np.where(y == c)[0]].mean() for c in (1, 2, 3)]))}
RESULT.with_suffix(".json").write_text(json.dumps({"k_per_class": K, "pool": pool, "results": results}, indent=2),
                                       encoding="utf-8")
lines = ["| subset | TSTR balanced acc (real val) | recall SR / AFIB / STACH / SBRAD |", "|---|---|---|"]
for n, r in results.items():
    rc = r["recall_mean"]
    lines.append(f"| {n}-{K} | {r['balanced_acc_mean']:.3f} ± {r['balanced_acc_std']:.3f} | "
                 + " / ".join(f"{rc[c]:.2f}" for c in CLASS_ORDER) + " |")
RESULT.with_suffix(".md").write_text("# Quality-filter sanity check\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
