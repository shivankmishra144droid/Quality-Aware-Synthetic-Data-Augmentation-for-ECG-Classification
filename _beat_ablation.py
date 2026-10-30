"""Find what makes template assembly detectable (realism detector; pass bar AUC <= 0.70).
Semi-synthetic variants of REAL train ECGs (<= half of each class so disjoint real ECGs remain;
all source/donor recordings go into src_idx and are excluded from the detector's real side).

Round 1 (results in scorecard/detector_ablation_*.json):
  A real, derived leads recomputed                          AUC 0.49
  C own template + Gaussian-process residual + own wander   AUC 1.00
  D own template + own residual + Gaussian-process wander   AUC 0.95
  E own template + own residual + own wander                AUC 0.49
  G own template + another recording's residual + own wander   AUC 1.00
  H as G, wander also from another recording                   AUC 1.00
Round 2 (this script):
  E        reference
  T_rand   ANOTHER same-class recording's template @ own R-peaks + own residual + own wander
  T_match  as T_rand, donor = same-class recording with the closest mean RR
  K        own template + own residual circularly shifted (>= 0.5 s) + own wander
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from synthetic_generation.beats import highpass, place_beats, recording_template
from synthetic_generation.config import DATA_DIR
from synthetic_generation.data import load_train
from synthetic_generation.leads import reconstruct_12, to_independent_8

OUT = Path("generator_comparison/beat_templates/ablation")
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ("E", "T_rand", "T_match", "K")
X, y = load_train(DATA_DIR)
rng = np.random.default_rng(7)
variants = {k: ([], []) for k in KEYS}
src = []
for c in range(4):
    idx = rng.choice(np.where(y == c)[0], min(500, int((y == c).sum()) // 2), replace=False)
    cache = []
    for i in idx:
        x8 = to_independent_8(X[i:i + 1])[0].astype(np.float64)
        tpl, r, info = recording_template(x8)
        if tpl is None:
            continue
        hp = highpass(x8)
        cache.append({"i": i, "x8": x8, "tpl": tpl, "r": r, "hp": hp, "rr": info["rr_mean"]})
    rr = np.array([d["rr"] for d in cache])
    for j, d in enumerate(cache):
        src.append(d["i"])
        resid, wander = d["hp"] - place_beats(d["tpl"], d["r"]), d["x8"] - d["hp"]
        other = cache[(j + 1 + rng.integers(len(cache) - 1)) % len(cache)]
        gap = np.abs(rr - d["rr"])
        gap[j] = np.inf
        match = cache[int(np.argmin(gap))]
        shift = int(rng.integers(50, 950))
        parts = {
            "E": place_beats(d["tpl"], d["r"]) + resid + wander,
            "T_rand": place_beats(other["tpl"], d["r"]) + resid + wander,
            "T_match": place_beats(match["tpl"], d["r"]) + resid + wander,
            "K": place_beats(d["tpl"], d["r"]) + np.roll(resid, shift, axis=-1) + wander,
        }
        for k, v in parts.items():
            variants[k][0].append(reconstruct_12(v[None].astype(np.float32))[0])
            variants[k][1].append(c)
for k, (xs, ys) in variants.items():
    np.savez_compressed(OUT / f"{k}.npz", X=np.stack(xs), y=np.asarray(ys, np.int64), src_idx=np.asarray(src))
    print("saved", k, len(xs), flush=True)
for k in KEYS:
    r = subprocess.run([sys.executable, "-m", "generator_comparison.realism_detector",
                        str(OUT / f"{k}.npz"), f"ablation_{k}"], capture_output=True, text=True)
    res = json.loads(r.stdout)
    print(k, "AUC mean %.3f" % res["auc_mean"], {c: round(v, 3) for c, v in res["seed0"]["per_class"].items()},
          flush=True)
