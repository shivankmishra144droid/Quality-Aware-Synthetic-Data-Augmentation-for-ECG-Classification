"""Balanced-sampling isolated run vs v1: 8x64, 4000 iters from scratch, no CFG.
Only change vs v1: WeightedRandomSampler so each class is drawn ~equally (v1 saw ~85% SR)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt, pairwise_corr, summarize_generator
from synthetic_generation.config import DATA_DIR, WAVENET_4GB
from synthetic_generation.data import load_train
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop
from synthetic_generation.utils import count_params

ROOT = Path("generator_comparison")
OUT = ROOT / "sssd_balanced"
X_train, y_train = load_train(DATA_DIR)
device = torch.device("cuda")

counts = np.bincount(y_train, minlength=4)
why = {
    "architecture": "WAVENET_4GB 8x64 (same as v1)",
    "single_change": "class-balanced WeightedRandomSampler (weight 1/class_count, with replacement)",
    "iters": 4000,
    "label_dropout_p": 0.0,
    "guidance_w": None,
    "fresh_run": True,
    "expected_draws_v1": {CLASS_ORDER[i]: round(4000 * counts[i] / counts.sum()) for i in range(4)},
    "expected_draws_balanced": {c: 1000 for c in CLASS_ORDER},
}
print("BALANCED PLAN", json.dumps(why, indent=2))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plan.json").write_text(json.dumps(why, indent=2), encoding="utf-8")

train_cfg = {
    "n_iters": 4000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 50,
    "ckpt_every": 1000,
    "use_amp": False,
    "balanced_sampling": True,
}
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="balanced_8x64",
    wavenet_config=WAVENET_4GB,
    ckpt_dir=OUT / "checkpoints",
    history_path=OUT / "history.json",
)
print("train last", hist[-1]["loss"], "min", min(r["loss"] for r in hist))

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
xs, ys = [], []
for cid, name in enumerate(CLASS_ORDER):
    x = generate_class_batch(net, cid, 32, device, batch_size=1, seed=42)
    print(f"{name} std={float(x.std()):.4f}")
    xs.append(x)
    ys.append(np.full(32, cid, np.int64))
X = np.concatenate(xs).astype(np.float32)
y = np.concatenate(ys)
np.save(OUT / "X_synthetic.npy", X)
np.save(OUT / "y_synthetic.npy", y)
metadata_rows(y).to_csv(OUT / "metadata.csv", index=False)

m = summarize_generator("SSSD-balanced", X, y, X_train, y_train)
m["per_class_std"] = {CLASS_ORDER[i]: float(X[y == i].std()) for i in range(4)}
m["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
m["hr_pt_by_class"] = {CLASS_ORDER[i]: lead_ii_hr_pt(X[y == i]) for i in range(4)}
m.update({"gen_sec": time.time() - t0, "gen_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
          "train_iters": 4000, "n_params": count_params(net), "peak_gb_train": peak})
(OUT / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
(OUT / "config.json").write_text(json.dumps({**why, "train_cfg": train_cfg, "checkpoint": str(last),
                                             "loss_last": hist[-1]["loss"]}, indent=2), encoding="utf-8")
print("std", m["waveform"]["std"],
      "HR_pt", {k: round(v["median_hr_peaks"], 1) for k, v in m["hr_pt_by_class"].items()},
      "periodicity", {k: round(v["median_periodicity"], 2) for k, v in m["hr_pt_by_class"].items()})
print("done")
