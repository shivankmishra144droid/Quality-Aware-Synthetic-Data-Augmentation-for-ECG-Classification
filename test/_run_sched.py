"""Noise-schedule isolated run vs v1: 8x64, 4000 iters from scratch, no CFG, same seed as v1.
v1 schedule (T=200, beta 1e-4 -> 0.02) ends at alpha_bar_T = 0.13 (36% signal left), but sampling
starts from pure N(0, 1). Only change: beta 5e-4 -> 0.1 (DDPM-1000 equivalent, alpha_bar_T ~ 3e-5).
Raw weights (sssd_sched_raw/) are the single-factor result; EMA (sssd_sched/) is sampled too."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt, pairwise_corr, summarize_generator
from synthetic_generation.config import DATA_DIR, WAVENET_4GB
from synthetic_generation.data import load_train
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop

ROOT = Path("generator_comparison")
X_train, y_train = load_train(DATA_DIR)
device = torch.device("cuda")

EMA_DECAY = 0.999
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}
why = {
    "architecture": "WAVENET_4GB 8x64 (same as v1)",
    "single_change": "noise schedule beta 1e-4->0.02 replaced by 5e-4->0.1 (terminal alpha_bar 0.13 -> 3e-5)",
    "diffusion_config": DIFF,
    "ema_decay_tracked": EMA_DECAY,
    "iters": 4000,
    "sample_batch_size": 8,
    "sample_seed": 42,
}
print("SCHED PLAN", json.dumps(why, indent=2))
train_dir = ROOT / "sssd_sched"
train_dir.mkdir(parents=True, exist_ok=True)
(train_dir / "plan.json").write_text(json.dumps(why, indent=2), encoding="utf-8")

train_cfg = {
    "n_iters": 4000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 50,
    "ckpt_every": 1000,
    "use_amp": False,
    "ema_decay": EMA_DECAY,
    "diffusion_config": DIFF,
}
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="sched_8x64",
    wavenet_config=WAVENET_4GB,
    ckpt_dir=train_dir / "checkpoints",
    history_path=train_dir / "history.json",
)
ck = torch.load(last, map_location=device)
ema_net = SSSD_ECG(**ck["wavenet_config"]).to(device)
ema_net.load_state_dict(ck["ema_state_dict"])

summary = {}
for label, model, out in (("raw", net, ROOT / "sssd_sched_raw"), ("ema", ema_net, train_dir)):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xs, ys = [], []
    for cid, name in enumerate(CLASS_ORDER):
        x = generate_class_batch(model, cid, 32, device, batch_size=8, seed=42, diffusion_config=DIFF)
        print(f"{label} {name} std={float(x.std()):.4f}")
        xs.append(x)
        ys.append(np.full(32, cid, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    metadata_rows(y).to_csv(out / "metadata.csv", index=False)
    m = summarize_generator(f"SSSD-sched-{label}", X, y, X_train, y_train)
    m["per_class_std"] = {CLASS_ORDER[i]: float(X[y == i].std()) for i in range(4)}
    m["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
    m["hr_pt_by_class"] = {CLASS_ORDER[i]: lead_ii_hr_pt(X[y == i]) for i in range(4)}
    m.update({"weights": label, "gen_sec": time.time() - t0, "train_iters": 4000, "peak_gb_train": peak})
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    summary[label] = {
        "std": m["waveform"]["std"],
        "hr": {k: round(v["median_hr_peaks"], 1) for k, v in m["hr_pt_by_class"].items()},
        "periodicity": {k: round(v["median_periodicity"], 2) for k, v in m["hr_pt_by_class"].items()},
        "agree": {k: round(v["agree_frac"], 2) for k, v in m["hr_pt_by_class"].items()},
    }
    print(label, json.dumps(summary[label]))

(train_dir / "config.json").write_text(json.dumps({**why, "train_cfg": train_cfg, "checkpoint": str(last),
                                                   "loss_last": hist[-1]["loss"], "raw_vs_ema": summary},
                                                  indent=2), encoding="utf-8")
print("done")
