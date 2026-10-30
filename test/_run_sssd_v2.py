"""sssd_v2: more iters only (4000 -> 16000), same 8x64. Does not overwrite sssd/."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import (
    CLASS_ORDER,
    pairwise_corr,
    summarize_generator,
    waveform_stats,
)
from synthetic_generation.config import DATA_DIR, WAVENET_4GB
from synthetic_generation.data import load_train
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop
from synthetic_generation.utils import count_params

OUT = Path("generator_comparison/sssd_v2")
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

X_train, y_train = load_train(DATA_DIR)
cfg = {
    "n_iters": 16000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 100,
    "ckpt_every": 4000,
    "use_amp": False,
}
resume = Path("synthetic_generation/checkpoints/full_last.pt")
net, hist, peak, last = train_loop(
    X_train,
    y_train,
    cfg,
    tag="v2_16000",
    wavenet_config=WAVENET_4GB,
    resume_path=resume,
    ckpt_dir=CKPT_DIR,
    history_path=OUT / "history.json",
)

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
xs, ys = [], []
for cid, name in enumerate(CLASS_ORDER):
    x = generate_class_batch(net, cid, 32, device, batch_size=1, seed=42)
    print("gen", name, x.shape, float(x.std()))
    xs.append(x)
    ys.append(np.full(32, cid, np.int64))
gen_sec = time.time() - t0
gen_gb = torch.cuda.max_memory_allocated() / 1024**3
X = np.concatenate(xs).astype(np.float32)
y = np.concatenate(ys)
np.save(OUT / "X_synthetic.npy", X)
np.save(OUT / "y_synthetic.npy", y)
meta = metadata_rows(y)
meta["generator"] = "SSSD-ECG-adapted-v2"
meta["seed"] = 42
meta.to_csv(OUT / "metadata.csv", index=False)

base = json.loads(Path("generator_comparison/comparison_results/sssd_metrics.json").read_text(encoding="utf-8"))
v2 = summarize_generator("SSSD-ECG-adapted-v2", X, y, X_train, y_train)
v2["per_class_std"] = {CLASS_ORDER[i]: float(X[y == i].std()) for i in range(4)}
v2["per_class_std_real"] = {CLASS_ORDER[i]: float(X_train[y_train == i].std()) for i in range(4)}
v2["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
v2["train_iters"] = 16000
v2["n_params"] = count_params(net)
v2["peak_gb_train"] = peak
v2["gen_sec"] = gen_sec
v2["gen_peak_gb"] = gen_gb
v2["single_change"] = "resume 4000 -> 16000 iters; architecture 8x64 unchanged; T=200 unchanged"

# Near-duplicate vs REAL train (same class), not Week-5 quality score.
def max_corr_to_real(x_syn, y_syn, x_real, y_real, n_real=256, thresh=0.95):
    out = {}
    for i, name in enumerate(CLASS_ORDER):
        syn = x_syn[y_syn == i].reshape(len(x_syn[y_syn == i]), -1)
        real = x_real[y_real == i]
        rng = np.random.default_rng(42 + i)
        take = min(n_real, len(real))
        idx = rng.choice(len(real), size=take, replace=False)
        real = real[idx].reshape(take, -1)
        syn = (syn - syn.mean(1, keepdims=True)) / (syn.std(1, keepdims=True) + 1e-8)
        real = (real - real.mean(1, keepdims=True)) / (real.std(1, keepdims=True) + 1e-8)
        c = (syn @ real.T) / syn.shape[1]
        mx = c.max(axis=1)
        out[name] = {
            "mean_max_corr": float(mx.mean()),
            "max_corr": float(mx.max()),
            "frac_gt_0.95": float((mx > thresh).mean()),
            "n_real_compared": int(take),
        }
    return out

v2["near_dup_vs_real"] = max_corr_to_real(X, y, X_train, y_train)
v2["baseline_std"] = base["waveform"]["std"]
v2["baseline_hr"] = base["hr_by_class"]
(OUT / "metrics.json").write_text(json.dumps(v2, indent=2), encoding="utf-8")
(OUT / "config.json").write_text(json.dumps({
    "change": v2["single_change"],
    "resume_from": str(resume),
    "wavenet": WAVENET_4GB,
    "train_cfg": cfg,
    "checkpoint": str(last),
}, indent=2), encoding="utf-8")
print("validity", v2["validity"])
print("std", v2["waveform"]["std"], "vs real", v2["real_waveform"]["std"], "vs v1", base["waveform"]["std"])
print("hr", v2["hr_by_class"])
print("div", v2["diversity"], v2["per_class_diversity"])
print("memorization", v2["near_dup_vs_real"])
print("saved", OUT)
