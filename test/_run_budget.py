"""Training-budget run: 8x64, fixed schedule (beta 5e-4 -> 0.1), EMA 0.999.
Every earlier run saw ~4000 ECGs at batch 1. This one: batch 8 x accum 4 (effective 32),
10000 optimizer steps (~320k ECGs, ~80x v1), lr 2e-4 with 500-step warmup + cosine decay.
Periodic evals (8/class) log periodicity so a flat run is visible early.
Not a single-factor run: schedule + EMA are kept from sssd_sched / sssd_ema, budget is the new factor."""
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
OUT = ROOT / "sssd_budget"
X_train, y_train = load_train(DATA_DIR)
device = torch.device("cuda")
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}

train_cfg = {
    "n_iters": 10000,
    "batch_size": 8,  # 16 hit a CUDA OOM at ~1.56 GB peak (shared 4 GB laptop GPU)
    "grad_accum": 4,
    "learning_rate": 2e-4,
    "warmup_iters": 500,
    "cosine_decay": True,
    "log_every": 100,
    "ckpt_every": 500,
    "eval_every": 1000,
    "use_amp": False,
    "ema_decay": 0.999,
    "diffusion_config": DIFF,
}
plan = {"train_cfg": train_cfg, "ecgs_seen": 10000 * 32, "v1_ecgs_seen": 4000,
        "carried_over": ["noise schedule 5e-4->0.1 (sssd_sched)", "EMA 0.999 (sssd_ema)"]}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print("BUDGET PLAN", json.dumps(plan, indent=2))


def quick_stats(model, n=8):
    xs, ys = [], []
    for cid in range(4):
        xs.append(generate_class_batch(model, cid, n, device, batch_size=n, seed=7, diffusion_config=DIFF))
        ys.append(np.full(n, cid))
    X, y = np.concatenate(xs), np.concatenate(ys)
    hr = {CLASS_ORDER[i]: lead_ii_hr_pt(X[y == i]) for i in range(4)}
    return {"std": round(float(X.std()), 3),
            "periodicity": round(float(np.median([hr[k]["median_periodicity"] for k in ("SR", "STACH", "SBRAD")])), 3),
            "hr": {k: round(v["median_hr_peaks"], 1) for k, v in hr.items()}}


def eval_fn(net, ema, n_iter):
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()  # sampling reseeds; keep train stream intact
    out = {"raw": quick_stats(net), "ema": quick_stats(ema)}
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    return out


# Resume from the newest checkpoint if a previous attempt died (transient CUDA OOM from other
# GPU apps on this 4 GB laptop; the same config ran 400 steps flat at 1.97 GB in isolation).
ckpts = sorted((OUT / "checkpoints").glob("budget_8x64_[0-9]*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
resume = ckpts[-1] if ckpts else None
print("resume from", resume)
t_start = time.time()
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="budget_8x64",
    wavenet_config=WAVENET_4GB,
    ckpt_dir=OUT / "checkpoints",
    history_path=OUT / "history.json",
    eval_fn=eval_fn,
    resume_path=resume,
)
train_sec = time.time() - t_start
ck = torch.load(last, map_location=device)
ema_net = SSSD_ECG(**ck["wavenet_config"]).to(device)
ema_net.load_state_dict(ck["ema_state_dict"])

summary = {}
for label, model, out in (("raw", net, ROOT / "sssd_budget_raw"), ("ema", ema_net, OUT)):
    out.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    for cid, name in enumerate(CLASS_ORDER):
        xs.append(generate_class_batch(model, cid, 32, device, batch_size=8, seed=42, diffusion_config=DIFF))
        ys.append(np.full(32, cid, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    metadata_rows(y).to_csv(out / "metadata.csv", index=False)
    m = summarize_generator(f"SSSD-budget-{label}", X, y, X_train, y_train)
    m["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
    m["hr_pt_by_class"] = {CLASS_ORDER[i]: lead_ii_hr_pt(X[y == i]) for i in range(4)}
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    summary[label] = {"std": m["waveform"]["std"],
                      "periodicity": {k: round(v["median_periodicity"], 2) for k, v in m["hr_pt_by_class"].items()},
                      "hr": {k: round(v["median_hr_peaks"], 1) for k, v in m["hr_pt_by_class"].items()}}
    print(label, json.dumps(summary[label]))

(OUT / "config.json").write_text(json.dumps({**plan, "checkpoint": str(last), "train_sec": train_sec,
                                             "peak_gb_train": peak, "loss_last": hist[-1]["loss"],
                                             "raw_vs_ema": summary}, indent=2), encoding="utf-8")
print("done")
