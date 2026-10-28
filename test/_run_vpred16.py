"""Capacity check: v-prediction at 16x128 (7.2M params) vs the 8x64 (1.53M) sssd_vpred run.
Only change vs sssd_vpred: model width/depth. Same effective batch (32, here 4 x accum 8 because
16x128 at batch 8 exceeds 4 GB), lr schedule, fixed noise schedule, v-pred, EMA warmup, seed.
16x128 runs ~9 ECG/s vs ~54 for 8x64, so the full 10k steps (~10 h) is not planned: compare the
evals against the 8x64 curve at equal steps (8x64 raw periodicity 0.17/0.18/0.22/0.25/0.25 at
1k-5k) and stop early via the STOP file once the answer is clear. Checkpoints every 500 steps."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt, pairwise_corr, summarize_generator
from synthetic_generation.config import DATA_DIR, WAVENET_4GB

WAVENET_16x128 = {**WAVENET_4GB, "num_res_layers": 16, "res_channels": 128, "skip_channels": 128}
from synthetic_generation.data import load_train
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop

ROOT = Path("generator_comparison")
OUT = ROOT / "sssd_vpred16"
PARAM = "v"
X_train, y_train = load_train(DATA_DIR)
device = torch.device("cuda")
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}

train_cfg = {
    "n_iters": 10000,
    "batch_size": 4,  # 16x128: batch 4 = 3.8 GB device, batch 8 overflows 4 GB
    "grad_accum": 8,
    "learning_rate": 2e-4,
    "warmup_iters": 500,
    "cosine_decay": True,
    "log_every": 100,
    "ckpt_every": 500,
    "eval_every": 500,
    "use_amp": False,
    "ema_decay": 0.999,
    "diffusion_config": DIFF,
    "parameterization": PARAM,
    "ema_warmup": True,
}
plan = {"train_cfg": train_cfg, "ecgs_seen": 10000 * 32, "v1_ecgs_seen": 4000,
        "compare_to": "sssd_vpred (8x64, same cfg except batch 8 x accum 4)",
        "new": "16x128 capacity", "wavenet": WAVENET_16x128}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print("VPRED16 PLAN", json.dumps(plan, indent=2))


def quick_stats(model, n=8):
    xs, ys = [], []
    for cid in range(4):
        xs.append(generate_class_batch(model, cid, n, device, batch_size=4, seed=7, diffusion_config=DIFF,
                                        parameterization=PARAM))
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
ckpts = sorted((OUT / "checkpoints").glob("vpred_16x128_[0-9]*.pt"), key=lambda q: int(q.stem.rsplit("_", 1)[1]))
resume = ckpts[-1] if ckpts else None
print("resume from", resume)
t_start = time.time()
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="vpred_16x128",
    wavenet_config=WAVENET_16x128,
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
param = ck["train_cfg"].get("parameterization", "eps")  # never sample a v-model with the eps conversion
for label, model, out in (("raw", net, ROOT / "sssd_vpred16_raw"), ("ema", ema_net, OUT)):
    out.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    for cid, name in enumerate(CLASS_ORDER):
        xs.append(generate_class_batch(model, cid, 32, device, batch_size=4, seed=42, diffusion_config=DIFF,
                                         parameterization=param))
        ys.append(np.full(32, cid, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    metadata_rows(y).to_csv(out / "metadata.csv", index=False)
    m = summarize_generator(f"SSSD-vpred16-{label}", X, y, X_train, y_train)
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
