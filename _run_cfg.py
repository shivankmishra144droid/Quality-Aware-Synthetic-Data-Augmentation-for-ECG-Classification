"""CFG isolated run vs v1: 8x64, 4000 iters from scratch, 10% label dropout."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER, pairwise_corr, summarize_generator
from synthetic_generation.config import DATA_DIR, WAVENET_4GB
from synthetic_generation.data import load_train
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop
from synthetic_generation.utils import count_params

ROOT = Path("generator_comparison")
X_train, y_train = load_train(DATA_DIR)
device = torch.device("cuda")

why = {
    "architecture": "WAVENET_4GB 8x64 (same as v1)",
    "why_not_v3_16x128": "CFG is a single-factor test; must not combine with the capacity change. CFG sampling also doubles NFE vs v3's 3.66 GB peak.",
    "why_not_v2_16000": "v2 loss was already degrading; v1/v3 4000-iter budget had better loss-vs-noise.",
    "iters": 4000,
    "label_dropout_p": 0.10,
    "null_cond": "zero one-hot -> zero embedding via label @ embedding.weight",
    "fresh_run": True,
    "not_finetuned_from_prior_ckpt": True,
}
print("CFG PLAN", json.dumps(why, indent=2))


def max_corr_to_real(x_syn, y_syn, x_real, y_real, n_real=256, thresh=0.95):
    out = {}
    for i, name in enumerate(CLASS_ORDER):
        syn = x_syn[y_syn == i].reshape((-1, 1000 * 12))
        real = x_real[y_real == i]
        rng = np.random.default_rng(42 + i)
        take = min(n_real, len(real))
        idx = rng.choice(len(real), size=take, replace=False)
        real = real[idx].reshape(take, -1)
        syn = (syn - syn.mean(1, keepdims=True)) / (syn.std(1, keepdims=True) + 1e-8)
        real = (real - real.mean(1, keepdims=True)) / (real.std(1, keepdims=True) + 1e-8)
        mx = (syn @ real.T / syn.shape[1]).max(axis=1)
        out[name] = {
            "mean_max_corr": float(mx.mean()),
            "max_corr": float(mx.max()),
            "frac_gt_0.95": float((mx > thresh).mean()),
        }
    return out


def pack(name, X, y, extra):
    m = summarize_generator(name, X, y, X_train, y_train)
    m["per_class_std"] = {CLASS_ORDER[i]: float(X[y == i].std()) for i in range(4)}
    m["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
    m["near_dup_vs_real"] = max_corr_to_real(X, y, X_train, y_train)
    m.update(extra)
    hrs = {k: v["syn_median_hr"] for k, v in m["hr_by_class"].items()}
    m["stach_minus_sbrad_hr"] = hrs["STACH"] - hrs["SBRAD"]
    return m


out_train = ROOT / "sssd_cfg"
out_train.mkdir(parents=True, exist_ok=True)
(out_train / "plan.json").write_text(json.dumps(why, indent=2), encoding="utf-8")

train_cfg = {
    "n_iters": 4000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 50,
    "ckpt_every": 500,
    "use_amp": False,
    "label_dropout_p": 0.10,
}
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="cfg_8x64",
    wavenet_config=WAVENET_4GB,
    ckpt_dir=out_train / "checkpoints",
    history_path=out_train / "history.json",
)
imin = int(np.argmin([r["loss"] for r in hist]))
dip = hist[imin]["iter"] < 3500 and hist[-1]["loss"] > hist[imin]["loss"] * 1.5
print("CFG train last", hist[-1]["loss"], "min", hist[imin]["iter"], hist[imin]["loss"], "dip_then_drift", dip)
if dip:
    print("NOTE: dip-then-drift recurred; not extending past 4000.")

(out_train / "config.json").write_text(json.dumps({
    **why,
    "train_cfg": train_cfg,
    "checkpoint": str(last),
    "n_params": count_params(net),
    "peak_gb_train": peak,
    "loss_min_iter": hist[imin]["iter"],
    "loss_min": hist[imin]["loss"],
    "loss_last": hist[-1]["loss"],
    "dip_then_drift": dip,
}, indent=2), encoding="utf-8")

results = {}
for w, dirname in ((2.0, "sssd_cfg_w2"), (4.0, "sssd_cfg_w4")):
    out = ROOT / dirname
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    xs, ys = [], []
    for cid, name in enumerate(CLASS_ORDER):
        x = generate_class_batch(net, cid, 32, device, batch_size=1, seed=42, guidance_w=w)
        print(f"w={w} {name} std={float(x.std()):.4f}")
        xs.append(x)
        ys.append(np.full(32, cid, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    metadata_rows(y).to_csv(out / "metadata.csv", index=False)
    m = pack(f"SSSD-CFG-w{int(w)}", X, y, {
        "guidance_w": w,
        "gen_sec": time.time() - t0,
        "gen_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "train_iters": 4000,
        "n_params": count_params(net),
        "label_dropout_p": 0.10,
    })
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    results[f"w{int(w)}"] = m
    print("w", w, "std", m["waveform"]["std"], "HR", {k: v["syn_median_hr"] for k, v in m["hr_by_class"].items()},
          "STACH-SBRAD", m["stach_minus_sbrad_hr"], "div", m["diversity"], "mem", m["near_dup_vs_real"])

v1 = json.loads((ROOT / "comparison_results" / "sssd_metrics.json").read_text(encoding="utf-8"))
cmp = {
    "v1_no_cfg": {
        "std": v1["waveform"]["std"],
        "hr": {k: v1["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
    },
    "cfg_w2": {
        "std": results["w2"]["waveform"]["std"],
        "hr": {k: results["w2"]["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
        "stach_minus_sbrad_hr": results["w2"]["stach_minus_sbrad_hr"],
        "diversity_mean": results["w2"]["diversity"]["mean"],
        "per_class_diversity": results["w2"]["per_class_diversity"],
        "near_dup_vs_real_max": {k: v["max_corr"] for k, v in results["w2"]["near_dup_vs_real"].items()},
        "validity": results["w2"]["validity"],
    },
    "cfg_w4": {
        "std": results["w4"]["waveform"]["std"],
        "hr": {k: results["w4"]["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
        "stach_minus_sbrad_hr": results["w4"]["stach_minus_sbrad_hr"],
        "diversity_mean": results["w4"]["diversity"]["mean"],
        "per_class_diversity": results["w4"]["per_class_diversity"],
        "near_dup_vs_real_max": {k: v["max_corr"] for k, v in results["w4"]["near_dup_vs_real"].items()},
        "validity": results["w4"]["validity"],
    },
    "dip_then_drift": dip,
}
(out_train / "cfg_comparison.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
print("COMPARISON", json.dumps(cmp, indent=2))
print("done")
