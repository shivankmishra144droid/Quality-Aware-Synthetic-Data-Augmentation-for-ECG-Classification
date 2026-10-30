"""Check A: sample from full_2000.pt. Check B only if A does not close the gap."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER, pairwise_corr, summarize_generator
from synthetic_generation.config import DATA_DIR, WAVENET_CONFIG
from synthetic_generation.data import load_train
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop
from synthetic_generation.utils import count_params

ROOT = Path("generator_comparison")
CLASS_ORDER = CLASS_ORDER
X_train, y_train = load_train(DATA_DIR)
v1 = json.loads((ROOT / "comparison_results" / "sssd_metrics.json").read_text(encoding="utf-8"))
v2 = json.loads((ROOT / "sssd_v2" / "metrics.json").read_text(encoding="utf-8"))
device = torch.device("cuda")


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
            "n_real_compared": int(take),
        }
    return out


def pack(name, X, y, extra):
    m = summarize_generator(name, X, y, X_train, y_train)
    m["per_class_std"] = {CLASS_ORDER[i]: float(X[y == i].std()) for i in range(4)}
    m["per_class_diversity"] = {CLASS_ORDER[i]: pairwise_corr(X[y == i]) for i in range(4)}
    m["near_dup_vs_real"] = max_corr_to_real(X, y, X_train, y_train)
    m.update(extra)
    return m


def generate_128(net, out: Path, tag: str, seed=42):
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    xs, ys = [], []
    for cid, name in enumerate(CLASS_ORDER):
        x = generate_class_batch(net, cid, 32, device, batch_size=1, seed=seed)
        print("gen", tag, name, float(x.std()))
        xs.append(x)
        ys.append(np.full(32, cid, np.int64))
    X = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    meta = metadata_rows(y)
    meta["generator"] = tag
    meta.to_csv(out / "metadata.csv", index=False)
    extra = {"gen_sec": time.time() - t0, "gen_peak_gb": torch.cuda.max_memory_allocated() / 1024**3}
    return X, y, extra


def hr_separated(m) -> bool:
    hrs = [m["hr_by_class"][c]["syn_median_hr"] for c in CLASS_ORDER]
    # STACH should be faster than SBRAD by a clear margin
    return (m["hr_by_class"]["STACH"]["syn_median_hr"] - m["hr_by_class"]["SBRAD"]["syn_median_hr"]) >= 20


def gap_closed(m) -> bool:
    std_ok = m["waveform"]["std"] >= 0.85
    return std_ok and hr_separated(m)


# ----- Check A -----
check_a = {
    "ckpt_every_week4": 1000,
    "loss_min_iter": 2600,
    "loss_min": 0.04485546797513962,
    "checkpoint_at_2600": False,
    "nearest_existing": "synthetic_generation/checkpoints/full_2000.pt",
    "nearest_iter": 2000,
    "nearest_logged_loss": 0.05352030694484711,
    "also_exists": ["full_1000.pt", "full_2000.pt", "full_3000.pt", "full_4000.pt"],
    "note": "No ckpt at 2600. Evaluating iter 2000 (closest saved point before post-min drift).",
}
print("CHECK A", json.dumps(check_a, indent=2))

ckpt = torch.load("synthetic_generation/checkpoints/full_2000.pt", map_location=device)
net = SSSD_ECG(**ckpt["wavenet_config"]).to(device)
net.load_state_dict(ckpt["model_state_dict"])
net.eval()
out_a = ROOT / "sssd_ckpt2000"
X_a, y_a, extra_a = generate_128(net, out_a, "SSSD-ECG-adapted-ckpt2000")
m_a = pack("SSSD-ECG-adapted-ckpt2000", X_a, y_a, {
    **extra_a,
    "train_iters": 2000,
    "n_params": count_params(net),
    "check": "A",
    "checkpoint": "full_2000.pt",
})
(out_a / "metrics.json").write_text(json.dumps(m_a, indent=2), encoding="utf-8")
(out_a / "check_a.json").write_text(json.dumps(check_a, indent=2), encoding="utf-8")
print("A std", m_a["waveform"]["std"], "hr", {k: v["syn_median_hr"] for k, v in m_a["hr_by_class"].items()})
print("A gap_closed", gap_closed(m_a))

summary = {"check_a": check_a, "ckpt2000": m_a, "v1_4000": {
    "std": v1["waveform"]["std"], "hr": v1["hr_by_class"], "validity": v1["validity"],
}, "v2_16000": {
    "std": v2["waveform"]["std"], "hr": v2["hr_by_class"], "validity": v2["validity"],
    "near_dup_vs_real": v2.get("near_dup_vs_real"),
}}

if gap_closed(m_a):
    summary["stopped_after"] = "A"
    summary["check_b_ran"] = False
    (ROOT / "sssd_ckpt2000" / "three_way.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Check A closed the gap. Skip B.")
    raise SystemExit(0)

print("Check A did not close amplitude/HR gap. Starting Check B (16x128, 4000 iters from scratch).")

WAVENET_16x128 = {
    **WAVENET_CONFIG,
    "num_res_layers": 16,
    "res_channels": 128,
    "skip_channels": 128,
}
out_b = ROOT / "sssd_v3"
out_b.mkdir(parents=True, exist_ok=True)
train_cfg = {
    "n_iters": 4000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 50,
    "ckpt_every": 500,
    "use_amp": False,
}
net, hist, peak, last = train_loop(
    X_train, y_train, train_cfg, tag="v3_16x128",
    wavenet_config=WAVENET_16x128,
    ckpt_dir=out_b / "checkpoints",
    history_path=out_b / "history.json",
)
losses = [(r["iter"], r["loss"]) for r in hist]
imin = int(np.argmin([r["loss"] for r in hist]))
print("B loss first", hist[0]["loss"], "last", hist[-1]["loss"], "min", hist[imin]["iter"], hist[imin]["loss"])

X_b, y_b, extra_b = generate_128(net, out_b, "SSSD-ECG-adapted-v3-16x128")
m_b = pack("SSSD-ECG-adapted-v3-16x128", X_b, y_b, {
    **extra_b,
    "train_iters": 4000,
    "n_params": count_params(net),
    "peak_gb_train": peak,
    "check": "B",
    "wavenet": {"num_res_layers": 16, "res_channels": 128},
    "loss_min_iter": hist[imin]["iter"],
    "loss_min": hist[imin]["loss"],
    "loss_last": hist[-1]["loss"],
    "dip_then_drift": hist[imin]["iter"] < 3500 and hist[-1]["loss"] > hist[imin]["loss"] * 1.5,
})
(out_b / "metrics.json").write_text(json.dumps(m_b, indent=2), encoding="utf-8")
(out_b / "config.json").write_text(json.dumps({
    "change": "16x128 from scratch, 4000 iters (v1 budget; v2 16000 was degrading)",
    "wavenet": WAVENET_16x128,
    "train_cfg": train_cfg,
    "checkpoint": str(last),
}, indent=2), encoding="utf-8")
summary["check_b_ran"] = True
summary["stopped_after"] = "B"
summary["v3_16x128"] = m_b
(out_b / "three_way.json").write_text(json.dumps({
    "v1_4000_std": v1["waveform"]["std"],
    "v2_16000_std": v2["waveform"]["std"],
    "ckpt2000_std": m_a["waveform"]["std"],
    "v3_std": m_b["waveform"]["std"],
    "v1_hr": {k: v1["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
    "v2_hr": {k: v2["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
    "ckpt2000_hr": {k: m_a["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
    "v3_hr": {k: m_b["hr_by_class"][k]["syn_median_hr"] for k in CLASS_ORDER},
    "v3_dip_then_drift": m_b["dip_then_drift"],
    "v3_loss_min_iter": m_b["loss_min_iter"],
}, indent=2), encoding="utf-8")
print("B std", m_b["waveform"]["std"], "hr", {k: v["syn_median_hr"] for k, v in m_b["hr_by_class"].items()})
print("B dip_then_drift", m_b["dip_then_drift"], "memorization", m_b["near_dup_vs_real"])
print("done")
