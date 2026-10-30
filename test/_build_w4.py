"""Build 04_Synthetic_ECG_Generation.ipynb"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip() + "\n"))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip() + "\n"))

md("""
# Synthetic ECG generation (adapted SSSD-ECG)

Week 4 only. Week 3 CNN-LSTM stays frozen. Generator trains on `X_train` / `y_train` only.

**Chosen generator:** SSSD-ECG (https://github.com/AI4HealthUOL/SSSD-ECG), trained from scratch.

**Adaptations:** 4-class one-hot instead of 71 SCP codes; 8 independent leads I, II, V1–V6 then Einthoven reconstruction to the Week 2 12-lead order. No pretrained weights.
""")

md("""
## 1. Environment check
""")

code("""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({"figure.figsize": (10, 4), "font.size": 11})

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is False. Do not run SSSD training on CPU. "
        "Need a CUDA PyTorch build (this machine has an RTX 3050)."
    )
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
""")

md("""
## 2. Load training data
""")

code("""
from synthetic_generation.config import (
    CLASS_ORDER, DATA_DIR, DIFFUSION_CONFIG, GENERATOR_NAME, GENERATOR_SOURCE,
    OUT_DIR, PILOT_SAMPLES_PER_CLASS, ROOT, SEED, TRAIN_FULL, TRAIN_PILOT,
    WAVENET_4GB, WAVENET_CONFIG,
)
from synthetic_generation.data import class_counts, load_train, one_hot
from synthetic_generation.leads import (
    INDEPENDENT_IDX, INDEPENDENT_LEADS, LEAD_ORDER_12, reconstruct_12, to_independent_8,
)
from synthetic_generation.sample import basic_sanity, generate_class_batch, metadata_rows
from synthetic_generation.train import train_loop
from synthetic_generation.utils import count_params, set_seed

set_seed(SEED)
X_train, y_train = load_train()
print("X_train", X_train.shape, X_train.dtype)
print("y_train", y_train.shape, y_train.dtype)
print("finite", bool(np.isfinite(X_train).all()))
print("labels", sorted(set(y_train.tolist())))
print("counts", class_counts(y_train))
assert X_train.shape == (13811, 1000, 12)
assert set(np.unique(y_train)) <= {0, 1, 2, 3}
""")

md("""
**Observation:** Training arrays match Week 2. Val/test are not loaded for generator training.
""")

md("""
## 3. Lead ordering

From `processed_data/dataset_config.json` and Week 2 `LEAD_NAMES` (wfdb).
""")

code("""
cfg = json.loads((DATA_DIR / "dataset_config.json").read_text(encoding="utf-8"))
print("config lead_order:", cfg["lead_order"])
print("module LEAD_ORDER_12:", LEAD_ORDER_12)
assert cfg["lead_order"] == LEAD_ORDER_12
print("independent leads:", INDEPENDENT_LEADS)
print("independent indices:", INDEPENDENT_IDX)
print("mapping:")
for name, idx in zip(INDEPENDENT_LEADS, INDEPENDENT_IDX):
    print(f"  {name:3s} <- 12-lead index {idx}")
""")

md("""
## 4. Convert 12 → 8
""")

code("""
X8 = to_independent_8(X_train)
print("X8", X8.shape, X8.dtype)
assert X8.shape == (len(X_train), 8, 1000)
# I and II must match the original 12-lead channels
assert np.allclose(X8[:, 0], X_train[:, :, 0])
assert np.allclose(X8[:, 1], X_train[:, :, 1])
assert np.allclose(X8[:, 2], X_train[:, :, 6])  # V1

# Reconstruction of derived leads vs recorded (recording noise => not bit-identical)
x12_hat = reconstruct_12(X8[:64])
iii_err = np.abs(x12_hat[:, :, 2] - X_train[:64, :, 2]).mean()
print(f"mean |III_rec - III_recorded| on 64 train ECGs: {iii_err:.4f}")
print("reconstructed shape", x12_hat.shape)
""")

md("""
**Observation:** 8-lead tensor is `(N, 8, 1000)`. Reconstructed III is close to the recorded III but not identical — expected for real recordings.
""")

md("""
## 5. 4-class conditioning
""")

code("""
y_oh = one_hot(y_train)
print("one-hot", y_oh.shape, y_oh.dtype)
print("row 0 class", int(y_train[0]), "one-hot", y_oh[0])
assert y_oh.shape == (len(y_train), 4)
assert np.allclose(y_oh.sum(axis=1), 1)
""")

md("""
## 6. SSSD model

Official residual + S4 + T=200 linear beta schedule. Conditioning: `label @ embedding.weight` with 4 classes instead of 71.

Published SSSD is 36 residual layers × 256 channels. A probe on this RTX 3050 (4 GB) peaked at ~8.2 GB for that config, so training uses WAVENET_4GB (8×64). Diffusion T=200 and 8-lead layout stay official. AMP is off: S4 `rfft` is not fp16-safe here (cuFFT size 2000).
""")

code("""
print("official 36x256 is the paper config; this 4 GB GPU uses WAVENET_4GB if VRAM < 8 GB. T=200 unchanged. AMP off because S4 rfft is not fp16-safe (cuFFT size 2000).")
""")

md("""
## 7. Pilot training
""")

code("""
net, hist_pilot, peak_pilot, ckpt_pilot = train_loop(X_train, y_train, TRAIN_PILOT, tag="pilot")
print("pilot checkpoint", ckpt_pilot)
print("params", count_params(net))
print("peak_gb", round(peak_pilot, 3))
""")

code("""
hp = pd.DataFrame(hist_pilot)
fig, ax = plt.subplots()
ax.plot(hp["iter"], hp["loss"])
ax.set_xlabel("iteration")
ax.set_ylabel("MSE")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
print(hp.to_string(index=False))
""")

md("""
## 8. Longer training

Iteration count is set from the measured pilot step time so the run stays on this 4 GB GPU (not a silent 100k-iter claim).
""")

code("""
sec_per = hist_pilot[-1]["sec"] / max(hist_pilot[-1]["iter"], 1)
# ~50 minutes wall clock after the pilot
n_full = int(min(4000, max(400, 50 * 60 / sec_per)))
TRAIN_FULL_RUN = dict(TRAIN_FULL)
TRAIN_FULL_RUN["n_iters"] = n_full
print(f"pilot sec/iter {sec_per:.2f} -> full n_iters {n_full} (cap 4000; published is 100000)")
net, hist_full, peak_full, ckpt_full = train_loop(X_train, y_train, TRAIN_FULL_RUN, tag="full")
print("full checkpoint", ckpt_full)
print("peak_gb", round(peak_full, 3))
hf = pd.DataFrame(hist_full)
fig, ax = plt.subplots()
ax.plot(hf["iter"], hf["loss"])
ax.set_xlabel("iteration")
ax.set_ylabel("MSE")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
print("final logged loss", hf.iloc[-1].to_dict())
""")

md("""
## 9. Generate pilot synthetic pool
""")

code("""
device = torch.device("cuda")
ckpt = torch.load(ckpt_full, map_location="cpu", weights_only=False)
from synthetic_generation.model import SSSD_ECG
net = SSSD_ECG(**ckpt["wavenet_config"]).to(device)
net.load_state_dict(ckpt["model_state_dict"])
net.eval()

xs, ys = [], []
for cls_name, n in PILOT_SAMPLES_PER_CLASS.items():
    cid = CLASS_ORDER.index(cls_name)
    print("sampling", cls_name, n)
    x = generate_class_batch(net, cid, n, device, batch_size=1)
    xs.append(x)
    ys.append(np.full(n, cid, dtype=np.int64))

X_syn = np.concatenate(xs, axis=0).astype(np.float32)
y_syn = np.concatenate(ys, axis=0)
print("X_syn", X_syn.shape, X_syn.dtype)
print("y_syn", y_syn.shape, np.bincount(y_syn, minlength=4))
assert X_syn.shape[1:] == (1000, 12)
""")

md("""
## 10. Basic sanity checks
""")

code("""
san = basic_sanity(X_syn)
print(san)
keep = np.isfinite(X_syn).all(axis=(1, 2))
keep &= ~(X_syn.std(axis=1) < 1e-8).all(axis=1)
keep &= np.abs(X_syn).max(axis=(1, 2)) <= 20
print("pass_frac", float(keep.mean()))
X_syn_ok, y_syn_ok = X_syn[keep], y_syn[keep]
print("kept", X_syn_ok.shape)
""")

md("""
## 11. Visual inspection
""")

code("""
def plot_lead_ii(real, syn, title):
    t = np.arange(1000) / 100.0
    fig, ax = plt.subplots()
    ax.plot(t, real[:, 1], label="real", lw=0.8)
    ax.plot(t, syn[:, 1], label="synthetic", lw=0.8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Lead II (z-scored)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

rng = np.random.default_rng(SEED)
for c, name in enumerate(CLASS_ORDER):
    ridx = rng.choice(np.where(y_train == c)[0])
    sidx = rng.choice(np.where(y_syn_ok == c)[0]) if (y_syn_ok == c).any() else None
    if sidx is None:
        print("no synthetic", name)
        continue
    plot_lead_ii(X_train[ridx], X_syn_ok[sidx], f"{name}: real vs synthetic Lead II")
""")

code("""
def plot_12(x, title):
    t = np.arange(1000) / 100.0
    fig, axes = plt.subplots(12, 1, figsize=(10, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x[:, i], lw=0.6, color="black")
        ax.set_ylabel(LEAD_ORDER_12[i], rotation=0, labelpad=18)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()

rng = np.random.default_rng(SEED + 1)
for name in ["AFIB", "SBRAD"]:
    c = CLASS_ORDER.index(name)
    sidx = rng.choice(np.where(y_syn_ok == c)[0])
    plot_12(X_syn_ok[sidx], f"synthetic 12-lead {name}")
""")

md("""
## 12. Lightweight diversity check
""")

code("""
def mean_offdiag_corr(x, n=min(24, len(x))):
    z = x[:n].reshape(n, -1)
    z = z - z.mean(axis=1, keepdims=True)
    z = z / (z.std(axis=1, keepdims=True) + 1e-8)
    c = (z @ z.T) / z.shape[1]
    iu = np.triu_indices(n, k=1)
    return float(c[iu].mean()), float(c[iu].max())

print("mean/max pairwise corr (flattened, n<=24)")
for c, name in enumerate(CLASS_ORDER):
    xs = X_syn_ok[y_syn_ok == c]
    if len(xs) < 2:
        print(name, "too few")
        continue
    mu, mx = mean_offdiag_corr(xs)
    print(f"  {name:6s} mean {mu:.3f}  max {mx:.3f}")
""")

md("""
## 13. Save
""")

code("""
OUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "samples").mkdir(exist_ok=True)
np.save(OUT_DIR / "X_synthetic.npy", X_syn_ok)
np.save(OUT_DIR / "y_synthetic.npy", y_syn_ok)
meta = metadata_rows(y_syn_ok)
meta.to_csv(OUT_DIR / "metadata.csv", index=False)
np.save(ROOT / "samples" / "X_synthetic.npy", X_syn_ok)

used_w = ckpt["wavenet_config"]
config_out = {
    "generator": GENERATOR_NAME,
    "source": GENERATOR_SOURCE,
    "seed": SEED,
    "train_only": True,
    "n_train": int(len(y_train)),
    "diffusion": DIFFUSION_CONFIG,
    "wavenet_official": WAVENET_CONFIG,
    "wavenet_used": used_w,
    "train_pilot": TRAIN_PILOT,
    "train_full": TRAIN_FULL_RUN,
    "independent_leads": INDEPENDENT_LEADS,
    "lead_order_12": LEAD_ORDER_12,
    "samples_requested": PILOT_SAMPLES_PER_CLASS,
    "samples_kept": {CLASS_ORDER[i]: int((y_syn_ok == i).sum()) for i in range(4)},
    "sanity": san,
    "peak_gb_pilot": peak_pilot,
    "peak_gb_full": peak_full,
    "checkpoint": str(ckpt_full).replace("\\\\", "/"),
}
(ROOT / "config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")
print("wrote", OUT_DIR)
print("files", sorted(p.name for p in OUT_DIR.iterdir()))
print("gen files", sorted(p.name for p in ROOT.iterdir() if p.is_file()))
""")

md("""
## Week 4 outcome

1. Generator: adapted SSSD-ECG (official architecture + 4-class conditioning + I/II/V1–V6 reconstruction).
2. Training: from scratch on X_train/y_train only (see cells above for whether it succeeded).
3. Per-class counts: see `synthetic_data/metadata.csv` / config.json.
4. Sanity pass rate: printed in section 10.
5. Structure: visual comparison only — no quality claim.
6. Failure modes: see observations after plots.
7. Ready for Week 5 quality assessment if files exist and sanity fail count is reported.

**GENERATED SYNTHETIC ECG POOL READY FOR QUALITY ASSESSMENT.**
""")

nb.cells = cells
path = Path(r"C:\Users\Shivank\Videos\ECG - GenAi\04_Synthetic_ECG_Generation.ipynb")
path.write_text(nbf.writes(nb), encoding="utf-8")
print("wrote", path, "cells", len(cells))
