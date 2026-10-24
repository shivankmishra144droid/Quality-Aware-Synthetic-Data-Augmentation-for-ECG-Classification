"""Build an assembled 12-lead pool from GENERATED beat templates (Phase 1.3 / 1.4).
Same assembly models as the oracle pool (generator_comparison/beat_templates/assembly_models.pkl),
so oracle vs generated differs only in where the templates come from.

Usage: py -3.14 _beat_gen_pool.py <template ckpt.pt> <n_per_class | SR,AFIB,STACH,SBRAD> <out.npz> [--weights ema|raw] [--scorecard-dir DIR]
  out.npz: X (N, 1000, 12), y (N,), templates (N, 8, 70)
  --scorecard-dir also writes a 32/class pool (X_synthetic.npy / y_synthetic.npy / metadata.csv) there.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.assembly import assemble
from synthetic_generation.beats import TAPER_W
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch, metadata_rows

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("n_per_class", help="one number for every class, or four comma-separated counts "
                                     "in SR,AFIB,STACH,SBRAD order (0 skips a class), e.g. 0,3200,1800,1400")
ap.add_argument("out")
ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
ap.add_argument("--scorecard-dir", default=None)
ap.add_argument("--seed", type=int, default=2024)
a = ap.parse_args()

device = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=device, weights_only=False)
key = "ema_state_dict" if a.weights == "ema" else "model_state_dict"
net = SSSD_ECG(**ck["wavenet_config"]).to(device)
net.load_state_dict(ck[key])
L = ck["wavenet_config"]["s4_lmax"]
param = ck["train_cfg"].get("parameterization", "eps")
models = pickle.load(open("generator_comparison/beat_templates/assembly_models.pkl", "rb"))
rng = np.random.default_rng(a.seed)

t0 = time.time()
xs, ys, ts = [], [], []
counts = [int(v) for v in a.n_per_class.split(",")]
counts = counts * 4 if len(counts) == 1 else counts
assert len(counts) == 4, "give one count or four"
for c in range(4):
    if counts[c] == 0:
        continue
    tpl = generate_class_batch(net, c, counts[c], device, batch_size=256, seed=a.seed + c,
                               diffusion_config=ck["diffusion_config"], parameterization=param,
                               length=L, return_x8=True).astype(np.float64)
    # Same post-processing as real templates at extraction (beats.recording_template): zero the
    # pre-P baseline and taper both ends, so overlap-add joins stay at zero.
    tpl = (tpl - tpl[:, :, :3].mean(axis=2, keepdims=True)) * TAPER_W
    for t in tpl:
        xs.append(assemble(t, models[c], rng))
        ys.append(c)
        ts.append(t)
    print(CLASS_ORDER[c], "templates std %.3f" % tpl.std(), flush=True)
X, y = np.stack(xs).astype(np.float32), np.asarray(ys, np.int64)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, X=X, y=y, templates=np.stack(ts).astype(np.float32))
print("saved", a.out, X.shape, "std %.3f" % X.std(), "%.0f s" % (time.time() - t0))

if a.scorecard_dir:
    d = Path(a.scorecard_dir)
    d.mkdir(parents=True, exist_ok=True)
    idx = np.concatenate([np.where(y == c)[0][:32] for c in range(4) if (y == c).any()])
    np.save(d / "X_synthetic.npy", X[idx])
    np.save(d / "y_synthetic.npy", y[idx])
    metadata_rows(y[idx]).assign(generator=f"beat-gen ({Path(a.ckpt).name}, {a.weights}) + rule-based assembly").to_csv(
        d / "metadata.csv", index=False)
    print("scorecard pool ->", d)
