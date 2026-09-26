"""Pipeline sanity test: can train_loop + generate_class_batch memorize real SBRAD ECGs?
N=1: one clean SBRAD train ECG (tiled so drop_last batching works). Pass = periodicity >= 0.6
and corr > 0.8 with the training ECG. Fail => wiring bug; no budget/capacity will fix it.
N=32 (only if N=1 passes): fail there but pass at N=1 => capacity/optimization, not a bug."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import hr_from_acf, lead_ii_hr_pt, qrs_envelope
from synthetic_generation.config import DATA_DIR, WAVENET_4GB
from synthetic_generation.data import load_train
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch
from synthetic_generation.train import train_loop

OUT = Path("generator_comparison/memorize_test")
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}
SBRAD = 3
PARAM = sys.argv[1] if len(sys.argv) > 1 else "eps"  # "eps" (official) or "v"
X, y = load_train(DATA_DIR)

# Clean SBRAD ECGs: high Lead-II periodicity so the pass bar is reachable.
cand = np.where(y == SBRAD)[0]
per = np.array([hr_from_acf(qrs_envelope(X[i][:, 1].astype(np.float64)))[1] for i in cand])
clean = cand[np.argsort(-per)]


def max_corr(samples, refs):
    s = samples.reshape(len(samples), -1)
    r = refs.reshape(len(refs), -1)
    s = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)
    r = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-8)
    return (s @ r.T / s.shape[1]).max(axis=1)


def run(n):
    idx = clean[:n]
    refs = X[idx]
    reps = max(1, 64 // n)
    x_tr = np.repeat(refs, reps, axis=0)
    y_tr = np.full(len(x_tr), SBRAD, dtype=y.dtype)
    cfg = {"n_iters": 2000, "batch_size": 8, "learning_rate": 2e-4, "warmup_iters": 200,
           "log_every": 250, "ckpt_every": 10 ** 9, "use_amp": False, "ema_decay": 0.999,
           "diffusion_config": DIFF, "parameterization": PARAM}
    d = Path(tempfile.mkdtemp())
    net, hist, _, last = train_loop(x_tr, y_tr, cfg, tag=f"mem{n}_{PARAM}", wavenet_config=WAVENET_4GB,
                                    ckpt_dir=d, history_path=d / "h.json")
    ck = torch.load(last, map_location=device)
    ema = SSSD_ECG(**ck["wavenet_config"]).to(device)
    ema.load_state_dict(ck["ema_state_dict"])
    res = {"n": n, "parameterization": PARAM, "ref_periodicity": [round(float(p), 2) for p in per[np.argsort(-per)][:n][:8]],
           "loss_last": hist[-1]["loss"]}
    for label, model in (("raw", net), ("ema", ema)):
        s = generate_class_batch(model, SBRAD, 8, device, batch_size=8, seed=42, diffusion_config=DIFF,
                                 parameterization=PARAM)
        h = lead_ii_hr_pt(s)
        mc = max_corr(s, refs)
        res[label] = {"std": round(float(s.std()), 3), "periodicity": round(h["median_periodicity"], 3),
                      "hr": round(h["median_hr_peaks"], 1), "median_max_corr": round(float(np.median(mc)), 3)}
        np.save(OUT / f"n{n}_{PARAM}_{label}.npy", s)
    res["pass"] = bool(max(res["raw"]["periodicity"], res["ema"]["periodicity"]) >= 0.6
                       and max(res["raw"]["median_max_corr"], res["ema"]["median_max_corr"]) > 0.8)
    print("RESULT", json.dumps(res), flush=True)
    return res


results = [run(1)]
if results[0]["pass"]:
    results.append(run(32))
(OUT / f"results_{PARAM}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("done")
