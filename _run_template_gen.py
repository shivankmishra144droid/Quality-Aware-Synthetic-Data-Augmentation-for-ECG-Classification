"""Phase 1.3: template generator. v-prediction SSSD (8x64, s4_lmax=70) trained on clean
per-recording median beat templates (8 independent leads x 70 samples = R-250..R+450 ms),
class-conditional, class-balanced sampling (SR is 85% of templates).
Evals: realism = median max-corr of generated templates to real same-class templates,
copy_frac = fraction of generated templates with corr > 0.98 to ANY real template.
Real-to-real reference (each real template vs its nearest OTHER real template, 200/class):
  NN corr median SR 0.937 / AFIB 0.905 / STACH 0.882 / SBRAD 0.907; frac > 0.98: 0.5-3 %.
So: realism ~0.88-0.94 and copy_frac ~0.0-0.05 look like real; copy_frac >> 0.05 = memorising.
Writes generator_comparison/template_gen/ (checkpoints gitignored)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.config import WAVENET_4GB
from synthetic_generation.sample import generate_class_batch
from synthetic_generation.train import train_loop

ROOT = Path("generator_comparison")
OUT = ROOT / "template_gen"
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")
DIFF = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}
WAVENET_TPL = {**WAVENET_4GB, "s4_lmax": 70}

d = np.load(ROOT / "beat_templates" / "templates_train.npz")
keep = d["clean"]
T, y = d["templates"][keep].astype(np.float32), d["y"][keep].astype(np.int64)
print("templates", {CLASS_ORDER[c]: int((y == c).sum()) for c in range(4)})


def zrows(a):
    a = a.reshape(len(a), -1)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)


REF = {c: zrows(T[y == c]) for c in range(4)}
REF_ALL = zrows(T)


def template_stats(model, n=16):
    out = {}
    for c in range(4):
        g = generate_class_batch(model, c, n, device, batch_size=n, seed=11, diffusion_config=DIFF,
                                 parameterization="v", length=70, return_x8=True)
        z = zrows(g)
        same = (z @ REF[c].T / z.shape[1]).max(1)
        anyr = (z @ REF_ALL.T / z.shape[1]).max(1)
        out[CLASS_ORDER[c]] = {"std": round(float(g.std()), 3), "realism": round(float(np.median(same)), 3),
                               "copy_frac": round(float((anyr > 0.98).mean()), 3)}
    return out


def eval_fn(net, ema, n_iter):
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    res = {"raw": template_stats(net), "ema": template_stats(ema)}
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(cuda)
    return res


train_cfg = {
    "n_iters": 15000, "batch_size": 64, "learning_rate": 2e-4, "warmup_iters": 500, "cosine_decay": True,
    "log_every": 250, "ckpt_every": 2500, "eval_every": 2500, "use_amp": False,
    "ema_decay": 0.999, "ema_warmup": True, "diffusion_config": DIFF, "parameterization": "v",
    "balanced_sampling": True,
}
(OUT / "plan.json").write_text(json.dumps({"train_cfg": train_cfg, "wavenet": WAVENET_TPL,
                                           "n_templates": int(len(T))}, indent=2), encoding="utf-8")
net, hist, peak, last = train_loop(T, y, train_cfg, tag="tpl_8x64", wavenet_config=WAVENET_TPL,
                                   ckpt_dir=OUT / "checkpoints", history_path=OUT / "history.json",
                                   eval_fn=eval_fn)
print("done", last)
