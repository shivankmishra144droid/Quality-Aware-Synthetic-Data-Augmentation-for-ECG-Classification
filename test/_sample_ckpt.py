"""Sample a 32/class pool (raw + EMA weights) from any saved checkpoint.
Parameterization and diffusion schedule are read from the checkpoint, never assumed.
Usage: py -3.14 _sample_ckpt.py <checkpoint.pt> <out_name>
  -> generator_comparison/<out_name>_raw/ and generator_comparison/<out_name>/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch, metadata_rows

ckpt_path, name = Path(sys.argv[1]), sys.argv[2]
device = torch.device("cuda")
ck = torch.load(ckpt_path, map_location=device)
param = ck["train_cfg"].get("parameterization", "eps")
diff = ck["diffusion_config"]
print("ckpt", ckpt_path, "iter", ck["iter"], "param", param, "diffusion", diff)

variants = [("raw", "model_state_dict", Path("generator_comparison") / f"{name}_raw")]
if ck.get("ema_state_dict") is not None:
    variants.append(("ema", "ema_state_dict", Path("generator_comparison") / name))
for label, key, out in variants:
    net = SSSD_ECG(**ck["wavenet_config"]).to(device)
    net.load_state_dict(ck[key])
    xs = [generate_class_batch(net, c, 32, device, batch_size=8, seed=42, diffusion_config=diff,
                               parameterization=param) for c in range(4)]
    X = np.concatenate(xs).astype(np.float32)
    y = np.repeat(np.arange(4), 32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_synthetic.npy", X)
    np.save(out / "y_synthetic.npy", y)
    metadata_rows(y).to_csv(out / "metadata.csv", index=False)
    print(label, "->", out, "std %.3f" % X.std(), {CLASS_ORDER[c]: round(float(xs[c].std()), 3) for c in range(4)})
print("done")
