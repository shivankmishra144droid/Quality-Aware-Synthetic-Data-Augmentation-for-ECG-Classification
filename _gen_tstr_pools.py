"""Generate larger class-balanced pools for TSTR (raw weights; schedule and parameterization are read
from each checkpoint). Writes generator_comparison/tstr/pool_<name>.npz with X (N,1000,12), y.
Usage: py -3.14 _gen_tstr_pools.py <n_per_class>"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from synthetic_generation.model import SSSD_ECG
from synthetic_generation.sample import generate_class_batch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 440
OUT = Path("generator_comparison/tstr")
OUT.mkdir(parents=True, exist_ok=True)
POOLS = {
    "v1": "generator_comparison/sssd/checkpoint.pt",                                  # Week-4 selected generator
    "vpred5k": "generator_comparison/sssd_vpred/checkpoints/vpred_8x64_5000.pt",      # best diffusion pool (6/9)
}
device = torch.device("cuda")
for name, path in POOLS.items():
    if (OUT / f"pool_{name}.npz").exists():
        print("exists", name)
        continue
    ck = torch.load(path, map_location=device, weights_only=False)
    param = (ck.get("train_cfg") or {}).get("parameterization", "eps")
    net = SSSD_ECG(**ck["wavenet_config"]).to(device)
    net.load_state_dict(ck["model_state_dict"])
    t0 = time.time()
    xs = [generate_class_batch(net, c, N, device, batch_size=40, seed=1234, diffusion_config=ck["diffusion_config"],
                               parameterization=param) for c in range(4)]
    X = np.concatenate(xs).astype(np.float32)
    y = np.repeat(np.arange(4), N)
    np.savez_compressed(OUT / f"pool_{name}.npz", X=X, y=y)
    print(name, "param", param, "std %.3f" % X.std(), "%.0f s" % (time.time() - t0), flush=True)
print("done")
