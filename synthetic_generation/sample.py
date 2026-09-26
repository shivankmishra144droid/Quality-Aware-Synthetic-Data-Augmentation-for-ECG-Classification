"""Class-conditional sampling + 8→12 reconstruction. Train split only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import CLASS_ORDER, DIFFUSION_CONFIG, GENERATOR_NAME, SEED
from .leads import reconstruct_12
from .utils import calc_diffusion_hyperparams, hyperparams_to_device, sampling_label, set_seed


@torch.no_grad()
def generate_class_batch(
    net, class_id: int, n: int, device, batch_size: int = 1, seed: int | None = None,
    guidance_w: float | None = None, diffusion_config: dict | None = None, clip_x0: float | None = None,
    parameterization: str = "eps",
) -> np.ndarray:
    set_seed((SEED if seed is None else seed) + 1000 + class_id)
    dh = hyperparams_to_device(calc_diffusion_hyperparams(**(diffusion_config or DIFFUSION_CONFIG)), device)
    net.eval()
    chunks = []
    left = n
    while left > 0:
        b = min(batch_size, left)
        cond = torch.zeros(b, 4, device=device)
        cond[:, class_id] = 1.0
        x8 = sampling_label(net, (b, 8, 1000), dh, cond=cond, guidance_w=guidance_w, clip_x0=clip_x0,
                            parameterization=parameterization)
        chunks.append(x8.detach().cpu().numpy())
        left -= b
    x8 = np.concatenate(chunks, axis=0)
    return reconstruct_12(x8)


def basic_sanity(x12: np.ndarray) -> dict:
    """Week 4 validity only — not a quality score."""
    n = len(x12)
    nan_inf = int((~np.isfinite(x12)).any(axis=(1, 2)).sum())
    flat = int((x12.std(axis=1) < 1e-8).all(axis=1).sum())
    extreme = int((np.abs(x12).max(axis=(1, 2)) > 20).sum())  # z-scored units
    bad_shape = 0 if x12.shape[1:] == (1000, 12) else n
    fail = int((((~np.isfinite(x12)).any(axis=(1, 2))) | ((x12.std(axis=1) < 1e-8).all(axis=1)) | (np.abs(x12).max(axis=(1, 2)) > 20)).sum())
    return {"n": n, "nan_inf": nan_inf, "flatline": flat, "extreme_amp": extreme, "bad_shape": bad_shape, "fail": fail}


def metadata_rows(y: np.ndarray, start_id: int = 0) -> pd.DataFrame:
    rows = []
    for i, cls in enumerate(y):
        rows.append({
            "sample_id": f"syn_{start_id + i:06d}",
            "class_id": int(cls),
            "class_name": CLASS_ORDER[int(cls)],
            "generator": GENERATOR_NAME,
            "seed": SEED,
            "source_split": "train",
            "sampling_rate": 100,
            "duration_seconds": 10,
            "num_leads": 12,
            "sample_index": start_id + i,
        })
    return pd.DataFrame(rows)
