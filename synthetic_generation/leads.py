"""Lead conversion for adapted SSSD-ECG.

Week 2 / PTB-XL wfdb order (from processed_data/dataset_config.json and
02_PTBXL_Exploration.ipynb LEAD_NAMES):

  0 I, 1 II, 2 III, 3 aVR, 4 aVL, 5 aVF, 6 V1, 7 V2, 8 V3, 9 V4, 10 V5, 11 V6

Official SSSD-ECG trains on 8 independent leads. This project uses
I, II, V1–V6 (not the I / V1–V6 / aVF indexing in their inference.py).

Derived leads (standard Einthoven / Goldberger):
  III = II - I
  aVR = -(I + II) / 2
  aVL = I - II / 2
  aVF = II - I / 2

These hold in millivolts, not in per-lead z-score space. reconstruct_12 therefore
de-normalizes I and II with the train-only stats, applies the relations, and
re-normalizes each derived lead with its own stats. (Pools made before this fix
applied the relations in z-space: shape exact, aVR/aVL/aVF amplitude ~13% low.)
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .config import DATA_DIR

LEAD_ORDER_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_IDX = [LEAD_ORDER_12.index(name) for name in INDEPENDENT_LEADS]


def to_independent_8(x_n_t_12: np.ndarray) -> np.ndarray:
    """(N, 1000, 12) channels-last -> (N, 8, 1000) channels-first for SSSD."""
    if x_n_t_12.ndim != 3 or x_n_t_12.shape[1:] != (1000, 12):
        raise ValueError(f"expected (N, 1000, 12), got {x_n_t_12.shape}")
    x8 = x_n_t_12[:, :, INDEPENDENT_IDX]
    return np.transpose(x8, (0, 2, 1)).copy()


@lru_cache(maxsize=1)
def _norm_stats() -> tuple[np.ndarray, np.ndarray]:
    st = json.loads((DATA_DIR / "normalization_stats.json").read_text(encoding="utf-8"))
    assert st["lead_order"] == LEAD_ORDER_12
    return np.asarray(st["mean"], np.float64), np.asarray(st["std"], np.float64)


def reconstruct_12(x_n_8_t: np.ndarray) -> np.ndarray:
    """(N, 8, 1000) I,II,V1..V6 -> (N, 1000, 12) in Week 2 lead order, all in z-score space."""
    if x_n_8_t.ndim != 3 or x_n_8_t.shape[1] != 8 or x_n_8_t.shape[2] != 1000:
        raise ValueError(f"expected (N, 8, 1000), got {x_n_8_t.shape}")
    mean, std = _norm_stats()
    lead_i = x_n_8_t[:, 0, :]
    lead_ii = x_n_8_t[:, 1, :]
    v = x_n_8_t[:, 2:8, :]
    i_mv = lead_i * std[0] + mean[0]
    ii_mv = lead_ii * std[1] + mean[1]

    def z(x_mv, idx):
        return (x_mv - mean[idx]) / std[idx]

    lead_iii = z(ii_mv - i_mv, 2)
    lead_avr = z(-(i_mv + ii_mv) / 2.0, 3)
    lead_avl = z(i_mv - ii_mv / 2.0, 4)
    lead_avf = z(ii_mv - i_mv / 2.0, 5)
    x12 = np.stack(
        [lead_i, lead_ii, lead_iii, lead_avr, lead_avl, lead_avf,
         v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5]],
        axis=-1,
    )
    return x12.astype(np.float32, copy=False)
