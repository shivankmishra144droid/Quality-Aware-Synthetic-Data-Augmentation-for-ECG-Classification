"""Load Week 2 training arrays only. Never reads val/test waveforms for training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASS_ORDER, DATA_DIR
from .leads import to_independent_8


def load_train(data_dir: Path = DATA_DIR):
    x = np.load(data_dir / "X_train.npy")
    y = np.load(data_dir / "y_train.npy")
    return x, y


def one_hot(y: np.ndarray, n_classes: int = 4) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return out


def class_counts(y: np.ndarray) -> dict:
    return {CLASS_ORDER[i]: int((y == i).sum()) for i in range(len(CLASS_ORDER))}


class TrainECG8(Dataset):
    def __init__(self, x12: np.ndarray, y: np.ndarray):
        self.x8 = to_independent_8(x12)
        self.y_oh = one_hot(y)

    def __len__(self):
        return len(self.y_oh)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x8[i]),
            torch.from_numpy(self.y_oh[i]),
        )
