"""PTB-XL access: raw database (strat_fold, scp_codes), raw records, and our Week-2 arrays.

Our arrays are raw wfdb mV (no filtering) z-scored per lead with train-only stats
(02_PTBXL_Exploration.ipynb), so mV <-> z is exact: z = (mV - mean) / std.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LEAD_ORDER_12, Paths

SPLITS = ("train", "val", "test")


def load_database(paths: Paths) -> pd.DataFrame:
    if not paths.ptbxl_database.exists():
        raise FileNotFoundError(f"{paths.ptbxl_database} not found. Place PTB-XL 1.0.1 under archive/ (README step 3).")
    db = pd.read_csv(paths.ptbxl_database)
    db["ecg_id"] = db["ecg_id"].astype(int)
    db["scp_dict"] = db["scp_codes"].apply(ast.literal_eval)
    return db.set_index("ecg_id", drop=False)


def scp_code_list(paths: Paths) -> list[str]:
    """All SCP statements, alphabetically. The PTB-XL benchmark's 'all' task (71 labels) uses
    MultiLabelBinarizer, whose classes_ are sorted, so this is the default 71-label order."""
    st = pd.read_csv(paths.scp_statements, index_col=0)
    return sorted(str(c) for c in st.index)


def label_vectors(db: pd.DataFrame, ecg_ids, label_order: list[str]) -> np.ndarray:
    """Multi-hot (N, len(label_order)) from scp_codes keys (confidence ignored, as in the benchmark)."""
    pos = {c: i for i, c in enumerate(label_order)}
    out = np.zeros((len(ecg_ids), len(label_order)), dtype=np.float32)
    for r, eid in enumerate(ecg_ids):
        for code in db.loc[int(eid), "scp_dict"]:
            if code in pos:
                out[r, pos[code]] = 1.0
    return out


def load_raw_record(paths: Paths, filename_lr: str) -> np.ndarray:
    """(1000, 12) mV in LEAD_ORDER_12, exactly as notebook 02 read it."""
    import wfdb  # local import: only phases that need raw records require wfdb

    sig, meta = wfdb.rdsamp(str(paths.ptbxl_dir / filename_lr))
    names = [n.upper() for n in meta["sig_name"]]  # PTB-XL writes AVR/AVL/AVF
    idx = [names.index(n.upper()) for n in LEAD_ORDER_12]
    return sig[:, idx].astype(np.float64)


# ---------------------------------------------------------------- our Week-2 arrays


def norm_stats(paths: Paths) -> tuple[np.ndarray, np.ndarray]:
    st = json.loads((paths.data_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    assert st["lead_order"] == LEAD_ORDER_12, st["lead_order"]
    mean = np.asarray(st["mean"], dtype=np.float64)
    std = np.asarray(st["std"], dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def load_metadata(paths: Paths) -> pd.DataFrame:
    """All three split metadata CSVs with the row index into their X_<split>.npy."""
    frames = []
    for split in SPLITS:
        m = pd.read_csv(paths.data_dir / f"{split}_metadata.csv")
        m["row"] = np.arange(len(m))
        m["split"] = split
        frames.append(m)
    meta = pd.concat(frames, ignore_index=True)
    meta["ecg_id"] = meta["ecg_id"].astype(int)
    return meta.set_index("ecg_id", drop=False)


class OurArrays:
    """Lazy, memory-mapped access to processed_data/X_<split>.npy by ecg_id."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self.meta = load_metadata(paths)
        self._x = {}

    def x(self, split: str) -> np.ndarray:
        if split not in self._x:
            p = self.paths.data_dir / f"X_{split}.npy"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing. Rebuild it with notebook 02 (README step 3).")
            self._x[split] = np.load(p, mmap_mode="r", allow_pickle=False)
        return self._x[split]

    def get(self, ecg_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns x (N,1000,12) z-space float32, y (N,), patient_id (N,)."""
        rows = self.meta.loc[list(map(int, ecg_ids))]
        x = np.empty((len(rows), 1000, 12), dtype=np.float32)
        for split in SPLITS:
            sel = np.flatnonzero(rows["split"].to_numpy() == split)
            if len(sel):
                x[sel] = self.x(split)[rows["row"].to_numpy()[sel]]
        return x, rows["label"].to_numpy().astype(int), rows["patient_id"].to_numpy()
