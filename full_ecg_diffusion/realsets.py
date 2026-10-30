"""Resolve real-record sets: manifest parts ("part:<name>") or whole splits ("split:<name>")."""
from __future__ import annotations

import numpy as np

from .config import SEALED_PART
from .ptbxl import OurArrays
from .splits import load_manifest


class SealedPartError(RuntimeError):
    pass


def resolve_ids(specs, paths, arrays: OurArrays, allow_sealed: bool = False) -> list[int]:
    manifest = None
    ids = []
    for spec in specs:
        kind, _, name = spec.partition(":")
        if kind == "part":
            if name == SEALED_PART and not allow_sealed:
                raise SealedPartError(f"'{SEALED_PART}' is reserved for the one-off final test (final_test.py)")
            manifest = manifest or load_manifest(paths)
            ids += manifest["parts"][name]["ecg_ids"]
        elif kind == "split":
            ids += arrays.meta.loc[arrays.meta["split"] == name, "ecg_id"].astype(int).tolist()
        else:
            raise ValueError(f"bad real-set spec {spec!r}; use part:<name> or split:<name>")
    return sorted(set(ids))


def load_real(specs, paths, arrays: OurArrays | None = None, allow_sealed: bool = False):
    arrays = arrays or OurArrays(paths)
    ids = resolve_ids(specs, paths, arrays, allow_sealed)
    x, y, pid = arrays.get(ids)
    return x, y, pid, np.asarray(ids)


def class_matched(y_a: np.ndarray, y_b: np.ndarray, max_per_class: int, seed: int):
    """Indices into a and b with equal per-class counts (min of both, capped)."""
    rng = np.random.default_rng(seed)
    ia, ib = [], []
    for c in np.union1d(np.unique(y_a), np.unique(y_b)):
        a, b = np.flatnonzero(y_a == c), np.flatnonzero(y_b == c)
        n = min(len(a), len(b), max_per_class)
        if n == 0:
            continue
        ia.append(rng.choice(a, n, replace=False))
        ib.append(rng.choice(b, n, replace=False))
    return np.sort(np.concatenate(ia)), np.sort(np.concatenate(ib))
