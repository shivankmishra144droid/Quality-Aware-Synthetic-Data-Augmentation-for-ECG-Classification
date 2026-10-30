"""Feature judge: per-record clinical/statistical features, KS distance real vs synthetic.

Pass/fail uses the pooled (all classes, class-matched) KS so small classes do not dominate by
noise; per-class KS and a real-vs-real noise floor are reported alongside.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import ks_2samp

from ..config import CLASS_ORDER, FS, LEAD_ORDER_12

BANDS = {"p_0.5_5": (0.5, 5.0), "p_5_15": (5.0, 15.0), "p_15_40": (15.0, 40.0)}
CORR_PAIRS = [("I", "II"), ("II", "V2"), ("V1", "V6"), ("V2", "V5"), ("I", "V6"), ("aVF", "V3")]


def record_features(x: np.ndarray, beats: dict) -> dict:
    """x (1000, 12) z-space."""
    f = {k: beats[k] for k in ("hr", "rr_cv", "rmssd_ms", "qrs_ms")}
    peaks = beats["peaks"]
    for j, lead in enumerate(LEAD_ORDER_12):
        s = x[:, j]
        f[f"std_{lead}"] = float(s.std())
        if len(peaks):
            win = [s[max(0, p - 5): p + 6] for p in peaks]
            base = np.median(s)
            f[f"r_amp_{lead}"] = float(np.median([w.max() for w in win]) - base)
            f[f"s_amp_{lead}"] = float(np.median([w.min() for w in win]) - base)
        else:
            f[f"r_amp_{lead}"] = f[f"s_amp_{lead}"] = np.nan
    fr, p = welch(x, fs=FS, nperseg=256, axis=0)
    tot = p[(fr >= 0.5) & (fr <= 45)].sum(axis=0) + 1e-12
    for name, (lo, hi) in BANDS.items():
        f[name] = float(np.mean(p[(fr >= lo) & (fr < hi)].sum(axis=0) / tot))
    for a, b in CORR_PAIRS:
        ia, ib = LEAD_ORDER_12.index(a), LEAD_ORDER_12.index(b)
        f[f"corr_{a}_{b}"] = float(np.corrcoef(x[:, ia], x[:, ib])[0, 1])
    return f


def feature_matrix(x: np.ndarray, beats: list[dict]) -> tuple[np.ndarray, list[str]]:
    rows = [record_features(r, b) for r, b in zip(x, beats)]
    names = list(rows[0].keys())
    return np.array([[r[k] for k in names] for r in rows], dtype=np.float64), names


def _ks_table(a: np.ndarray, b: np.ndarray, names: list[str]) -> dict:
    out = {}
    for j, n in enumerate(names):
        aa, bb = a[:, j], b[:, j]
        aa, bb = aa[np.isfinite(aa)], bb[np.isfinite(bb)]
        out[n] = float(ks_2samp(aa, bb).statistic) if len(aa) > 1 and len(bb) > 1 else 1.0
    return out


def feature_check(f_real, y_real, f_syn, y_syn, names, th, seed: int = 0) -> dict:
    pooled = _ks_table(f_real, f_syn, names)
    per_class = {}
    for c, cname in enumerate(CLASS_ORDER):
        if (y_real == c).sum() > 1 and (y_syn == c).sum() > 1:
            t = _ks_table(f_real[y_real == c], f_syn[y_syn == c], names)
            per_class[cname] = {"median_ks": float(np.median(list(t.values()))), "max_ks": float(max(t.values())),
                                "worst": sorted(t, key=t.get, reverse=True)[:5]}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(f_real))
    half = len(perm) // 2
    floor = _ks_table(f_real[perm[:half]], f_real[perm[half:]], names)
    med, mx = float(np.median(list(pooled.values()))), float(max(pooled.values()))
    return {
        "pooled_ks": pooled,
        "median_ks": med, "max_ks": mx,
        "worst_features": sorted(pooled, key=pooled.get, reverse=True)[:8],
        "real_vs_real_floor": {"median_ks": float(np.median(list(floor.values()))), "max_ks": float(max(floor.values()))},
        "per_class": per_class,
        "pass": bool(med <= th.feature_median_ks and mx <= th.feature_max_ks),
    }
