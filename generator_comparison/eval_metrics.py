"""Pilot comparison metrics only. Not Week 5 quality filtering."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from synthetic_generation.sample import basic_sanity

LEAD_ORDER_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_ORDER = ["SR", "AFIB", "STACH", "SBRAD"]
FS = 100


def waveform_stats(x12: np.ndarray) -> dict:
    return {
        "mean": float(x12.mean()),
        "std": float(x12.std()),
        "amp_min": float(x12.min()),
        "amp_max": float(x12.max()),
        "per_lead_std": {LEAD_ORDER_12[i]: float(x12[:, :, i].std()) for i in range(12)},
    }


def lead_ii_hr(x12: np.ndarray) -> dict:
    hrs, npeaks, rr_std = [], [], []
    for sig in x12:
        ii = sig[:, 1]
        ii = (ii - ii.mean()) / (ii.std() + 1e-8)
        peaks, _ = find_peaks(ii, distance=int(0.30 * FS), prominence=0.35)
        npeaks.append(int(len(peaks)))
        if len(peaks) >= 2:
            rr = np.diff(peaks) / FS
            hrs.append(float(60.0 / rr.mean()))
            rr_std.append(float(rr.std()))
        else:
            hrs.append(np.nan)
            rr_std.append(np.nan)
    hrs = np.asarray(hrs, dtype=np.float64)
    return {
        "median_hr": float(np.nanmedian(hrs)) if np.isfinite(hrs).any() else float("nan"),
        "mean_hr": float(np.nanmean(hrs)) if np.isfinite(hrs).any() else float("nan"),
        "detectable_frac": float(np.isfinite(hrs).mean()),
        "mean_r_count": float(np.mean(npeaks)),
        "median_rr_std": float(np.nanmedian(rr_std)) if np.isfinite(rr_std).any() else float("nan"),
    }


def pairwise_corr(x12: np.ndarray, n_max: int = 32) -> dict:
    n = min(n_max, len(x12))
    if n < 2:
        return {"n": n, "mean": float("nan"), "max": float("nan"), "near_dup_frac": float("nan")}
    z = x12[:n].reshape(n, -1)
    z = z - z.mean(axis=1, keepdims=True)
    z = z / (z.std(axis=1, keepdims=True) + 1e-8)
    c = (z @ z.T) / z.shape[1]
    iu = np.triu_indices(n, k=1)
    vals = c[iu]
    return {
        "n": n,
        "mean": float(vals.mean()),
        "max": float(vals.max()),
        "near_dup_frac": float((vals > 0.99).mean()),
    }


def per_class_hr_table(x12: np.ndarray, y: np.ndarray, x_real: np.ndarray, y_real: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(CLASS_ORDER):
        syn = x12[y == i]
        real = x_real[y_real == i]
        if len(syn) == 0:
            continue
        hs = lead_ii_hr(syn)
        hr = lead_ii_hr(real[: min(256, len(real))])
        out[name] = {
            "syn_median_hr": hs["median_hr"],
            "real_median_hr": hr["median_hr"],
            "abs_hr_diff": abs(hs["median_hr"] - hr["median_hr"])
            if np.isfinite(hs["median_hr"]) and np.isfinite(hr["median_hr"])
            else float("nan"),
            "syn_r_count": hs["mean_r_count"],
            "real_r_count": hr["mean_r_count"],
            "syn_detectable": hs["detectable_frac"],
        }
    return out


def summarize_generator(name: str, x12: np.ndarray | None, y: np.ndarray | None, x_real, y_real) -> dict:
    if x12 is None:
        return {"generator": name, "runnable": False}
    san = basic_sanity(x12)
    corr = pairwise_corr(x12)
    hr_tab = per_class_hr_table(x12, y, x_real, y_real)
    hr_mae = np.nanmean([v["abs_hr_diff"] for v in hr_tab.values()])
    return {
        "generator": name,
        "runnable": True,
        "n": int(len(x12)),
        "counts": {CLASS_ORDER[i]: int((y == i).sum()) for i in range(4)},
        "validity": san,
        "waveform": waveform_stats(x12),
        "real_waveform": waveform_stats(x_real),
        "diversity": corr,
        "hr_by_class": hr_tab,
        "hr_mae_vs_real": float(hr_mae) if np.isfinite(hr_mae) else float("nan"),
    }
