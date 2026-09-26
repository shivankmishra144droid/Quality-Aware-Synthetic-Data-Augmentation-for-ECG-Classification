"""Pilot comparison metrics only. Not Week 5 quality filtering."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

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


_BP = butter(3, [5 / (FS / 2), 15 / (FS / 2)], btype="band")


def qrs_envelope(sig: np.ndarray) -> np.ndarray:
    """Pan-Tompkins style: 5-15 Hz bandpass -> derivative -> square -> 150 ms moving average.
    Polarity-free and amplitude-free, so synthetic damping does not change the count."""
    f = filtfilt(*_BP, sig)
    d = np.gradient(f)
    w = int(0.15 * FS)
    return np.convolve(d * d, np.ones(w) / w, mode="same")


def hr_from_peaks(env: np.ndarray) -> float:
    peaks, _ = find_peaks(env, height=0.3 * np.percentile(env, 98), distance=int(0.27 * FS))
    return float(60.0 / (np.diff(peaks).mean() / FS)) if len(peaks) >= 2 else float("nan")


def hr_from_acf(env: np.ndarray) -> tuple[float, float]:
    """Dominant period of the QRS envelope over 30-220 bpm, and its ACF height
    (periodicity strength: ~1 = clean regular beats, ~0 = no rhythm)."""
    e = env - env.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.27 * FS), int(2.0 * FS)
    seg = ac[lo:hi + 1]
    p, _ = find_peaks(seg)
    if len(p) == 0:
        return float("nan"), 0.0
    k = p[np.argmax(seg[p])]
    return float(60.0 * FS / (k + lo)), float(seg[k])


def lead_ii_hr_pt(x12: np.ndarray, agree_tol: float = 0.15) -> dict:
    """Replacement for lead_ii_hr. The old raw find_peaks counted T/P waves
    (real SBRAD read ~96 bpm). Peak HR is cross-checked against an autocorrelation period."""
    hp, ha, st = [], [], []
    for sig in x12:
        env = qrs_envelope(np.asarray(sig[:, 1], dtype=np.float64))
        hp.append(hr_from_peaks(env))
        h, s = hr_from_acf(env)
        ha.append(h)
        st.append(s)
    hp, ha, st = np.asarray(hp), np.asarray(ha), np.asarray(st)
    ok = np.isfinite(hp) & np.isfinite(ha)
    agree = np.abs(hp[ok] - ha[ok]) / ha[ok] < agree_tol

    def med(v):
        return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

    return {
        "n": int(len(hp)),
        "median_hr_peaks": med(hp),
        "median_hr_acf": med(ha),
        "iqr_hr_peaks": [float(np.nanpercentile(hp, 25)), float(np.nanpercentile(hp, 75))] if ok.any() else None,
        "agree_frac": float(agree.mean()) if ok.any() else float("nan"),
        "median_periodicity": float(np.median(st)),
        "detectable_frac": float(np.isfinite(hp).mean()),
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
