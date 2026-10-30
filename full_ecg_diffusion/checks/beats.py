"""Rhythm checks with a multi-lead R-peak detector, calibrated on real ECGs before it judges anything.

Detector: 5-15 Hz bandpass on all 12 leads -> per-lead robust scaling -> combined slope energy
-> 80 ms smoothing -> peaks >= 250 ms apart above 30% of the record's 99th percentile.
(The Week-4 lead-II detector in generator_comparison/eval_metrics.py counted T waves and gave
real SBRAD a 96 bpm median, which is why a new one is needed and why it is calibrated first.)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, peak_widths, sosfiltfilt

from ..config import CLASS_ORDER, FS

_SOS = butter(2, [5.0, 15.0], btype="bandpass", fs=FS, output="sos")
MIN_RR_S = 0.25
REL_HEIGHT = 0.30


def detect_r_peaks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (1000, 12) -> (peak indices, envelope)."""
    f = sosfiltfilt(_SOS, np.asarray(x, dtype=np.float64), axis=0)
    mad = np.median(np.abs(f - np.median(f, axis=0)), axis=0) + 1e-8
    slope = np.diff(f / mad, axis=0, prepend=f[:1] / mad)
    env = np.sqrt((slope ** 2).sum(axis=1))
    env = np.convolve(env, np.ones(8) / 8.0, mode="same")
    thr = REL_HEIGHT * np.percentile(env, 99)
    peaks, _ = find_peaks(env, height=thr, distance=int(MIN_RR_S * FS))
    return peaks, env


def beat_features(x: np.ndarray) -> dict:
    peaks, env = detect_r_peaks(x)
    out = {"n_beats": int(len(peaks)), "hr": np.nan, "rr_cv": np.nan, "rmssd_ms": np.nan,
           "qrs_ms": np.nan, "detectable": False}
    if len(peaks) >= 3:
        rr = np.diff(peaks) / FS
        hr = 60.0 / rr.mean()
        out.update(hr=float(hr), rr_cv=float(rr.std() / rr.mean()),
                   rmssd_ms=float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0) if len(rr) > 1 else np.nan,
                   qrs_ms=float(np.median(peak_widths(env, peaks, rel_height=0.5)[0]) * 1000.0 / FS),
                   detectable=bool(25.0 <= hr <= 250.0))
    out["peaks"] = peaks
    return out


def batch_beats(x: np.ndarray) -> list[dict]:
    return [beat_features(r) for r in x]


def _summ(feats: list[dict], y: np.ndarray) -> dict:
    out = {"detectable_frac": float(np.mean([f["detectable"] for f in feats])) if feats else float("nan")}
    for c, name in enumerate(CLASS_ORDER):
        sel = [f for f, yy in zip(feats, y) if yy == c and f["detectable"]]
        n_all = int((y == c).sum())
        out[name] = {
            "n": n_all,
            "detectable_frac": len(sel) / n_all if n_all else float("nan"),
            "median_hr": float(np.median([f["hr"] for f in sel])) if sel else float("nan"),
            "median_rr_cv": float(np.median([f["rr_cv"] for f in sel])) if sel else float("nan"),
        }
    return out


def rhythm_check(real_feats, y_real, syn_feats, y_syn, th) -> dict:
    real = _summ(real_feats, y_real)
    syn = _summ(syn_feats, y_syn)
    present = [n for c, n in enumerate(CLASS_ORDER) if (y_syn == c).any() and (y_real == c).any()]

    def ordering(s):
        hr = {n: s[n]["median_hr"] for n in CLASS_ORDER}
        ok = True
        if "SBRAD" in present:
            ok &= hr["SBRAD"] < th.sbrad_max_hr
        if "STACH" in present:
            ok &= hr["STACH"] > th.stach_min_hr
        if {"SBRAD", "SR"} <= set(present):
            ok &= hr["SBRAD"] < hr["SR"]
        if {"SR", "STACH"} <= set(present):
            ok &= hr["SR"] < hr["STACH"]
        return bool(ok)

    def afib_ratio(s):
        a, r = s["AFIB"]["median_rr_cv"], s["SR"]["median_rr_cv"]
        return a / r if np.isfinite(a) and np.isfinite(r) and r > 0 else float("nan")

    calibrated = real["detectable_frac"] >= 0.90 and ordering(real)
    hr_diff = {n: abs(syn[n]["median_hr"] - real[n]["median_hr"]) for n in present}
    checks = {
        "detectable": syn["detectable_frac"] >= th.detectable_ratio * real["detectable_frac"],
        "hr_within_tol": all(np.isfinite(v) and v <= th.hr_tol_bpm for v in hr_diff.values()),
        "class_ordering": ordering(syn),
    }
    if {"AFIB", "SR"} <= set(present):
        rr, rs = afib_ratio(real), afib_ratio(syn)
        checks["afib_irregular"] = bool(np.isfinite(rs) and rs > 1.0 and rs >= th.afib_irregularity_ratio * rr)
    else:
        rr = rs = float("nan")
    return {
        "detector_calibrated_on_real": bool(calibrated),
        "real": real, "syn": syn, "abs_hr_diff": hr_diff,
        "afib_sr_rrcv_ratio": {"real": rr, "syn": rs},
        "checks": checks,
        "pass": bool(calibrated and all(checks.values())),
    }
