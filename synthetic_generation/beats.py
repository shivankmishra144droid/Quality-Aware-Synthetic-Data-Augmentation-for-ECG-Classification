"""Beat templates and rhythm assembly for template-based generation.

A beat is a fixed absolute-time window around its R-peak (PRE samples before, POST after,
at 100 Hz), on the 8 independent leads (I, II, V1-V6) after 0.5 Hz high-pass. Windows are
tapered to zero at both ends and placed by overlap-add, so QRS width never depends on RR:
SBRAD gets a flat TP gap, STACH gets T-P overlap, and joins land in near-zero baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from generator_comparison.eval_metrics import hr_from_acf, hr_from_peaks, qrs_envelope

FS = 100
PRE, POST = 25, 45            # R - 250 ms ... R + 450 ms
WIN = PRE + POST              # 70 samples
TAPER = 8                     # samples of cosine ramp at each end
_HP = butter(2, 0.5 / (FS / 2), btype="high")
_BP = butter(3, [5 / (FS / 2), 15 / (FS / 2)], btype="band")


def taper_window(n: int = WIN, ramp: int = TAPER) -> np.ndarray:
    w = np.ones(n)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    w[:ramp] = r
    w[-ramp:] = r[::-1]
    return w


TAPER_W = taper_window()


def highpass(x8: np.ndarray) -> np.ndarray:
    """(8, T) -> (8, T) baseline-free."""
    return filtfilt(*_HP, x8, axis=-1)


def detect_r_peaks(x8: np.ndarray) -> np.ndarray:
    """R-peak sample indices from Lead II's QRS envelope, refined to the max of the
    8-lead bandpassed derivative energy within +/-60 ms (a consistent fiducial)."""
    env = qrs_envelope(x8[1].astype(np.float64))
    pk, _ = find_peaks(env, height=0.3 * np.percentile(env, 98), distance=int(0.27 * FS))
    e = (np.gradient(filtfilt(*_BP, x8, axis=-1), axis=-1) ** 2).sum(0)
    out = []
    for p in pk:
        lo, hi = max(0, p - 6), min(x8.shape[1], p + 7)
        out.append(lo + int(np.argmax(e[lo:hi])))
    return np.unique(np.asarray(out, dtype=np.int64))


def recording_template(x8: np.ndarray):
    """Median beat of one recording. x8: (8, T) z-scored independent leads.
    Returns (template (8, WIN) or None, r_peaks, info dict)."""
    hp = highpass(x8.astype(np.float64))
    r = detect_r_peaks(x8)
    full = r[(r - PRE >= 0) & (r + POST <= x8.shape[1])]
    info = {"n_peaks": int(len(r)), "n_beats": int(len(full))}
    if len(full) < 3:
        return None, r, info
    beats = np.stack([hp[:, p - PRE:p + POST] for p in full])        # (n, 8, WIN)
    tpl = np.median(beats, axis=0)
    tpl = tpl - tpl[:, :3].mean(axis=1, keepdims=True)               # zero the pre-P baseline
    rr = np.diff(r) / FS
    info.update({"rr_mean": float(rr.mean()), "rr_std": float(rr.std()),
                 "rr_cv": float(rr.std() / rr.mean())})
    return tpl * TAPER_W, r, info


def place_beats(templates: np.ndarray, r_positions, length: int = 1000) -> np.ndarray:
    """Overlap-add. templates: (8, WIN) (same beat everywhere) or (n_beats, 8, WIN).
    Beats whose window falls partly outside [0, length) are clipped, not dropped."""
    out = np.zeros((8, length))
    tpl = np.asarray(templates)
    for k, p in enumerate(np.asarray(r_positions, dtype=np.int64)):
        t = tpl if tpl.ndim == 2 else tpl[k]
        a, b = p - PRE, p + POST
        sa, sb = max(0, a), min(length, b)
        if sb > sa:
            out[:, sa:sb] += t[:, sa - a:sb - a]
    return out


def recording_is_clean(x8: np.ndarray, tol: float = 0.15) -> bool:
    """Peak-count HR and ACF HR agree (the scorecard's heartbeat criterion, per recording)."""
    env = qrs_envelope(x8[1].astype(np.float64))
    hp, (ha, _) = hr_from_peaks(env), hr_from_acf(env)
    return bool(np.isfinite(hp) and np.isfinite(ha) and abs(hp - ha) / ha < tol)
