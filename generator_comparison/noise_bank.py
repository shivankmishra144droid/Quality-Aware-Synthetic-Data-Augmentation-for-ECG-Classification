"""Real-noise-bank variants (report/PREREGISTRATION_3.md, part B). Pure functions on one recording.

A donor is a REAL train ECG (8 independent leads, z-space). Its own decomposition (synthetic_generation.beats) is
    x8 = own_tpl@r + residual + wander,   residual = hp - own_tpl@r,   wander = x8 - hp
so swapping the template keeps everything else real:
    syn = T'@r + residual + wander = x8 + (T' - own_tpl)@r
`synthesize` scales that difference by alpha (0 = the donor itself, 1 = full swap). Study 2's ablation T_rand /
T_match (a real template from another recording, no alignment, alpha 1) scored detector AUC ~0.95; part B asks
whether aligning T' to the donor's own beat and shrinking the difference makes it undetectable while still novel.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt

from synthetic_generation.beats import _BP, FS, POST, PRE, TAPER_W, highpass, place_beats, recording_template

QRS = slice(PRE - 6, PRE + 8)        # -60 ms .. +70 ms around R inside a template window


def donor_parts(x8: np.ndarray):
    """(8, T) z-space -> dict(x8, hp, tpl, r) or None if fewer than 3 full beats (as recording_template)."""
    x8 = np.asarray(x8, dtype=np.float64)
    tpl, r, info = recording_template(x8)
    if tpl is None:
        return None
    return {"x8": x8, "hp": highpass(x8), "tpl": tpl, "r": r, "rr_mean": info["rr_mean"]}


def _qrs_energy(tpl: np.ndarray) -> np.ndarray:
    return (np.gradient(filtfilt(*_BP, tpl, axis=-1, padlen=min(15, tpl.shape[-1] - 1)), axis=-1) ** 2).sum(0)


def _shift(tpl: np.ndarray, lag: float) -> np.ndarray:
    """Delay a (8, W) template by `lag` samples (positive = later) with linear interpolation, then re-zero the
    pre-P baseline and taper, exactly the post-processing _beat_gen_pool.py applies to generated templates."""
    t = np.arange(tpl.shape[1], dtype=np.float64)
    out = np.stack([np.interp(t - lag, t, lead, left=lead[0], right=lead[-1]) for lead in tpl])
    return (out - out[:, :3].mean(axis=1, keepdims=True)) * TAPER_W


def estimate_lag(tpl: np.ndarray, ref: np.ndarray, max_lag: int = 4) -> float:
    """Sub-sample lag that best aligns tpl's QRS energy with ref's (cross-correlation + parabolic refinement)."""
    a, b = _qrs_energy(tpl), _qrs_energy(ref)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.dot(np.roll(a, k), b) for k in lags])
    i = int(np.argmax(cc))
    if 0 < i < len(lags) - 1:
        den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
        frac = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den != 0 else 0.0
        return float(lags[i] + np.clip(frac, -0.5, 0.5))
    return float(lags[i])


def align(gen_tpl: np.ndarray, own_tpl: np.ndarray, max_lag: int = 4) -> tuple[np.ndarray, float]:
    lag = estimate_lag(gen_tpl, own_tpl, max_lag)
    return _shift(gen_tpl, lag), lag


def _unit(a: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def nearest_template(bank: np.ndarray, own_tpl: np.ndarray, available: np.ndarray, max_lag: int = 4) -> int:
    """Index (into bank) of the available generated template with the highest correlation to own_tpl at any
    integer lag within +/-max_lag."""
    idx = np.flatnonzero(available)
    ref = _unit(own_tpl[None])[0]
    best = np.full(len(idx), -np.inf)
    for k in range(-max_lag, max_lag + 1):
        best = np.maximum(best, _unit(np.roll(bank[idx], k, axis=-1)) @ ref)
    return int(idx[int(np.argmax(best))])


def synthesize(parts: dict, template: np.ndarray, alpha: float) -> np.ndarray:
    """(8, T): donor + alpha * (template - own template) placed at the donor's own R-peaks."""
    diff = place_beats(np.asarray(template) - parts["tpl"], parts["r"], length=parts["x8"].shape[1])
    return parts["x8"] + alpha * diff


def donor_corr(x_syn: np.ndarray, x_donor: np.ndarray) -> np.ndarray:
    """Whole-signal Pearson correlation of each synthetic ECG with its own donor (rows paired)."""
    return np.sum(_unit(x_syn) * _unit(x_donor), axis=1)


def time_warp(x8: np.ndarray, r: np.ndarray, s: float, length: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Slow the whole donor (all leads identically) by factor s >= 1 (new RR = s * RR), keeping each QRS
    (R-50 ms .. R+60 ms) and the P/PR region (200 ms before the next R) rigid, stretching ST-T by sqrt(s) and
    letting the TP segment absorb the rest. Returns the first `length` samples and the new R positions.
    Linear in the signal, so lead relations (Einthoven/Goldberger) are preserved."""
    if s < 1:
        raise ValueError("only slowing (s >= 1) keeps a full 10 s of donor signal")
    r = np.asarray(r, dtype=np.float64)
    q_post, pr, st = 0.06 * FS, 0.20 * FS, 0.30 * FS            # R-50 ms lies inside the rigid PR window
    src, dst = [0.0], [0.0]
    t_out = r[0] * s                                             # lead-in before the first R scales uniformly
    src.append(r[0]), dst.append(t_out)
    new_r = [t_out]
    for a, b in zip(r[:-1], r[1:]):
        rr = b - a
        st_in = min(st, max(0.0, rr - q_post - pr))
        tp_in = max(0.0, rr - q_post - st_in - pr)
        st_out = st_in * np.sqrt(s)
        tp_out = s * rr - q_post - st_out - min(pr, rr - q_post - st_in)
        for seg_in, seg_out in ((q_post, q_post), (st_in, st_out), (tp_in, max(tp_out, 0.0)),
                                (rr - q_post - st_in - tp_in, rr - q_post - st_in - tp_in)):
            src.append(src[-1] + seg_in)
            dst.append(dst[-1] + seg_out)
        new_r.append(dst[-1])
    tail = (x8.shape[1] - 1) - r[-1]
    src.append(r[-1] + tail), dst.append(dst[-1] + tail * s)
    src, dst = np.asarray(src), np.asarray(dst)
    keep = np.r_[True, np.diff(dst) > 1e-9]
    t_src = np.interp(np.arange(length, dtype=np.float64), dst[keep], src[keep])
    t = np.arange(x8.shape[1], dtype=np.float64)
    out = np.stack([np.interp(t_src, t, lead) for lead in x8])
    new_r = np.asarray(new_r)
    return out, np.round(new_r[new_r < length]).astype(np.int64)
