"""Rhythm assembly around beat templates (rule-based, fitted on the train split only).

- RR model: per class, a KDE over per-recording (log mean RR, log RR coefficient of variation).
  Sinus classes (SR/STACH/SBRAD) get smooth AR(1) RR variability; AFIB gets independent RRs.
- Residual model: what is left after subtracting each recording's own template (noise, beat-to-
  beat variation and, for AFIB, fibrillatory waves) is modelled per class as a stationary
  multichannel Gaussian process with the real 8x8 cross-spectrum, so cross-lead correlation and
  the AFIB f-wave band (strongest in V1) are reproduced. Atrial activity is parametric, not learned.
- Baseline wander: same Gaussian-process model on the < 0.5 Hz part of the signal.
All noise is added on the 8 independent leads; reconstruct_12 derives the other four afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from .beats import FS, PRE, POST, highpass, place_beats, recording_is_clean, recording_template
from .leads import reconstruct_12, to_independent_8

L = 1000
SINUS_AR = 0.8        # lag-1 correlation of sinus RR fluctuations


class CrossSpectralGP:
    """Stationary multichannel Gaussian process with an empirical cross-spectrum."""

    def __init__(self, signals: np.ndarray):          # (n, 8, L)
        f = np.fft.rfft(signals, axis=-1)                # (n, 8, F)
        c = np.einsum("nif,njf->fij", f, f.conj()) / len(signals)
        self.chol = np.stack([np.linalg.cholesky(ci + 1e-9 * np.eye(8)) for ci in c])  # (F, 8, 8)
        self.rms = np.sqrt((signals ** 2).mean(axis=(1, 2)))   # per-recording scale, for sampling

    def sample(self, rng) -> np.ndarray:
        F = self.chol.shape[0]
        z = (rng.standard_normal((F, 8)) + 1j * rng.standard_normal((F, 8))) / np.sqrt(2)
        spec = np.einsum("fij,fj->if", self.chol, z)
        spec[:, 0] = spec[:, 0].real * np.sqrt(2)       # DC / Nyquist bins are real
        spec[:, -1] = spec[:, -1].real * np.sqrt(2)
        x = np.fft.irfft(spec, n=L, axis=-1)
        scale = rng.choice(self.rms) / (np.sqrt((x ** 2).mean()) + 1e-12)
        return x * scale


class RhythmModel:
    def __init__(self, rr_mean: np.ndarray, rr_cv: np.ndarray, irregular: bool):
        self.kde = gaussian_kde(np.vstack([np.log(rr_mean), np.log(np.maximum(rr_cv, 1e-3))]))
        self.irregular = irregular

    def sample_r_positions(self, rng) -> np.ndarray:
        lm, lc = self.kde.resample(1, seed=rng)[:, 0]
        mean, cv = float(np.exp(lm)), float(np.exp(lc))
        mean = float(np.clip(mean, 0.3, 2.0))
        n = int(L / FS / mean) + 4
        if self.irregular:     # AFIB: independent RRs, gamma with the sampled mean and cv
            k = 1.0 / max(cv, 0.05) ** 2
            rr = rng.gamma(k, mean / k, size=n)
        else:                  # sinus: smooth AR(1) fluctuation
            e = np.zeros(n)
            for i in range(1, n):
                e[i] = SINUS_AR * e[i - 1] + np.sqrt(1 - SINUS_AR ** 2) * rng.standard_normal()
            rr = mean * (1 + cv * e)
        rr = np.clip(rr, 0.27, 2.2)
        t0 = rng.uniform(0, rr[0])
        pos = np.round((t0 + np.concatenate([[0], np.cumsum(rr)])) * FS).astype(np.int64)
        return pos[(pos > -POST) & (pos < L + PRE)]


def fit_class_models(X12: np.ndarray, y: np.ndarray, cls: int, n_fit: int, rng):
    """Fit rhythm / residual / wander models and collect templates for one class (train only)."""
    idx = rng.choice(np.where(y == cls)[0], min(n_fit, int((y == cls).sum())), replace=False)
    tpls, rr_m, rr_cv, resid, wander = [], [], [], [], []
    for i in idx:
        x8 = to_independent_8(X12[i:i + 1])[0].astype(np.float64)
        tpl, r, info = recording_template(x8)
        if tpl is None:
            continue
        if cls != 1 and not recording_is_clean(x8):   # AFIB is irregular by definition
            continue
        hp = highpass(x8)
        tpls.append(tpl)
        rr_m.append(info["rr_mean"])
        rr_cv.append(info["rr_cv"])
        resid.append(hp - place_beats(tpl, r))
        wander.append(x8 - hp)
    return {
        "templates": np.stack(tpls),
        "rhythm": RhythmModel(np.array(rr_m), np.array(rr_cv), irregular=(cls == 1)),
        "residual": CrossSpectralGP(np.stack(resid)),
        "wander": CrossSpectralGP(np.stack(wander)),
        "n_used": len(tpls),
    }


def assemble(template: np.ndarray, models: dict, rng, resp_mod: tuple = (0.03, 0.10)) -> np.ndarray:
    """One synthetic 12-lead ECG (L, 12) in the Week-2 z-score space."""
    r = models["rhythm"].sample_r_positions(rng)
    m = rng.uniform(*resp_mod)                          # respiratory amplitude modulation
    f_resp, ph = rng.uniform(0.15, 0.35), rng.uniform(0, 2 * np.pi)
    amp = 1 + m * np.sin(2 * np.pi * f_resp * r / FS + ph)
    beats = template[None] * amp[:, None, None]
    x8 = place_beats(beats, r) + models["residual"].sample(rng) + models["wander"].sample(rng)
    return reconstruct_12(x8[None].astype(np.float32))[0]
