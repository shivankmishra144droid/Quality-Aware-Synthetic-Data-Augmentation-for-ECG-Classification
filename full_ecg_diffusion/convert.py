"""The single format conversion used everywhere, plus lead-order and unit inference.

A *layout* describes a foreign ECG array:
  {"axis": "channels_first"|"channels_last", "channels": [lead names in stored order],
   "units": "mV"|"uV", "scale_to_mV": float}
Conversion to our space: reorder -> mV -> rebuild missing limb leads (Einthoven/Goldberger)
-> z = (mV - mean) / std with our train-only stats. Chest leads V1-V6 must all be present.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from .config import CHEST_LEADS, LEAD_ORDER_12, LIMB_LEADS, N_T

# Every limb lead as a combination of (I, II).
LIMB_COEF = {
    "I": (1.0, 0.0), "II": (0.0, 1.0), "III": (-1.0, 1.0),
    "aVR": (-0.5, -0.5), "aVL": (1.0, -0.5), "aVF": (-0.5, 1.0),
}
UNIT_SCALES = {"mV": 1.0, "uV": 1e-3}


# ---------------------------------------------------------------- conversion


def to_channels_first(x: np.ndarray, axis: str) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"expected 3-D array, got {x.shape}")
    return np.transpose(x, (0, 2, 1)) if axis == "channels_last" else x


def rebuild_12_mV(x_cf_mV: np.ndarray, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    """(N, C, T) mV in `channels` order -> (N, T, 12) mV in LEAD_ORDER_12, plus names rebuilt."""
    missing_chest = [c for c in CHEST_LEADS if c not in channels]
    if missing_chest:
        raise ValueError(f"layout lacks chest leads {missing_chest}")
    limb_present = [c for c in channels if c in LIMB_COEF]
    if len(limb_present) < 2:
        raise ValueError("need at least two limb leads to rebuild the hexaxial set")
    n, _, t = x_cf_mV.shape
    out = np.empty((n, t, 12), dtype=np.float64)
    pos = {name: i for i, name in enumerate(channels)}
    # Solve (I, II) from the first two independent limb leads present.
    a, b = None, None
    for p, q in combinations(limb_present, 2):
        m = np.array([LIMB_COEF[p], LIMB_COEF[q]])
        if abs(np.linalg.det(m)) > 1e-9:
            a, b = p, q
            break
    inv = np.linalg.inv(np.array([LIMB_COEF[a], LIMB_COEF[b]]))
    xa, xb = x_cf_mV[:, pos[a], :], x_cf_mV[:, pos[b], :]
    lead_i = inv[0, 0] * xa + inv[0, 1] * xb
    lead_ii = inv[1, 0] * xa + inv[1, 1] * xb
    rebuilt = []
    for j, name in enumerate(LEAD_ORDER_12):
        if name in pos:
            out[:, :, j] = x_cf_mV[:, pos[name], :]
        else:
            ci, cii = LIMB_COEF[name]
            out[:, :, j] = ci * lead_i + cii * lead_ii
            rebuilt.append(name)
    return out, rebuilt


def foreign_to_z(x: np.ndarray, layout: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Foreign array in `layout` -> (N, 1000, 12) float32 in our train-only z-space."""
    x_cf = to_channels_first(np.asarray(x, dtype=np.float64), layout["axis"])
    if x_cf.shape[1] != len(layout["channels"]) or x_cf.shape[2] != N_T:
        raise ValueError(f"shape {x_cf.shape} does not match layout {layout['channels']}")
    mv12, _ = rebuild_12_mV(x_cf * float(layout["scale_to_mV"]), layout["channels"])
    return ((mv12 - mean) / std).astype(np.float32)


def z_to_foreign(x_z: np.ndarray, layout: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse direction: our (N, 1000, 12) z -> layout channels, units and axis."""
    mv = np.asarray(x_z, dtype=np.float64) * std + mean
    idx = [LEAD_ORDER_12.index(c) for c in layout["channels"]]
    cf = np.transpose(mv[:, :, idx], (0, 2, 1)) / float(layout["scale_to_mV"])
    out = cf if layout["axis"] == "channels_first" else np.transpose(cf, (0, 2, 1))
    return out.astype(np.float32)


def harmonize_real(x_z: np.ndarray, layout: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Pass real ECGs through the same channel restriction + limb rebuild as the synthetic ones,
    so a rebuilt-lead artifact cannot be the detector's cue."""
    return foreign_to_z(z_to_foreign(x_z, layout, mean, std), layout, mean, std)


# ---------------------------------------------------------------- reference statistics


def lead_stats(x_cf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled covariance, correlation and per-channel std of (N, C, T) after per-record demeaning."""
    x = np.asarray(x_cf, dtype=np.float64)
    x = x - x.mean(axis=2, keepdims=True)
    flat = np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)
    cov = flat @ flat.T / flat.shape[1]
    sd = np.sqrt(np.clip(np.diag(cov), 1e-24, None))
    return cov, cov / np.outer(sd, sd), sd


def reference_from_real(x_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    mv = np.asarray(x_z, dtype=np.float64) * std + mean
    _, corr, sd = lead_stats(np.transpose(mv, (0, 2, 1)))
    return {"corr": corr, "std_mV": sd}


# ---------------------------------------------------------------- inference


def infer_axis(shape) -> str:
    if len(shape) != 3:
        raise ValueError(f"signals must be 3-D, got {shape}")
    if shape[2] == N_T and shape[1] != N_T:
        return "channels_first"
    if shape[1] == N_T and shape[2] != N_T:
        return "channels_last"
    raise ValueError(f"cannot tell time axis from shape {shape}")


def _identity_residual(cov: np.ndarray, limb_pos: list[int], names: tuple[str, ...]) -> float:
    """Mean relative residual variance of every limb lead against its (I, II)-solve from two others."""
    sub = {n: p for n, p in zip(names, limb_pos)}
    a, b = names[0], names[1]
    m = np.array([LIMB_COEF[a], LIMB_COEF[b]])
    if abs(np.linalg.det(m)) < 1e-9:
        return np.inf
    inv = np.linalg.inv(m)
    res = []
    for n in names[2:]:
        ci, cii = LIMB_COEF[n]
        w = np.zeros(cov.shape[0])
        coef = np.array([ci, cii]) @ inv          # n = coef[0]*a + coef[1]*b
        w[sub[n]] += 1.0
        w[sub[a]] -= coef[0]
        w[sub[b]] -= coef[1]
        res.append(float(w @ cov @ w) / float(cov[sub[n], sub[n]]))
    return float(np.mean(res))


def infer_channels(x_cf: np.ndarray, ref: dict, max_records: int = 2000) -> dict:
    """Assign lead names to the channels of (N, C, T) by matching against real PTB-XL.

    Limb channels (C - 6 of them) are found as the rank-2 subset when there are >= 3, then named by
    the exact Einthoven/Goldberger identities. Chest channels (and a 2-lead limb pair) are named by
    matching the inter-lead correlation matrix and the per-lead amplitude profile to real data.
    Returns names plus the best/second-best score so a weak (ambiguous) assignment is visible.
    """
    x = np.asarray(x_cf[:max_records], dtype=np.float64)
    c = x.shape[1]
    n_limb = c - 6
    if not 2 <= n_limb <= 6:
        raise ValueError(f"{c} channels: expected V1-V6 plus 2..6 limb leads")
    cov, corr, sd = lead_stats(x)
    ref_corr, ref_sd = ref["corr"], ref["std_mV"]
    li = {n: LEAD_ORDER_12.index(n) for n in LEAD_ORDER_12}
    chest_ref = [li[n] for n in CHEST_LEADS]
    chest_perms = np.array(list(permutations(range(6))))

    def score_all(limb_pos, limb_names):
        """Vectorised score over all 720 chest-name permutations for a fixed limb assignment."""
        chest_pos = [p for p in range(c) if p not in limb_pos]
        ridx = np.empty((len(chest_perms), c), dtype=int)
        for p, nme in zip(limb_pos, limb_names):
            ridx[:, p] = li[nme]
        for slot, p in enumerate(chest_pos):
            ridx[:, p] = np.asarray(chest_ref)[chest_perms[:, slot]]
        d = corr[None] - ref_corr[ridx[:, :, None], ridx[:, None, :]]
        dl = np.log(sd)[None] - np.log(ref_sd[ridx])
        dl = dl - dl.mean(axis=1, keepdims=True)
        scores = (d ** 2).sum(axis=(1, 2)) + (dl ** 2).sum(axis=1)
        return scores, chest_pos

    candidates = []  # (score, names)
    identity = None
    if n_limb >= 3:
        best_sub, best_ratio = None, np.inf
        for sub in combinations(range(c), n_limb):
            ev = np.sort(np.linalg.eigvalsh(cov[np.ix_(sub, sub)]))[::-1]
            ratio = float(ev[2:].sum() / ev.sum())
            if ratio < best_ratio:
                best_sub, best_ratio = list(sub), ratio
        limb_pos = best_sub
        named = sorted(
            ((_identity_residual(cov, limb_pos, perm), perm) for perm in permutations(LIMB_LEADS, n_limb)),
            key=lambda t: t[0],
        )
        identity = {"rank2_residual_share": best_ratio, "best_identity_residual": named[0][0],
                    "second_identity_residual": named[1][0] if len(named) > 1 else None}
        # Keep the few best identity namings and let correlation break near-ties.
        for res, perm in named[:4]:
            scores, chest_pos = score_all(limb_pos, perm)
            for k in np.argsort(scores)[:3]:
                candidates.append((float(scores[k]) + res, limb_pos, perm, chest_pos, chest_perms[k]))
    else:
        for limb_pos in combinations(range(c), n_limb):
            for perm in permutations(LIMB_LEADS, n_limb):
                scores, chest_pos = score_all(list(limb_pos), perm)
                for k in np.argsort(scores)[:2]:
                    candidates.append((float(scores[k]), list(limb_pos), perm, chest_pos, chest_perms[k]))

    candidates.sort(key=lambda t: t[0])

    def to_names(cand):
        _, limb_pos, perm, chest_pos, chest_perm = cand
        names = [None] * c
        for p, nme in zip(limb_pos, perm):
            names[p] = nme
        for slot, p in enumerate(chest_pos):
            names[p] = CHEST_LEADS[chest_perm[slot]]
        return names

    best = to_names(candidates[0])
    second = next((cd for cd in candidates[1:] if to_names(cd) != best), None)
    return {
        "channels": best,
        "score": candidates[0][0],
        "second_score": second[0] if second else None,
        "second_channels": to_names(second) if second else None,
        "margin": (second[0] - candidates[0][0]) if second else None,
        "identity": identity,
    }


def infer_units(x_cf: np.ndarray, channels: list[str], ref_std_mV: np.ndarray) -> dict:
    _, _, sd = lead_stats(np.asarray(x_cf[:2000], dtype=np.float64))
    ratio = float(np.median(sd / ref_std_mV[[LEAD_ORDER_12.index(n) for n in channels]]))
    if 0.4 <= ratio <= 2.5:
        units = "mV"
    elif 400 <= ratio <= 2500:
        units = "uV"
    else:
        units = None
    return {"std_ratio_vs_real_mV": ratio, "units": units,
            "scale_to_mV": UNIT_SCALES[units] if units else None,
            "per_channel_std": sd.tolist()}
