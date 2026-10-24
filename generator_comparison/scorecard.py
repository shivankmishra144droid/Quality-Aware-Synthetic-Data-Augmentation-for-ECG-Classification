"""Generator scorecard: one pass/fail table for every synthetic pool vs real train.

Generator-quality gate (is this pool good enough to be worth filtering?). Not the
Week 5 per-sample quality filter. A held-out real subset ("real_ref", 32/class,
disjoint from the reference set) is scored the same way and must pass every check;
if it does not, the threshold is wrong, not the generator.

Run from repo root:  py -3.14 -m generator_comparison.scorecard
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

from generator_comparison.eval_metrics import CLASS_ORDER, LEAD_ORDER_12, lead_ii_hr_pt, pairwise_corr
from synthetic_generation.leads import reconstruct_12, to_independent_8
from synthetic_generation.sample import basic_sanity

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "processed_data"
OUT = ROOT / "scorecard"
FS = 100
N_REF = 256  # real reference ECGs per class
N_POOL = 32  # size of the real_ref pool per class (matches synthetic pools)

POOLS = {
    "week4_pool": ROOT.parent / "synthetic_data",
    "v1_sssd": ROOT / "sssd",
    "v2_16000": ROOT / "sssd_v2",
    "ckpt2000": ROOT / "sssd_ckpt2000",
    "v3_16x128": ROOT / "sssd_v3",
    "cfg_w2": ROOT / "sssd_cfg_w2",
    "cfg_w4": ROOT / "sssd_cfg_w4",
    "balanced": ROOT / "sssd_balanced",
    "ema_raw": ROOT / "sssd_ema_raw",
    "ema": ROOT / "sssd_ema",
    "sched_raw": ROOT / "sssd_sched_raw",
    "sched_ema": ROOT / "sssd_sched",
    "budget_raw": ROOT / "sssd_budget_raw",
    "budget_ema": ROOT / "sssd_budget",
    "vpred_raw": ROOT / "sssd_vpred_raw",
    "vpred_ema": ROOT / "sssd_vpred",
    "vpred5k_raw": ROOT / "sssd_vpred5k_raw",
    "vpred5k_ema": ROOT / "sssd_vpred5k",
    "vpred16_raw": ROOT / "sssd_vpred16_raw",
    "vpred16_ema": ROOT / "sssd_vpred16",
    "beat_oracle": ROOT / "beat_oracle",
    "beat_gen": ROOT / "beat_gen",
}

# Pass bar. Each check is a (name, description) pair; values are computed in score_pool.
THRESH = {
    "std_lo": 0.8, "std_hi": 1.2,           # global amplitude in z-space (real ~1.02)
    "lead_std_logratio_max": 0.2,           # mean |log(syn lead std / real lead std)| ~ within 20%
    "periodicity_min": 0.5,                 # median ACF periodicity over SR/STACH/SBRAD (real ~0.8)
    "agree_min": 0.6,                       # peak-vs-ACF HR agreement over SR/STACH/SBRAD (real ~0.9)
    "hr_mae_max": 15.0,                     # bpm, per-class median HR vs real
    "psd_factor": 2.0,                      # PSD distance <= factor x real_ref's own distance
    "diversity_margin": 0.15,               # mean pairwise corr <= real_ref's + margin (every class)
    "memorized_max": 0.0,                   # fraction of samples with corr > 0.95 to a real train ECG
    "extreme_frac_max": 0.02,               # |z| > 20 samples; real train ECGs: ~1%
}


def log_psd(x12: np.ndarray) -> np.ndarray:
    """Mean log10 Welch PSD per lead over 0.5-40 Hz, each sample/lead normalized to unit
    power so this measures spectral SHAPE (amplitude has its own checks). Returns (12, F)."""
    f, p = welch(x12, fs=FS, nperseg=256, axis=1)  # (N, F, 12)
    keep = (f >= 0.5) & (f <= 40)
    p = p[:, keep, :]
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return np.log10(p.mean(axis=0) + 1e-12).T


def psd_distance(x12, y, ref_psd) -> float:
    """Mean |log-PSD difference| in dB, averaged over classes, leads, frequencies."""
    return float(np.mean([10 * np.abs(log_psd(x12[y == i]) - ref_psd[i]).mean()
                          for i in range(4) if (y == i).any()]))


def _zrows(a):
    a = a.reshape(len(a), -1).astype(np.float32)
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)


def memorized_frac(x12, y, X_all, y_all, exclude=None, thresh=0.95, chunk=2048) -> float:
    """Fraction of samples whose max corr to ANY same-class train ECG exceeds thresh."""
    fr = []
    for i in range(4):
        s = _zrows(x12[y == i])
        idx = np.where(y_all == i)[0]
        if exclude is not None:
            idx = np.setdiff1d(idx, exclude)
        best = np.full(len(s), -1.0, np.float32)
        for c in range(0, len(idx), chunk):
            r = _zrows(np.asarray(X_all[idx[c:c + chunk]]))
            best = np.maximum(best, (s @ r.T / s.shape[1]).max(axis=1))
        fr.append((best > thresh).mean())
    return float(np.mean(fr))


def score_pool(x12, y, ref, real_std_lead, X_all, y_all, exclude=None):
    x_ref, y_ref, ref_psd, ref_hr = ref
    hr = {CLASS_ORDER[i]: lead_ii_hr_pt(x12[y == i]) for i in range(4)}
    rhythmic = ("SR", "STACH", "SBRAD")  # AFIB is irregular by definition
    med = {k: v["median_hr_peaks"] for k, v in hr.items()}
    lead_std = x12.std(axis=(0, 1))
    return {
        # NaN / flatline must be zero; |z| > 20 also occurs in ~1% of REAL train ECGs (17 of 1760,
        # 440/class), so extreme amplitude is allowed up to extreme_frac_max (recalibrated 2026-10-25
        # after real ECGs were found to fail the old zero-tolerance rule).
        "invalid": (lambda s: s["nan_inf"] + s["flatline"] + s["bad_shape"])(basic_sanity(x12)),
        "extreme_frac": basic_sanity(x12)["extreme_amp"] / len(x12),
        "std": float(x12.std()),
        "lead_std_logratio": float(np.mean(np.abs(np.log(lead_std / real_std_lead)))),
        "periodicity": float(np.median([hr[k]["median_periodicity"] for k in rhythmic])),
        "agree": float(np.mean([hr[k]["agree_frac"] for k in rhythmic])),
        "hr": {k: round(v, 1) for k, v in med.items()},
        "hr_order_ok": bool(med["STACH"] > med["SR"] > med["SBRAD"]),
        "hr_mae": float(np.mean([abs(med[k] - ref_hr[k]) for k in CLASS_ORDER])),
        "psd_db": psd_distance(x12, y, ref_psd),
        "diversity": {CLASS_ORDER[i]: pairwise_corr(x12[y == i])["mean"] for i in range(4)},
        "memorized": memorized_frac(x12, y, X_all, y_all, exclude),
    }


def checks(s, real_ref):
    t = THRESH
    beat = s["periodicity"] >= t["periodicity_min"] and s["agree"] >= t["agree_min"]
    return {
        "valid": s["invalid"] == 0 and s["extreme_frac"] <= t["extreme_frac_max"],
        "amplitude": t["std_lo"] <= s["std"] <= t["std_hi"],
        "per_lead_amp": s["lead_std_logratio"] <= t["lead_std_logratio_max"],
        "heartbeat": beat,
        # HR on signals without a heartbeat is noise, so HR checks require the heartbeat check.
        "hr_order": beat and s["hr_order_ok"],
        "hr_mae": beat and s["hr_mae"] <= t["hr_mae_max"],
        "spectrum": s["psd_db"] <= t["psd_factor"] * real_ref["psd_db"],
        "diversity": all(s["diversity"][k] <= real_ref["diversity"][k] + t["diversity_margin"] for k in CLASS_ORDER),
        "not_memorized": s["memorized"] <= t["memorized_max"],
    }


def main():
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    y = np.load(DATA / "y_train.npy")
    rng = np.random.default_rng(42)
    ref_idx, pool_idx = [], []
    for i in range(4):
        idx = rng.permutation(np.where(y == i)[0])
        ref_idx.append(idx[:N_REF])
        pool_idx.append(idx[N_REF:N_REF + N_POOL])
    ref_idx, pool_idx = np.sort(np.concatenate(ref_idx)), np.sort(np.concatenate(pool_idx))
    x_ref, y_ref = np.asarray(X[ref_idx]), y[ref_idx]
    ref_psd = [log_psd(x_ref[y_ref == i]) for i in range(4)]
    ref_hr = {CLASS_ORDER[i]: lead_ii_hr_pt(x_ref[y_ref == i])["median_hr_peaks"] for i in range(4)}
    real_std_lead = x_ref.std(axis=(0, 1))
    ref = (x_ref, y_ref, ref_psd, ref_hr)

    x_rp, y_rp = np.asarray(X[pool_idx]), y[pool_idx]
    real_ref = score_pool(x_rp, y_rp, ref, real_std_lead, X, y, exclude=pool_idx)

    rows = {"real_ref": real_ref}
    for tag, d in POOLS.items():
        if (d / "X_synthetic.npy").exists():
            # Re-derive III/aVR/aVL/aVF with the fixed mV-space reconstruction (I, II, V1-V6 are
            # stored unchanged), so pools made before/after the leads.py fix are comparable.
            xs = reconstruct_12(to_independent_8(np.load(d / "X_synthetic.npy")))
            rows[tag] = score_pool(xs, np.load(d / "y_synthetic.npy"), ref, real_std_lead, X, y)
    result = {tag: {**s, "checks": checks(s, real_ref)} for tag, s in rows.items()}
    for s in result.values():
        s["passed"] = sum(s["checks"].values())
        s["n_checks"] = len(s["checks"])

    OUT.mkdir(exist_ok=True)
    (OUT / "scorecard.json").write_text(json.dumps({"thresholds": THRESH, "n_ref_per_class": N_REF,
                                                    "pools": result}, indent=2), encoding="utf-8")
    names = list(result["real_ref"]["checks"])
    lines = ["| pool | " + " | ".join(names) + " | std | period | HR SR/AF/ST/SB | PSD dB | score |",
             "|" + "---|" * (len(names) + 5)]
    for tag, s in result.items():
        mark = " | ".join("✅" if s["checks"][n] else "❌" for n in names)
        h = s["hr"]
        lines.append(f"| {tag} | {mark} | {s['std']:.2f} | {s['periodicity']:.2f} | "
                     f"{h['SR']:.0f}/{h['AFIB']:.0f}/{h['STACH']:.0f}/{h['SBRAD']:.0f} | "
                     f"{s['psd_db']:.1f} | {s['passed']}/{s['n_checks']} |")
    table = "\n".join(lines)
    (OUT / "scorecard.md").write_text("# Generator scorecard\n\n" + table + "\n\nThresholds: `"
                                      + json.dumps(THRESH) + "`\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(table)
    bad = [n for n, ok in result["real_ref"]["checks"].items() if not ok]
    if bad:
        print("WARNING: real_ref fails", bad, "- thresholds need recalibration")


if __name__ == "__main__":
    main()
