"""Phase 2 checks for the study-2 pools (report/WORKFLOW_2.md). Train-split models only; no val/test data.

Per pool:
  1. pick('filtered') is the true per-class top-k: every selected score >= every unselected score, and overlap 1.00
     with a stable-sort top-k (ties at the cut-off broken by pool order, as select() does)
  2. realism top-third share by CV fold: max - min < 0.10
  3. replay: re-assembling every ECG from the stored template and the pool seed reproduces X exactly, and gives
     the R-peak positions actually placed; true rate out of class (STACH <= 100 bpm, SBRAD >= 60 bpm) <= 2 %
  4. sanity: no NaN / inf; share of ECGs with |z| > 20
Writes generator_comparison/pools2/checks.json and prints a table.

Usage: py -3.14 _check_pools2.py"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from generator_comparison.augment_experiment import pick
from synthetic_generation.assembly import MEAN_RR_BOUNDS, assemble

D = Path("generator_comparison/pools2")
COUNTS = {1: 1056, 2: 568, 3: 446}
OUT_OF_CLASS = {2: lambda hr: hr <= 100, 3: lambda hr: hr >= 60}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(k, models):
    pool, s = np.load(D / f"pool_{k}.npz"), dict(np.load(D / f"scores_{k}.npz"))
    X, y, T = pool["X"], pool["y"], pool["templates"]
    out = {}
    sel = pick("filtered", y, s, COUNTS, 0)
    out["topk_overlap"], out["topk_is_top"] = {}, {}
    for c, n in COUNTS.items():
        ic = np.where(y == c)[0]
        # Scores are means of ranks, so ties at the cut-off happen; select() breaks them by pool order (stable sort).
        true = set(ic[np.argsort(-s["score"][ic], kind="stable")[:n]])
        got = sel[y[sel] == c]
        rest = np.setdiff1d(ic, got)
        out["topk_overlap"][c] = len(true & set(got)) / n
        out["topk_is_top"][c] = bool(len(got) == n and s["score"][got].min() >= s["score"][rest].max())
    spread = {}
    for c in COUNTS:
        m = y == c
        r, f = s["realism"][m], s["realism_fold"][m]
        thr = np.quantile(r, 2 / 3)
        sh = [float((r[f == j] > thr).mean()) for j in np.unique(f)]
        spread[c] = max(sh) - min(sh)
    out["fold_share_spread"] = spread
    rng = np.random.default_rng(10 * k + 1)                   # _beat_gen_pool --seed used for pool k (11/21/31/41)
    max_err, true_hr = 0.0, {c: [] for c in COUNTS}
    for i in range(len(y)):
        c = int(y[i])
        state = rng.bit_generator.state
        r = models[c]["rhythm"].sample_r_positions(rng)
        rng.bit_generator.state = state
        x = assemble(T[i].astype(np.float64), models[c], rng)
        max_err = max(max_err, float(np.abs(x - X[i]).max()))
        true_hr[c].append(60 * 100 / np.diff(r).mean())
    out["replay_max_abs_err"] = max_err
    out["true_out_of_class"] = {c: float(np.mean(OUT_OF_CLASS[c](np.array(true_hr[c])))) for c in OUT_OF_CLASS}
    out["true_median_hr"] = {c: float(np.median(true_hr[c])) for c in COUNTS}
    out["nonfinite"] = int((~np.isfinite(X)).any(axis=(1, 2)).sum())
    out["abs_z_gt_20_frac"] = float((np.abs(X).max(axis=(1, 2)) > 20).mean())
    out["counts"] = np.bincount(y, minlength=4).tolist()
    out["sha256"] = {f"pool_{k}.npz": sha256(D / f"pool_{k}.npz"), f"scores_{k}.npz": sha256(D / f"scores_{k}.npz")}
    out["pass"] = bool(all(v == 1.0 for v in out["topk_overlap"].values())
                       and all(out["topk_is_top"].values())
                       and all(v < 0.10 for v in spread.values())
                       and max_err < 1e-4
                       and all(v <= 0.02 for v in out["true_out_of_class"].values())
                       and out["nonfinite"] == 0)
    return out


if __name__ == "__main__":
    models = pickle.load(open("generator_comparison/beat_templates/assembly_models.pkl", "rb"))
    for c, m in models.items():
        m["rhythm"].mean_rr_bounds = MEAN_RR_BOUNDS[c]         # as _beat_gen_pool does by default
    res = {f"pool_{k}": check(k, models) for k in (1, 2, 3, 4)}
    res["inputs_sha256"] = {
        "assembly_models.pkl": sha256("generator_comparison/beat_templates/assembly_models.pkl"),
        "tpl_8x64_last.pt": sha256("generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt"),
        "cnn_lstm_baseline.pt": sha256("baseline/cnn_lstm_baseline.pt")}
    (D / "checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("| pool | top-k overlap A/S/B | fold spread A/S/B | replay err | true out-of-class S/B | |z|>20 | non-finite | pass |")
    print("|---|---|---|---|---|---|---|---|")
    for k in (1, 2, 3, 4):
        v = res[f"pool_{k}"]
        print(f"| {k} | " + " / ".join(f"{x:.4f}" for x in v["topk_overlap"].values())
              + " | " + " / ".join(f"{x:.2f}" for x in v["fold_share_spread"].values())
              + f" | {v['replay_max_abs_err']:.1e} | "
              + " / ".join(f"{x:.3f}" for x in v["true_out_of_class"].values())
              + f" | {v['abs_z_gt_20_frac']:.4f} | {v['nonfinite']} | {'PASS' if v['pass'] else 'FAIL'} |")
