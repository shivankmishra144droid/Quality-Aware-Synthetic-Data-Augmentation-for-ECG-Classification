"""Part B of report/PREREGISTRATION_3.md: score every real-noise-bank variant against the registered gate.
Train and validation data only; the test split is never loaded.

Per variant (generator_comparison/noise_bank/<variant>.npz):
  detector    generator_comparison.realism_detector.run, seeds 0 and 1, donors excluded from the real side
  scorecard   generator_comparison.scorecard.score_pool / checks against a real_ref built like scorecard.main
              (rng 42, 256/class reference + 32/class real_ref), with every donor kept out of both
  novelty     donor_corr (each sample vs its own donor): median and share above 0.98
  TSTR        generator_comparison.tstr.train_eval at N_TSTR/class, seeds 0-2, on the real VAL split, with the
              same N for the reference rows TRTR and beat_gen (tstr/pool_beatgen.npz)
Gate (all must hold): detector auc_mean <= 0.70; every scorecard check passes; median donor_corr <= 0.95 and
share > 0.98 <= 1 %; TSTR balanced accuracy >= beat_gen's at the same N (0.68 at N=440 if beat_gen is missing).

Output: generator_comparison/noise_bank/results.{json,md}, realism_novelty.png
Usage:  py -3.14 _noise_bank_eval.py [--variants N0,N1,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generator_comparison import realism_detector as RD
from generator_comparison import scorecard as SC
from generator_comparison import tstr as TS
from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed_data"
OUT = ROOT / "generator_comparison" / "noise_bank"
BEAT_GEN = ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz"
N_TSTR = 200                       # SBRAD donors are capped at 223 (half of 446), so 440/class is impossible
GATE = {"detector_max": 0.70, "donor_corr_median_max": 0.95, "donor_corr_hi": 0.98, "donor_corr_hi_share_max": 0.01,
        "tstr_fallback": 0.68}


def reference(X, y, exclude):
    """scorecard.main's reference construction, with `exclude` (all donors) kept out of both sets."""
    rng = np.random.default_rng(42)
    ref_idx, pool_idx = [], []
    for c in range(4):
        idx = rng.permutation(np.setdiff1d(np.where(y == c)[0], exclude))
        ref_idx.append(idx[:SC.N_REF])
        pool_idx.append(idx[SC.N_REF:SC.N_REF + SC.N_POOL])
    ref_idx, pool_idx = np.sort(np.concatenate(ref_idx)), np.sort(np.concatenate(pool_idx))
    x_ref, y_ref = np.asarray(X[ref_idx]), y[ref_idx]
    ref = (x_ref, y_ref, [SC.log_psd(x_ref[y_ref == c]) for c in range(4)],
           {CLASS_ORDER[c]: lead_ii_hr_pt(x_ref[y_ref == c])["median_hr_peaks"] for c in range(4)})
    real_std_lead = x_ref.std(axis=(0, 1))
    real_ref = SC.score_pool(np.asarray(X[pool_idx]), y[pool_idx], ref, real_std_lead, X, y, exclude=pool_idx)
    return ref, real_std_lead, real_ref


def tstr(Xs, ys, Xva, yva, n):
    runs = []
    for s in TS.SEEDS:
        Xt, yt = TS.balanced_subset(Xs, ys, n, np.random.default_rng(100 + s))
        runs.append(TS.train_eval(Xt, yt, Xva, yva, s))
    return {"n_per_class": n, "balanced_acc_mean": float(np.mean([r["balanced_acc"] for r in runs])),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs]))}


def gate(row, tstr_bar):
    g = {"detector": row["detector"]["auc_mean"] <= GATE["detector_max"],
         "scorecard": all(row["scorecard"]["checks"].values()),
         "novelty": row["novelty"]["median"] <= GATE["donor_corr_median_max"]
         and row["novelty"]["share_gt_0.98"] <= GATE["donor_corr_hi_share_max"],
         "tstr": row["tstr"]["balanced_acc_mean"] >= tstr_bar}
    return g, all(g.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None, help="comma list; default: every npz in the output folder")
    a = ap.parse_args(argv)
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    y = np.load(DATA / "y_train.npy")
    Xva, yva = np.load(DATA / "X_val.npy").astype(np.float32), np.load(DATA / "y_val.npy")
    names = a.variants.split(",") if a.variants else sorted(p.stem for p in OUT.glob("*.npz"))
    data = {v: np.load(OUT / f"{v}.npz") for v in names}
    all_src = np.unique(np.concatenate([d["src_idx"] for d in data.values()]))
    ref, real_std_lead, real_ref = reference(X, y, all_src)
    n_tstr = min([N_TSTR] + [int((d["y"] == c).sum()) for d in data.values() for c in range(4)])

    refs = {"TRTR": tstr(X, y, Xva, yva, n_tstr)}
    if BEAT_GEN.exists():
        b = np.load(BEAT_GEN)
        refs["beat_gen"] = tstr(b["X"], b["y"], Xva, yva, n_tstr)
    tstr_bar = refs["beat_gen"]["balanced_acc_mean"] if "beat_gen" in refs else GATE["tstr_fallback"]
    print("TSTR references", json.dumps(refs), flush=True)

    rows = {}
    for v, d in data.items():
        Xs, ys, src = d["X"], d["y"], d["src_idx"]
        det = {f"seed{s}": RD.run(Xs, ys, seed=s, exclude=src) for s in (0, 1)}
        det["auc_mean"] = float(np.mean([r["auc"] for r in det.values()]))
        sc = SC.score_pool(Xs, ys, ref, real_std_lead, X, y, exclude=src)
        sc["checks"] = SC.checks(sc, real_ref)
        dc = d["donor_corr"]
        row = {"n": int(len(ys)), "detector": det, "scorecard": sc,
               "novelty": {"median": float(np.median(dc)), "p90": float(np.percentile(dc, 90)),
                           "share_gt_0.98": float((dc > GATE["donor_corr_hi"]).mean())},
               "tstr": tstr(Xs, ys, Xva, yva, n_tstr)}
        row["gate"], row["pass"] = gate(row, tstr_bar)
        rows[v] = row
        print(f"{v}: AUC {det['auc_mean']:.3f} | median donor corr {row['novelty']['median']:.3f} | "
              f"TSTR {row['tstr']['balanced_acc_mean']:.3f} | gate {row['gate']} -> {'PASS' if row['pass'] else 'fail'}",
              flush=True)
    out = {"gate": GATE, "tstr_bar": tstr_bar, "n_tstr": n_tstr, "tstr_references": refs,
           "real_ref_checks": SC.checks(real_ref, real_ref), "variants": rows,
           "passing": [v for v, r in rows.items() if r["pass"]]}
    (OUT / "results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (OUT / "results.md").write_text(to_md(out), encoding="utf-8")
    plot(out)
    print(to_md(out))


def to_md(out):
    L = ["# Real-noise-bank variants (pre-registered, part B)", "",
         f"Gate: detector <= {GATE['detector_max']}, all scorecard checks, median donor corr <= "
         f"{GATE['donor_corr_median_max']} and share > {GATE['donor_corr_hi']} <= {GATE['donor_corr_hi_share_max']}, "
         f"TSTR balanced acc >= {out['tstr_bar']:.3f} (N={out['n_tstr']}/class).", "",
         "TSTR references: " + ", ".join(f"{k} {v['balanced_acc_mean']:.3f}" for k, v in out["tstr_references"].items()),
         "", "| variant | n | detector AUC | scorecard | median donor corr | share >0.98 | TSTR bal. acc | pass |",
         "|---|---|---|---|---|---|---|---|"]
    for v, r in out["variants"].items():
        sc = r["scorecard"]["checks"]
        L.append(f"| {v} | {r['n']} | {r['detector']['auc_mean']:.3f} | {sum(sc.values())}/{len(sc)} | "
                 f"{r['novelty']['median']:.3f} | {r['novelty']['share_gt_0.98']:.3f} | "
                 f"{r['tstr']['balanced_acc_mean']:.3f} | {'**PASS**' if r['pass'] else 'fail'} |")
    L += ["", f"Passing: {', '.join(out['passing']) or 'none'}."]
    return "\n".join(L) + "\n"


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for v, r in out["variants"].items():
        ax.scatter(r["novelty"]["median"], r["detector"]["auc_mean"], c="tab:green" if r["pass"] else "tab:red")
        ax.annotate(v, (r["novelty"]["median"], r["detector"]["auc_mean"]), fontsize=7)
    ax.axhline(GATE["detector_max"], ls="--", c="gray")
    ax.axvline(GATE["donor_corr_median_max"], ls=":", c="gray")
    ax.set_xlabel("median correlation to own donor (higher = less novel)")
    ax.set_ylabel("detector AUC (lower = more realistic)")
    fig.tight_layout()
    fig.savefig(OUT / "realism_novelty.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
