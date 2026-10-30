"""Frozen analysis for part A of report/PREREGISTRATION_3.md (threshold-tuned real-only control).
Do not edit after registration. Reads generator_comparison/experiment/threshold_control/pool<k>/; writes
threshold_result.json / .md next to them. Refuses to run if any of the 80 runs is missing, and refuses a second run.

Conditions per unit (40 units = pools 1-4 x 10 seeds, as study 2): real, real_tuned, random, random_tuned.
`*_tuned` = the same model's test probabilities with the class weights fitted on its own validation probabilities.
  T1 (primary)  random - real_tuned       does augmentation still beat a tuned real-only model?
  T2            real_tuned - real         what tuning alone buys
  T3            random_tuned - real_tuned augmentation when both are tuned
Two-sided Wilcoxon on test macro-F1 (40 paired units); Holm over T1-T3; TOST (paired t, +/-0.01) for T1;
95 % CIs from the three-level bootstrap of study 2 (test patients x pools x seeds), 2000 resamples, seed 0.
share_recovered = mean(real_tuned - real) / mean(random - real).
Decision: "MOSTLY THRESHOLD" if share >= 0.5 and T1 not significant; "AUGMENTATION DOES REAL WORK" if T1 is
significant and positive; otherwise "MIXED".

Usage: py -3.14 _threshold_analysis.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

from _study2_analysis import POOLS, SEEDS
from generator_comparison.eval_metrics import CLASS_ORDER
from generator_comparison.stats import fast_f1, holm
from generator_comparison.threshold_tuning import apply_weights

BASE = ("real", "random")
CONDS = ("real", "real_tuned", "random", "random_tuned")
HYP = {"T1": ("random", "real_tuned"), "T2": ("real_tuned", "real"), "T3": ("random_tuned", "real_tuned")}
PRIMARY = "T1"
ALPHA, SESOI, BOOT, SHARE_RULE = 0.05, 0.010, 2000, 0.5
ROOT = Path(__file__).resolve().parent
D = ROOT / "generator_comparison" / "experiment" / "threshold_control"


def load(d: Path):
    """Per-unit test probabilities for the 4 conditions (tuned ones reweighted), plus the raw results."""
    units = [(k, s) for k in POOLS for s in SEEDS[k]]
    res = {k: json.loads((d / f"pool{k}" / "results.json").read_text()) if (d / f"pool{k}" / "results.json").exists()
           else {} for k in POOLS}
    missing = [f"pool{k}/{c}/seed{s}" for k, s in units for c in BASE if f"{c}/seed{s}" not in res[k]]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} runs missing, e.g. {missing[:3]}. The analysis needs all 80.")
    prob = {c: [] for c in CONDS}
    for k, s in units:
        for c in BASE:
            p = np.load(d / f"pool{k}" / f"test_prob_{c}_seed{s}.npy")
            w = np.array([res[k][f"{c}/seed{s}"]["class_weights_tuned"][cl] for cl in CLASS_ORDER])
            prob[c].append(p)
            prob[f"{c}_tuned"].append(apply_weights(p, w))
    return units, prob, res


def bootstrap(units, pred, y, pid, boot, seed=0):
    """Study 2's three-level bootstrap (patients, then pools, then seeds within each drawn pool), for HYP here."""
    rng = np.random.default_rng(seed)
    upid = np.unique(pid)
    rows = {u: np.where(pid == u)[0] for u in upid}
    pos = {k: [i for i, (kk, _) in enumerate(units) if kk == k] for k in POOLS}
    out = np.zeros((boot, len(HYP)))
    for b in range(boot):
        idx = np.concatenate([rows[u] for u in rng.choice(upid, len(upid), replace=True)])
        draw = np.concatenate([rng.choice(pos[k], len(pos[k]), replace=True) for k in rng.choice(POOLS, len(POOLS))])
        f = {c: {i: fast_f1(y[idx], pred[c][i][idx])[0] for i in set(draw)} for c in CONDS}
        out[b] = [np.mean([f[a][i] - f[r][i] for i in draw]) for a, r in HYP.values()]
    return out


def descriptive(prob, y):
    from sklearn.metrics import average_precision_score, precision_score, recall_score
    out = {}
    for c in CONDS:
        preds = [p.argmax(1) for p in prob[c]]
        out[c] = {
            "sbrad_f1": float(np.mean([fast_f1(y, p)[1] for p in preds])),
            "recall": {cl: float(np.mean([recall_score(y, p, labels=[j], average=None, zero_division=0)[0] for p in preds]))
                       for j, cl in enumerate(CLASS_ORDER)},
            "precision": {cl: float(np.mean([precision_score(y, p, labels=[j], average=None, zero_division=0)[0]
                                             for p in preds])) for j, cl in enumerate(CLASS_ORDER)},
            "auprc": {cl: float(np.mean([average_precision_score(y == j, p[:, j]) for p in prob[c]]))
                      for j, cl in enumerate(CLASS_ORDER)}}
    return out


def analyse(d: Path, y, pid, boot=BOOT):
    units, prob, res = load(d)
    pred = {c: [p.argmax(1) for p in prob[c]] for c in CONDS}
    f1 = {c: np.array([fast_f1(y, p)[0] for p in pred[c]]) for c in CONDS}
    pool_of = np.array([k for k, _ in units])
    rows = {}
    for h, (a, r) in HYP.items():
        diff = f1[a] - f1[r]
        rows[h] = {"comparison": f"{a} - {r}", "n_units": len(diff), "mean_diff": float(diff.mean()),
                   "median_diff": float(np.median(diff)), "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                   "dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan"),
                   "wilcoxon_p": float(wilcoxon(f1[a], f1[r]).pvalue) if np.any(diff != 0) else 1.0,
                   "pool_mean_diffs": {f"pool{k}": float(diff[pool_of == k].mean()) for k in POOLS}}
    for h, p in zip(HYP, holm([rows[h]["wilcoxon_p"] for h in HYP])):
        rows[h].update(adjusted_p=float(p), adjustment="Holm over T1-T3")
    bd = bootstrap(units, pred, y, pid, boot)
    for j, h in enumerate(HYP):
        rows[h]["bootstrap_ci95"] = [float(np.percentile(bd[:, j], 2.5)), float(np.percentile(bd[:, j], 97.5))]
    d1 = f1["random"] - f1["real_tuned"]
    p_lo = float(ttest_1samp(d1, -SESOI, alternative="greater").pvalue)
    p_hi = float(ttest_1samp(d1, SESOI, alternative="less").pvalue)
    rows["T1"]["tost"] = {"bounds": [-SESOI, SESOI], "p_lower": p_lo, "p_upper": p_hi,
                          "equivalent": bool(max(p_lo, p_hi) < ALPHA)}
    gain = float((f1["random"] - f1["real"]).mean())
    share = float((f1["real_tuned"] - f1["real"]).mean() / gain) if gain != 0 else float("nan")
    t1 = rows["T1"]
    t1_sig = t1["adjusted_p"] < ALPHA
    if t1_sig and t1["mean_diff"] > 0:
        decision = "AUGMENTATION DOES REAL WORK: random beats the threshold-tuned real-only model"
    elif share >= SHARE_RULE and not t1_sig:
        decision = "MOSTLY THRESHOLD: tuning the real-only model recovers most of the augmentation gain"
    else:
        decision = "MIXED"
    flat = [r for k in POOLS for r in res[k].values()]
    repro = [r["reproduction"] for r in flat if r.get("reproduction", {}).get("reference")]
    optimism = {c: {key: float(np.mean([res[k][f"{c}/seed{s}"]["val_optimism"][key] for k, s in units]))
                    for key in ("insample_gain", "heldout_gain")} for c in BASE}
    return {"alpha": ALPHA, "sesoi": SESOI, "boot": boot, "share_rule": SHARE_RULE,
            "condition_means": {c: float(f1[c].mean()) for c in CONDS},
            "augmentation_gain_untuned": gain, "share_recovered": share, "decision": decision,
            "hypotheses": rows, "descriptive": descriptive(prob, y), "val_optimism": optimism,
            "reproduction": {"n_compared": len(repro), "n_identical": int(sum(r["identical"] for r in repro)),
                             "max_abs_diff": float(max((r["max_abs_diff"] for r in repro), default=float("nan"))),
                             "min_same_pred_frac": float(min((r["same_pred_frac"] for r in repro), default=float("nan")))}}


def to_md(out):
    L = ["# Threshold-tuned real-only control (pre-registered, part A)", "",
         f"**Decision: {out['decision']}**", "",
         f"Augmentation gain (random - real, untuned): {out['augmentation_gain_untuned']:+.4f}; "
         f"share recovered by tuning real-only: {out['share_recovered']:.2f} (rule: >= {out['share_rule']}).", "",
         "| condition | test macro-F1 (mean of 40) |", "|---|---|"]
    L += [f"| {c} | {m:.4f} |" for c, m in out["condition_means"].items()]
    L += ["", "| | comparison | mean diff | wins | dz | Wilcoxon p | Holm p | 95 % CI |", "|---|---|---|---|---|---|---|---|"]
    for h, v in out["hypotheses"].items():
        ci = v["bootstrap_ci95"]
        L.append(f"| {h} | {v['comparison']} | {v['mean_diff']:+.4f} | {v['wins']}/{v['n_units']} | {v['dz']:.2f} | "
                 f"{v['wilcoxon_p']:.4f} | {v['adjusted_p']:.4f} | {ci[0]:+.4f} to {ci[1]:+.4f} |")
    t = out["hypotheses"]["T1"]["tost"]
    L += ["", f"T1 TOST (±{out['sesoi']}): p_lower {t['p_lower']:.4f}, p_upper {t['p_upper']:.4f}, "
          f"equivalent: {t['equivalent']}.", "", "## Descriptive", "",
          "| condition | SBRAD F1 | recall " + " / ".join(CLASS_ORDER) + " | precision " + " / ".join(CLASS_ORDER)
          + " | AUPRC " + " / ".join(CLASS_ORDER) + " |", "|---|---|---|---|---|"]
    for c, v in out["descriptive"].items():
        L.append(f"| {c} | {v['sbrad_f1']:.3f} | " + " / ".join(f"{x:.2f}" for x in v["recall"].values()) + " | "
                 + " / ".join(f"{x:.2f}" for x in v["precision"].values()) + " | "
                 + " / ".join(f"{x:.3f}" for x in v["auprc"].values()) + " |")
    o, r = out["val_optimism"], out["reproduction"]
    L += ["", "Validation optimism (half-split, macro-F1 gain from tuning): "
          + "; ".join(f"{c} in-sample {v['insample_gain']:+.4f}, held-out {v['heldout_gain']:+.4f}" for c, v in o.items()),
          "", f"Reproduction vs committed study-2 test probabilities: {r['n_identical']}/{r['n_compared']} identical; "
          f"max abs diff {r['max_abs_diff']:.2e}; min same-prediction fraction {r['min_same_pred_frac']:.4f}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import pandas as pd
    if (D / "threshold_result.json").exists():
        raise SystemExit("threshold_result.json exists: the pre-registered analysis runs once. Not overwriting.")
    y = np.load(ROOT / "processed_data" / "y_test.npy").astype(np.int64)
    pid = pd.read_csv(ROOT / "processed_data" / "test_metadata.csv")["patient_id"].to_numpy()
    out = analyse(D, y, pid)
    (D / "threshold_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = to_md(out)
    (D / "threshold_result.md").write_text(md, encoding="utf-8")
    print(md)
