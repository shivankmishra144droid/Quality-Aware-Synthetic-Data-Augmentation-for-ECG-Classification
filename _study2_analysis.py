"""Frozen analysis for study 2 (report/PREREGISTRATION_2.md). Do not edit after registration.
Reads generator_comparison/experiment/study2/pool<k>/; writes study2_result.json / .md next to them.
Refuses to run if any of the 160 runs is missing, and refuses to run a second time.

H1 filtered - random, H2 random - real, H3 random - oversample (test macro-F1, 40 paired units = 4 pools x 10 seeds):
two-sided Wilcoxon signed-rank. H1 is the single primary test at alpha 0.05 (no adjustment); H2 and H3 form one
secondary family, Holm-adjusted, family-wise alpha 0.05. H1 also gets TOST (paired t, +/-0.01).
95 % CIs from a three-level bootstrap: test patients x pools x seeds within pool, 2000 resamples, seed 0.
Sensitivity (descriptive): per-pool mean differences and a one-sample t-test on the 4 pool means.

Usage: py -3.14 _study2_analysis.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

from generator_comparison.stats import fast_f1, holm

POOLS = (1, 2, 3, 4)
SEEDS = {k: list(range(200 + 10 * (k - 1), 210 + 10 * (k - 1))) for k in POOLS}
CONDS = ("real", "oversample", "random", "filtered")
HYP = {"H1": ("filtered", "random"), "H2": ("random", "real"), "H3": ("random", "oversample")}
PRIMARY, SECONDARY = "H1", ("H2", "H3")
ALPHA, SESOI, BOOT = 0.05, 0.010, 2000
ROOT = Path(__file__).resolve().parent
D = ROOT / "generator_comparison" / "experiment" / "study2"


def load(d: Path):
    """Per-unit test macro-F1 {cond: (40,)} in unit order [(pool, seed)], and per-unit test predictions."""
    units = [(k, s) for k in POOLS for s in SEEDS[k]]
    res = {k: json.loads((d / f"pool{k}" / "results.json").read_text()) if (d / f"pool{k}" / "results.json").exists()
           else {} for k in POOLS}
    missing = [f"pool{k}/{c}/seed{s}" for k, s in units for c in CONDS if "test" not in res[k].get(f"{c}/seed{s}", {})]
    if missing:
        raise SystemExit(f"Incomplete: {len(missing)} runs missing, e.g. {missing[:3]}. The analysis needs all 160.")
    f1 = {c: np.array([res[k][f"{c}/seed{s}"]["test"]["macro_f1"] for k, s in units]) for c in CONDS}
    pred = {c: [np.load(d / f"pool{k}" / f"test_prob_{c}_seed{s}.npy").argmax(1) for k, s in units] for c in CONDS}
    return units, f1, pred


def bootstrap(units, pred, y, pid, boot, seed=0):
    """(boot, n_hyp) resampled mean differences: patients, then pools, then seeds within each drawn pool."""
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


def analyse(d: Path, y, pid, boot=BOOT):
    units, f1, pred = load(d)
    pool_of = np.array([k for k, _ in units])
    rows = {}
    for h, (a, r) in HYP.items():
        diff = f1[a] - f1[r]
        rows[h] = {"comparison": f"{a} - {r}", "n_units": len(diff), "mean_diff": float(diff.mean()),
                   "median_diff": float(np.median(diff)), "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                   "dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan"),
                   "wilcoxon_p": float(wilcoxon(f1[a], f1[r]).pvalue) if np.any(diff != 0) else 1.0,
                   "pool_mean_diffs": {f"pool{k}": float(diff[pool_of == k].mean()) for k in POOLS},
                   "pool_level_t_p": float(ttest_1samp([diff[pool_of == k].mean() for k in POOLS], 0).pvalue)}
    rows[PRIMARY].update(adjusted_p=rows[PRIMARY]["wilcoxon_p"], adjustment="none (single primary test)")
    for h, p in zip(SECONDARY, holm([rows[h]["wilcoxon_p"] for h in SECONDARY])):
        rows[h].update(adjusted_p=float(p), adjustment="Holm over H2-H3")
    bd = bootstrap(units, pred, y, pid, boot)
    for j, h in enumerate(HYP):
        rows[h]["bootstrap_ci95"] = [float(np.percentile(bd[:, j], 2.5)), float(np.percentile(bd[:, j], 97.5))]
    d1 = f1["filtered"] - f1["random"]
    p_lo = float(ttest_1samp(d1, -SESOI, alternative="greater").pvalue)
    p_hi = float(ttest_1samp(d1, SESOI, alternative="less").pvalue)
    rows["H1"]["tost"] = {"bounds": [-SESOI, SESOI], "p_lower": p_lo, "p_upper": p_hi,
                          "equivalent": bool(max(p_lo, p_hi) < ALPHA)}
    for h, (a, r) in HYP.items():
        v = rows[h]
        if v["adjusted_p"] < ALPHA:
            v["decision"] = f"SUPPORTED: {a} > {r}" if v["mean_diff"] > 0 else f"SIGNIFICANT, OPPOSITE: {r} > {a}"
        elif h == "H1" and v["tost"]["equivalent"]:
            v["decision"] = f"NOT SUPPORTED; equivalent within +/-{SESOI} (TOST)"
        else:
            v["decision"] = "NOT SUPPORTED; inconclusive (not equivalent within the bounds)" if h == "H1" \
                else "NOT SUPPORTED"
    means = {c: float(f1[c].mean()) for c in CONDS}
    return {"alpha": ALPHA, "sesoi": SESOI, "boot": boot, "condition_means": means, "hypotheses": rows}


def to_md(out):
    L = ["# Study 2 result (pre-registered)", "", "| condition | test macro-F1 (mean of 40) |", "|---|---|"]
    L += [f"| {c} | {m:.4f} |" for c, m in out["condition_means"].items()]
    L += ["", "| | comparison | mean diff | wins | dz | Wilcoxon p | adjusted p | 95 % CI | decision |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h, v in out["hypotheses"].items():
        ci = v["bootstrap_ci95"]
        L.append(f"| {h} | {v['comparison']} | {v['mean_diff']:+.4f} | {v['wins']}/{v['n_units']} | {v['dz']:.2f} | "
                 f"{v['wilcoxon_p']:.4f} | {v['adjusted_p']:.4f} ({v['adjustment']}) | {ci[0]:+.4f} to {ci[1]:+.4f} | "
                 f"**{v['decision']}** |")
    t = out["hypotheses"]["H1"]["tost"]
    L += ["", f"H1 TOST (paired t, bounds ±{out['sesoi']}): p_lower {t['p_lower']:.4f}, p_upper {t['p_upper']:.4f}.", "",
          "Sensitivity (descriptive): per-pool mean differences "
          + "; ".join(f"{h} " + ", ".join(f"{x:+.4f}" for x in v["pool_mean_diffs"].values())
                      + f" (pool-level t p {v['pool_level_t_p']:.3f})" for h, v in out["hypotheses"].items()) + "."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import pandas as pd
    if (D / "study2_result.json").exists():
        raise SystemExit("study2_result.json exists: the pre-registered analysis runs once. Not overwriting.")
    y = np.load(ROOT / "processed_data" / "y_test.npy").astype(np.int64)
    pid = pd.read_csv(ROOT / "processed_data" / "test_metadata.csv")["patient_id"].to_numpy()
    out = analyse(D, y, pid)
    (D / "study2_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = to_md(out)
    (D / "study2_result.md").write_text(md, encoding="utf-8")
    print(md)
