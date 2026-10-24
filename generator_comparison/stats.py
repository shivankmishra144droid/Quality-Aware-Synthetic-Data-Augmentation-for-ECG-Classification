"""Phase 4: statistics for the main experiment (reads experiment/<tag>/).

1. Per-seed paired comparisons on TEST macro-F1 and SBRAD F1 (pairs share the seed):
   Wilcoxon signed-rank (two-sided) and paired t-test, mean difference and Cohen's dz.
   Note: with n seeds, the smallest possible two-sided Wilcoxon p is 2 / 2^n (n=5: 0.0625).
2. Patient-level cluster bootstrap on the test set (2000 resamples of test PATIENTS):
   95% CI of each condition's seed-averaged macro-F1 / SBRAD F1, and of the paired differences.

Usage: py -3.14 -m generator_comparison.stats [--tag main]
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "processed_data"
ALL_CONDS = ("real", "random", "filtered", "filtered_nolabel")


def per_run(res, cond, metric):
    seeds = sorted(int(k.split("seed")[1]) for k in res if k.startswith(cond + "/"))
    get = (lambda r: r["test"]["macro_f1"]) if metric == "macro_f1" else (lambda r: r["test"]["f1"]["SBRAD"])
    return seeds, np.array([get(res[f"{cond}/seed{s}"]) for s in seeds])


def fast_f1(y, pred, n=4):
    cm = np.zeros((n, n))
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return f1.mean(), f1[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = ROOT / "experiment" / a.tag
    res = json.loads((d / "results.json").read_text())
    CONDS = tuple(c for c in ALL_CONDS if any(k.startswith(c + "/") for k in res))
    yte = np.load(DATA / "y_test.npy").astype(np.int64)
    pid = pd.read_csv(DATA / "test_metadata.csv")["patient_id"].to_numpy()
    out = {"tag": a.tag, "summary": {}, "paired": {}, "bootstrap": {}}

    for cond in CONDS:
        out["summary"][cond] = {}
        for m in ("macro_f1", "sbrad_f1"):
            s, v = per_run(res, cond, m)
            out["summary"][cond][m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "n": len(v)}

    for a_, b_ in combinations(CONDS, 2):
        for m in ("macro_f1", "sbrad_f1"):
            sa, va = per_run(res, a_, m)
            sb, vb = per_run(res, b_, m)
            common = sorted(set(sa) & set(sb))
            x = np.array([va[sa.index(s)] for s in common])
            z = np.array([vb[sb.index(s)] for s in common])
            diff = z - x
            sd = diff.std(ddof=1)
            out["paired"][f"{b_} - {a_} [{m}]"] = {
                "n": len(common), "mean_diff": float(diff.mean()),
                "cohen_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
                "wilcoxon_p": float(wilcoxon(z, x).pvalue) if np.any(diff != 0) else 1.0,
                "ttest_p": float(ttest_rel(z, x).pvalue) if sd > 0 else 1.0,
                "wins": int((diff > 0).sum())}

    # Cluster bootstrap over test patients.
    preds = {c: [np.load(d / f"test_prob_{c}_seed{s}.npy").argmax(1) for s in per_run(res, c, "macro_f1")[0]]
             for c in CONDS}
    upid = np.unique(pid)
    rows = {p: np.where(pid == p)[0] for p in upid}
    rng = np.random.default_rng(0)
    boot = {c: [] for c in CONDS}
    for _ in range(a.boot):
        idx = np.concatenate([rows[p] for p in rng.choice(upid, len(upid), replace=True)])
        for c in CONDS:
            vals = np.array([fast_f1(yte[idx], p[idx]) for p in preds[c]])
            boot[c].append(vals.mean(0))
    boot = {c: np.array(v) for c, v in boot.items()}
    for c in CONDS:
        out["bootstrap"][c] = {m: [float(np.percentile(boot[c][:, j], q)) for q in (2.5, 50, 97.5)]
                               for j, m in enumerate(("macro_f1", "sbrad_f1"))}
    for a_, b_ in combinations(CONDS, 2):
        dd = boot[b_] - boot[a_]
        out["bootstrap"][f"{b_} - {a_}"] = {m: [float(np.percentile(dd[:, j], q)) for q in (2.5, 50, 97.5)]
                                           + [float((dd[:, j] > 0).mean())]
                                           for j, m in enumerate(("macro_f1", "sbrad_f1"))}
    (d / "stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["| condition | test macro-F1 (mean ± sd over seeds) | 95% CI (patient bootstrap) | test SBRAD F1 | 95% CI |",
         "|---|---|---|---|---|"]
    for c in CONDS:
        s, b = out["summary"][c], out["bootstrap"][c]
        L.append(f"| {c} | {s['macro_f1']['mean']:.3f} ± {s['macro_f1']['std']:.3f} | "
                 f"{b['macro_f1'][0]:.3f}–{b['macro_f1'][2]:.3f} | {s['sbrad_f1']['mean']:.3f} ± "
                 f"{s['sbrad_f1']['std']:.3f} | {b['sbrad_f1'][0]:.3f}–{b['sbrad_f1'][2]:.3f} |")
    L += ["", "| comparison | metric | mean diff | dz | Wilcoxon p | paired t p | wins | bootstrap 95% CI of diff |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in out["paired"].items():
        comp, m = k.split(" [")
        m = m.rstrip("]")
        ci = out["bootstrap"][comp][m]
        L.append(f"| {comp} | {m} | {v['mean_diff']:+.4f} | {v['cohen_dz']:.2f} | {v['wilcoxon_p']:.4f} | "
                 f"{v['ttest_p']:.4f} | {v['wins']}/{v['n']} | {ci[0]:+.4f} to {ci[2]:+.4f} |")
    table = "\n".join(L)
    (d / "stats.md").write_text(f"# Statistics ({a.tag})\n\n{table}\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
