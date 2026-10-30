"""Frozen primary analysis for the pre-registered confirmatory run (report/PREREGISTRATION.md).
Do not edit after the run starts. Reads generator_comparison/experiment/confirmatory/.
Writes confirmatory_result.json / .md next to the results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from generator_comparison.stats import fast_f1

TAG = sys.argv[1] if len(sys.argv) > 1 else "confirmatory"
SEEDS = list(range(100, 130))
ALPHA = 0.05
D = Path("generator_comparison/experiment") / TAG
DATA = Path("processed_data")

res = json.loads((D / "results.json").read_text())
missing = [f"{c}/seed{s}" for c in ("filtered", "random") for s in SEEDS if f"{c}/seed{s}" not in res]
if missing:
    raise SystemExit(f"Incomplete: {len(missing)} runs missing, e.g. {missing[:3]}. The pre-registered analysis needs all 60.")

f = np.array([res[f"filtered/seed{s}"]["test"]["macro_f1"] for s in SEEDS])
r = np.array([res[f"random/seed{s}"]["test"]["macro_f1"] for s in SEEDS])
diff = f - r
p = float(wilcoxon(f, r, alternative="two-sided").pvalue)
dz = float(diff.mean() / diff.std(ddof=1))

# Two-level bootstrap CI of the mean difference (test patients x seeds), 2000 resamples, seed 0.
yte = np.load(DATA / "y_test.npy").astype(np.int64)
pid = pd.read_csv(DATA / "test_metadata.csv")["patient_id"].to_numpy()
pred = {c: {s: np.load(D / f"test_prob_{c}_seed{s}.npy").argmax(1) for s in SEEDS} for c in ("filtered", "random")}
upid = np.unique(pid)
rows = {u: np.where(pid == u)[0] for u in upid}
rng = np.random.default_rng(0)
bd = []
for _ in range(2000):
    idx = np.concatenate([rows[u] for u in rng.choice(upid, len(upid), replace=True)])
    sd = rng.choice(SEEDS, len(SEEDS), replace=True)
    bd.append(np.mean([fast_f1(yte[idx], pred["filtered"][s][idx])[0] - fast_f1(yte[idx], pred["random"][s][idx])[0]
                       for s in sd]))
ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]

if p < ALPHA and diff.mean() > 0:
    decision = "H1 SUPPORTED: filtered > random (p < 0.05)"
elif p < ALPHA:
    decision = "Significant in the OPPOSITE direction: random > filtered (p < 0.05)"
else:
    decision = "NOT SUPPORTED: no significant difference (p >= 0.05)"

fs = np.array([res[f"filtered/seed{s}"]["test"]["f1"]["SBRAD"] for s in SEEDS])
rs = np.array([res[f"random/seed{s}"]["test"]["f1"]["SBRAD"] for s in SEEDS])
out = {
    "primary": {"metric": "test macro-F1", "n_pairs": len(SEEDS), "test": "two-sided Wilcoxon signed-rank",
                "alpha": ALPHA, "p": p, "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "dz": dz,
                "bootstrap_ci95_mean_diff": ci, "filtered_mean": float(f.mean()), "random_mean": float(r.mean()),
                "decision": decision},
    "secondary_descriptive": {"paired_t_p_macro_f1": float(ttest_rel(f, r).pvalue),
                              "sbrad_f1_mean_diff": float((fs - rs).mean()), "sbrad_f1_wins": int((fs > rs).sum())},
}
(D / "confirmatory_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
md = (f"# Confirmatory result (pre-registered)\n\n**{decision}**\n\n"
      f"- test macro-F1: filtered {f.mean():.4f} vs random {r.mean():.4f}; mean diff {diff.mean():+.4f} "
      f"(median {np.median(diff):+.4f}), filtered better in {int((diff > 0).sum())}/{len(SEEDS)} seeds, dz {dz:.2f}\n"
      f"- two-sided Wilcoxon signed-rank p = {p:.4f} (alpha {ALPHA})\n"
      f"- 95 % CI of mean difference (patients x seeds bootstrap): {ci[0]:+.4f} to {ci[1]:+.4f}\n\n"
      f"Secondary (descriptive): paired t p = {out['secondary_descriptive']['paired_t_p_macro_f1']:.4f}; "
      f"SBRAD F1 diff {(fs - rs).mean():+.4f}, filtered better in {int((fs > rs).sum())}/{len(SEEDS)} seeds.\n")
(D / "confirmatory_result.md").write_text(md, encoding="utf-8")
print(md)
