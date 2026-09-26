"""Re-score every existing synthetic pool with the fixed HR metric (lead_ii_hr_pt).
CPU only. Writes generator_comparison/hr_rescore/. Does not touch any pool or old metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr, lead_ii_hr_pt
from synthetic_generation.config import DATA_DIR
from synthetic_generation.data import load_train

ROOT = Path("generator_comparison")
OUT = ROOT / "hr_rescore"
OUT.mkdir(parents=True, exist_ok=True)

POOLS = {
    "week4_pool": Path("synthetic_data"),
    "v1_sssd": ROOT / "sssd",
    "v2_16000": ROOT / "sssd_v2",
    "ckpt2000": ROOT / "sssd_ckpt2000",
    "v3_16x128": ROOT / "sssd_v3",
    "cfg_w2": ROOT / "sssd_cfg_w2",
    "cfg_w4": ROOT / "sssd_cfg_w4",
    "balanced": ROOT / "sssd_balanced",
}


def per_class(x, y):
    return {name: lead_ii_hr_pt(x[y == i]) for i, name in enumerate(CLASS_ORDER) if (y == i).any()}


X_train, y_train = load_train(DATA_DIR)

# Acceptance gate on ALL real train ECGs.
real = per_class(X_train, y_train)
real_old = {name: lead_ii_hr(X_train[y_train == i][:256])["median_hr"] for i, name in enumerate(CLASS_ORDER)}
gate = {
    "SBRAD_median_lt_60": real["SBRAD"]["median_hr_peaks"] < 60,
    "STACH_median_gt_100": real["STACH"]["median_hr_peaks"] > 100,
}
print("REAL (new)", {k: round(v["median_hr_peaks"], 1) for k, v in real.items()})
print("REAL periodicity / agree", {k: (round(v["median_periodicity"], 2), round(v["agree_frac"], 2)) for k, v in real.items()})
print("REAL (old detector, 256/class)", {k: round(v, 1) for k, v in real_old.items()})
print("GATE", gate)
if not all(gate.values()):
    raise SystemExit("Acceptance gate failed; not re-scoring pools.")

pools = {}
for tag, d in POOLS.items():
    if not (d / "X_synthetic.npy").exists():
        print("skip", tag, "(no pool)")
        continue
    x = np.load(d / "X_synthetic.npy")
    y = np.load(d / "y_synthetic.npy")
    pc = per_class(x, y)
    hr = {k: v["median_hr_peaks"] for k, v in pc.items()}
    pools[tag] = {
        "path": str(d),
        "hr_by_class": pc,
        "stach_minus_sbrad": hr["STACH"] - hr["SBRAD"],
        "real_direction_ok": hr["STACH"] > hr["SR"] > hr["SBRAD"],
        "hr_mae_vs_real": float(np.mean([abs(hr[k] - real[k]["median_hr_peaks"]) for k in hr])),
        "median_periodicity": float(np.median([v["median_periodicity"] for v in pc.values()])),
        "mean_agree_frac": float(np.mean([v["agree_frac"] for v in pc.values()])),
    }
    print(f"{tag:12s}", {k: round(v, 1) for k, v in hr.items()},
          "STACH-SBRAD", round(pools[tag]["stach_minus_sbrad"], 1),
          "order_ok", pools[tag]["real_direction_ok"], "MAE", round(pools[tag]["hr_mae_vs_real"], 1),
          "period", round(pools[tag]["median_periodicity"], 2), "agree", round(pools[tag]["mean_agree_frac"], 2))

# Figure: Lead-II + QRS envelope, real vs v1 vs cfg_w4, SBRAD and STACH.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generator_comparison.eval_metrics import qrs_envelope

t = np.arange(1000) / 100
rows = [("real", X_train, y_train)] + [
    (tag, np.load(POOLS[tag] / "X_synthetic.npy"), np.load(POOLS[tag] / "y_synthetic.npy"))
    for tag in ("v1_sssd", "cfg_w4") if tag in pools]
fig, axes = plt.subplots(len(rows), 2, figsize=(14, 2.4 * len(rows)), sharex=True)
for r, (tag, x, y) in enumerate(rows):
    for c, cls in enumerate(("SBRAD", "STACH")):
        sig = x[y == CLASS_ORDER.index(cls)][0][:, 1].astype(np.float64)
        env = qrs_envelope(sig)
        ax = axes[r, c]
        ax.plot(t, (sig - sig.mean()) / (sig.std() + 1e-8), lw=0.7, color="k")
        ax.plot(t, env / (env.max() + 1e-12) * 3 - 4, lw=0.8, color="tab:red")
        ax.set_title(f"{tag} {cls}  (black: Lead II z, red: QRS envelope)", fontsize=9)
        ax.set_yticks([])
axes[-1, 0].set_xlabel("s")
axes[-1, 1].set_xlabel("s")
fig.tight_layout()
fig.savefig(OUT / "lead_ii_envelope_real_vs_syn.png", dpi=110)

payload = {
    "metric": "lead_ii_hr_pt: 5-15 Hz bandpass, derivative^2, 150 ms MWI, find_peaks(0.3*p98, 0.27 s); ACF cross-check",
    "real_train_new": real,
    "real_train_old_detector": real_old,
    "acceptance_gate": gate,
    "real_stach_minus_sbrad": real["STACH"]["median_hr_peaks"] - real["SBRAD"]["median_hr_peaks"],
    "pools": pools,
}
(OUT / "hr_rescore.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("wrote", OUT / "hr_rescore.json")
