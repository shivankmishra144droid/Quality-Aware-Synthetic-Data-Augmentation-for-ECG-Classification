"""Report figures (Phase 5). Writes report/figures/*.png.
  fig1_templates.png    real vs generated median beat templates (Lead II), per class
  fig2_ecgs.png         real vs generated 10 s Lead-II ECGs, per class
  fig3_tstr.png         TSTR balanced accuracy by training source
  fig4_experiment.png   main experiment: test macro-F1 and SBRAD F1 per condition (if results exist)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator_comparison.eval_metrics import CLASS_ORDER

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
GC = Path("generator_comparison")
FS = 100
rng = np.random.default_rng(3)

# Fig 1: templates (Lead II = independent-lead index 1)
rt = np.load(GC / "beat_templates/templates_train.npz")
gp = np.load(GC / "final_pool/pool.npz")
tg = np.load(GC / "tstr/pool_beatgen.npz")
t = (np.arange(70) - 25) / FS * 1000
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for c, ax in enumerate(axes):
    real = rt["templates"][(rt["y"] == c) & rt["clean"]]
    gen = tg["templates"][tg["y"] == c]
    for k in rng.choice(len(real), 12, replace=False):
        ax.plot(t, real[k, 1], color="0.6", lw=0.7)
    for k in rng.choice(len(gen), 12, replace=False):
        ax.plot(t, gen[k, 1], color="tab:red", lw=0.7, alpha=0.8)
    ax.set_title(CLASS_ORDER[c])
    ax.set_xlabel("ms from R")
axes[0].plot([], [], color="0.6", label="real (12)")
axes[0].plot([], [], color="tab:red", label="generated (12)")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("Lead II (z)")
fig.suptitle("Median beat templates: real vs generated", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig1_templates.png", dpi=130)

# Fig 2: full ECGs
X = np.load("processed_data/X_train.npy", mmap_mode="r")
y = np.load("processed_data/y_train.npy")
ts = np.arange(1000) / FS
fig, axes = plt.subplots(4, 2, figsize=(15, 8), sharex=True)
for c in range(4):
    i = rng.choice(np.where(y == c)[0])
    j = rng.choice(np.where(tg["y"] == c)[0])
    axes[c, 0].plot(ts, X[i][:, 1], "k", lw=0.7)
    axes[c, 1].plot(ts, tg["X"][j][:, 1], "tab:red", lw=0.7)
    axes[c, 0].set_ylabel(CLASS_ORDER[c])
axes[0, 0].set_title("real (train)")
axes[0, 1].set_title("generated template + rule-based assembly")
for ax in axes[-1]:
    ax.set_xlabel("s")
fig.tight_layout()
fig.savefig(OUT / "fig2_ecgs.png", dpi=130)

# Fig 3: TSTR
r = json.loads((GC / "tstr/tstr_results.json").read_text())["results"]
names = ["real (TRTR)", "beat_oracle", "beat_gen", "vpred5k", "v1"]
labels = ["real ECGs", "beat assembly\n(real templates)", "beat assembly\n(generated)", "diffusion\nv-pred 8x64", "diffusion\nv1 (Week 4)"]
vals = [r[n]["balanced_acc_mean"] for n in names]
fig, ax = plt.subplots(figsize=(8, 3.6))
bars = ax.bar(labels, vals, color=["0.4", "tab:orange", "tab:red", "tab:blue", "tab:cyan"])
ax.axhline(0.25, ls="--", color="0.5", lw=0.8)
ax.text(4.45, 0.26, "chance", fontsize=8, color="0.4", ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_ylabel("balanced accuracy on real val")
ax.set_ylim(0, 1)
ax.set_title("Train on synthetic, test on real (440/class, 3 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig3_tstr.png", dpi=130)

# Fig 4: main experiment
res_path = GC / "experiment/main/results.json"
if res_path.exists():
    res = json.loads(res_path.read_text())
    conds = [c for c in ("real", "oversample", "random", "filtered", "filtered_nolabel") if any(k.startswith(c + "/") for k in res)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (metric, get) in zip(axes, (("test macro-F1", lambda v: v["test"]["macro_f1"]),
                                        ("test SBRAD F1", lambda v: v["test"]["f1"]["SBRAD"]))):
        data = [[get(v) for k, v in res.items() if k.startswith(c + "/")] for c in conds]
        ax.boxplot(data, tick_labels=conds)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + rng.uniform(-0.08, 0.08, len(d)), d, s=10, color="tab:red", zorder=3)
        ax.set_title(metric)
    fig.suptitle("Main experiment (10 seeds per condition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_experiment.png", dpi=130)
print("figures ->", OUT)
