"""Round-trip ceiling for template assembly (CPU only).
Rebuild real train ECGs from their OWN median beat placed at their OWN R-peaks and compare with
the high-passed original. This is the best template assembly can do; the AFIB gap is what the
f-wave model must supply. Writes generator_comparison/beat_templates/roundtrip.json + figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.beats import highpass, place_beats, recording_is_clean, recording_template
from synthetic_generation.config import DATA_DIR
from synthetic_generation.data import load_train
from synthetic_generation.leads import to_independent_8

OUT = Path("generator_comparison/beat_templates")
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 300

X, y = load_train(DATA_DIR)
rng = np.random.default_rng(0)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


res, examples = {}, {}
for c, name in enumerate(CLASS_ORDER):
    idx = rng.choice(np.where(y == c)[0], N_PER_CLASS, replace=False)
    c_all, c_ii, resid, clean, nb, fail = [], [], [], [], [], 0
    for i in idx:
        x8 = to_independent_8(X[i:i + 1])[0]
        tpl, r, info = recording_template(x8)
        if tpl is None:
            fail += 1
            continue
        orig = highpass(x8.astype(np.float64))
        rec = place_beats(tpl, r)
        c_all.append(corr(orig, rec))
        c_ii.append(corr(orig[1], rec[1]))
        resid.append(float(((orig - rec) ** 2).sum() / (orig ** 2).sum()))
        clean.append(recording_is_clean(x8))
        nb.append(info["n_beats"])
        if name not in examples:
            examples[name] = (orig, rec)
    clean = np.asarray(clean)
    c_all = np.asarray(c_all)
    res[name] = {
        "n": len(c_all), "template_fail": fail,
        "corr_8lead_median": float(np.median(c_all)),
        "corr_8lead_p10": float(np.percentile(c_all, 10)),
        "corr_leadII_median": float(np.median(c_ii)),
        "residual_energy_median": float(np.median(resid)),
        "clean_frac": float(clean.mean()),
        "corr_8lead_median_clean": float(np.median(c_all[clean])) if clean.any() else None,
        "beats_per_rec_median": float(np.median(nb)),
    }
    print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()}))

(OUT / "roundtrip.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

t = np.arange(1000) / 100
fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
for ax, name in zip(axes, CLASS_ORDER):
    o, r = examples[name]
    ax.plot(t, o[1], "k", lw=0.8, label="real (high-passed)")
    ax.plot(t, r[1], "tab:red", lw=0.8, alpha=0.8, label="own template at own R-peaks")
    ax.set_title(f"{name}  Lead II", fontsize=9)
    ax.set_yticks([])
axes[0].legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("s")
fig.tight_layout()
fig.savefig(OUT / "roundtrip_lead_ii.png", dpi=110)
print("done")
