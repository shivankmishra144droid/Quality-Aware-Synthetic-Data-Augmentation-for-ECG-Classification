"""Oracle pool for template assembly: REAL median-beat templates + fitted rhythm/residual/wander
models (train split only). If this pool is not close to real on the scorecard and the
real-vs-synthetic detector, generated templates will not be either, so no generator is trained
until this works. Also saves a larger 500/class set for the detector.
Writes generator_comparison/beat_oracle/ (32/class pool) and beat_oracle_big.npz."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt
from synthetic_generation.assembly import assemble, fit_class_models
from synthetic_generation.config import DATA_DIR
from synthetic_generation.data import load_train
from synthetic_generation.sample import metadata_rows

ROOT = Path("generator_comparison")
OUT = ROOT / "beat_oracle"
OUT.mkdir(parents=True, exist_ok=True)
N_FIT, N_POOL, N_BIG = 600, 32, 500

X, y = load_train(DATA_DIR)
rng = np.random.default_rng(42)
models = {}
for c, name in enumerate(CLASS_ORDER):
    models[c] = fit_class_models(X, y, c, N_FIT, rng)
    print(name, "templates used", models[c]["n_used"], flush=True)
with open(ROOT / "beat_templates" / "assembly_models.pkl", "wb") as f:
    pickle.dump(models, f)


def build(n):
    xs, ys = [], []
    for c in range(4):
        T = models[c]["templates"]
        for _ in range(n):
            xs.append(assemble(T[rng.integers(len(T))], models[c], rng))
            ys.append(c)
    return np.stack(xs).astype(np.float32), np.asarray(ys, np.int64)


Xp, yp = build(N_POOL)
np.save(OUT / "X_synthetic.npy", Xp)
np.save(OUT / "y_synthetic.npy", yp)
metadata_rows(yp).assign(generator="beat-oracle (real templates + rule-based assembly)").to_csv(
    OUT / "metadata.csv", index=False)
summary = {CLASS_ORDER[c]: {k: round(v, 3) if isinstance(v, float) else v
                            for k, v in lead_ii_hr_pt(Xp[yp == c]).items() if k != "iqr_hr_peaks"}
           for c in range(4)}
print("pool std %.3f" % Xp.std(), json.dumps(summary))
Xb, yb = build(N_BIG)
np.savez_compressed(ROOT / "beat_templates" / "beat_oracle_big.npz", X=Xb, y=yb)
(OUT / "config.json").write_text(json.dumps({
    "what": "real per-recording median templates (fixed R-250..R+450 ms window) + fitted RR KDE, "
            "cross-spectral residual GP (incl. AFIB f-waves), wander GP, respiratory modulation",
    "n_fit_per_class": N_FIT, "templates_used": {CLASS_ORDER[c]: models[c]["n_used"] for c in range(4)},
    "lead_ii_hr_pt": summary}, indent=2), encoding="utf-8")
print("done")
