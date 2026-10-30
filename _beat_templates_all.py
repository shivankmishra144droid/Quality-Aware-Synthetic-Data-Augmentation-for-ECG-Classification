"""Extract per-recording median beat templates from ALL train ECGs (train split only).
Training data for the template generator (Phase 1.3). Writes
generator_comparison/beat_templates/templates_train.npz (gitignored):
  templates (N, 8, 70), y (N,), rr_mean (N,), rr_cv (N,), clean (N,) bool, src_idx (N,)
clean = the two HR estimators agree (non-AFIB); AFIB is irregular by definition so always True."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generator_comparison.eval_metrics import CLASS_ORDER
from synthetic_generation.beats import recording_is_clean, recording_template
from synthetic_generation.config import DATA_DIR
from synthetic_generation.data import load_train
from synthetic_generation.leads import to_independent_8

OUT = Path("generator_comparison/beat_templates")
X, y = load_train(DATA_DIR)
T, Y, RM, RC, CL, SRC = [], [], [], [], [], []
for i in range(len(X)):
    x8 = to_independent_8(X[i:i + 1])[0].astype(np.float64)
    tpl, _, info = recording_template(x8)
    if tpl is None:
        continue
    T.append(tpl.astype(np.float32))
    Y.append(y[i])
    RM.append(info["rr_mean"])
    RC.append(info["rr_cv"])
    CL.append(True if y[i] == 1 else recording_is_clean(x8))
    SRC.append(i)
    if i % 2000 == 0:
        print(i, flush=True)
Y, CL = np.asarray(Y), np.asarray(CL)
np.savez_compressed(OUT / "templates_train.npz", templates=np.stack(T), y=Y, rr_mean=np.asarray(RM),
                    rr_cv=np.asarray(RC), clean=CL, src_idx=np.asarray(SRC))
for c, n in enumerate(CLASS_ORDER):
    print(n, "templates", int((Y == c).sum()), "clean", int(((Y == c) & CL).sum()))
print("done")
