"""Real-vs-synthetic detector: can a small CNN tell synthetic ECGs from real train ECGs?
AUC ~0.5 = indistinguishable; ~1.0 = artifacts (joins, noise statistics, warping) that a
downstream classifier could exploit. A fresh CNN each run, never the frozen baseline.

Usage: py -3.14 -m generator_comparison.realism_detector <synthetic.npz with X, y> [tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from generator_comparison.eval_metrics import CLASS_ORDER

DATA = Path(__file__).resolve().parent.parent / "processed_data"
OUT = Path(__file__).resolve().parent / "scorecard"


class Det(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.head = nn.Linear(128, 1)

    def forward(self, x):                      # x (B, 1000, 12)
        h = self.f(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)


def run(x_syn, y_syn, seed=0, epochs=15, exclude=None):
    """exclude: train indices the synthetic set was built from; kept out of the real side."""
    rng = np.random.default_rng(seed)
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    y = np.load(DATA / "y_train.npy")
    # Equal real/synthetic counts per class, so the class mix is not a shortcut.
    pool = [np.setdiff1d(np.where(y == c)[0], exclude if exclude is not None else []) for c in range(4)]
    n_c = [min(len(pool[c]), int((y_syn == c).sum())) for c in range(4)]
    real_idx = np.sort(np.concatenate([rng.choice(pool[c], n_c[c], replace=False) for c in range(4)]))
    syn_idx = np.concatenate([rng.choice(np.where(y_syn == c)[0], n_c[c], replace=False) for c in range(4)])
    x_syn, y_syn = x_syn[syn_idx], y_syn[syn_idx]
    xr, yr = np.asarray(X[real_idx], np.float32), y[real_idx]
    xa = np.concatenate([xr, x_syn.astype(np.float32)])
    ca = np.concatenate([yr, y_syn])
    lab = np.concatenate([np.zeros(len(xr)), np.ones(len(x_syn))]).astype(np.float32)
    perm = rng.permutation(len(xa))
    n_tr = int(0.7 * len(xa))
    tr, te = perm[:n_tr], perm[n_tr:]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    net = Det().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt, Lt = torch.from_numpy(xa), torch.from_numpy(lab)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            lossf(net(Xt[b].to(dev)), Lt[b].to(dev)).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(net(Xt[b].to(dev))).cpu()
                       for b in np.array_split(te, max(1, len(te) // 256))]).numpy()
    out = {"auc": float(roc_auc_score(lab[te], p)), "n_test": int(len(te)), "per_class": {}}
    for c in range(4):
        m = ca[te] == c
        if len(np.unique(lab[te][m])) == 2:
            out["per_class"][CLASS_ORDER[c]] = float(roc_auc_score(lab[te][m], p[m]))
    return out


if __name__ == "__main__":
    path, tag = Path(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem)
    d = np.load(path)
    ex = d["src_idx"] if "src_idx" in d.files else None
    res = {f"seed{s}": run(d["X"], d["y"], seed=s, exclude=ex) for s in (0, 1)}
    res["auc_mean"] = float(np.mean([r["auc"] for r in res.values()]))
    OUT.mkdir(exist_ok=True)
    (OUT / f"detector_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
