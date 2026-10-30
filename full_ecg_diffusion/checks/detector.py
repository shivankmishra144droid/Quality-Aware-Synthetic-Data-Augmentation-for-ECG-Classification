"""Real-vs-synthetic detector: a small 1D-CNN, stratified 5-fold, patient-grouped AUC.

Independent of the frozen baseline. Out-of-fold scores give the pooled AUC, per-class AUC and a
bootstrap CI. `control` runs the same detector on two disjoint real halves (expected ~0.5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ..config import CLASS_ORDER, DETECTOR, SEED


class Detector(nn.Module):
    def __init__(self, c_in: int = 12):
        super().__init__()

        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=2, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.f = nn.Sequential(blk(c_in, 32, 7), blk(32, 64, 7), blk(64, 64, 5), blk(64, 128, 5), blk(128, 128, 3))
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)).squeeze(1)


def _fit_predict(xtr, ytr, xte, cfg, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Inner 10% split picks the best epoch, so the test fold never steers training.
    idx = rng.permutation(len(xtr))
    n_in = max(1, len(idx) // 10)
    iv, it = idx[:n_in], idx[n_in:]
    net = Detector(xtr.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    lossf = nn.BCEWithLogitsLoss()
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)  # noqa: E731

    def predict(x):
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(x), 256):
                out.append(torch.sigmoid(net(t(x[s:s + 256]))).cpu().numpy())
        return np.concatenate(out)

    best, best_state = -1.0, None
    for _ in range(cfg["epochs"]):
        net.train()
        order = rng.permutation(it)
        for s in range(0, len(order), cfg["batch_size"]):
            b = order[s:s + cfg["batch_size"]]
            if len(b) < 2:
                continue
            opt.zero_grad()
            lossf(net(t(xtr[b])), t(ytr[b])).backward()
            opt.step()
        yv = ytr[iv]
        auc = roc_auc_score(yv, predict(xtr[iv])) if len(np.unique(yv)) == 2 else 0.0
        if auc > best:
            best, best_state = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return predict(xte)


def _bootstrap_ci(y, s, n, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_detector(x_a, x_b, groups_a, groups_b, cls_a, cls_b, cfg=None, seed: int = SEED, device=None) -> dict:
    """x_a = real (label 0), x_b = synthetic (label 1); inputs (N, 1000, 12)."""
    cfg = cfg or DETECTOR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = np.concatenate([x_a, x_b]).transpose(0, 2, 1).astype(np.float32)
    y = np.r_[np.zeros(len(x_a)), np.ones(len(x_b))].astype(np.float32)
    g = np.r_[[f"a{v}" for v in groups_a], [f"b{v}" for v in groups_b]]
    cls = np.r_[cls_a, cls_b]
    strat = (y * 10 + cls).astype(int)  # stratify on real/syn x class
    oof = np.zeros(len(y))
    folds = list(StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=seed).split(x, strat, g))
    fold_auc = []
    for k, (tr, te) in enumerate(folds):
        oof[te] = _fit_predict(x[tr], y[tr], x[te], cfg, device, seed + k)
        if len(np.unique(y[te])) == 2:
            fold_auc.append(float(roc_auc_score(y[te], oof[te])))
    per_class = {}
    for c, name in enumerate(CLASS_ORDER):
        m = cls == c
        if len(np.unique(y[m])) == 2:
            per_class[name] = float(roc_auc_score(y[m], oof[m]))
    return {
        "auc": float(roc_auc_score(y, oof)),
        "auc_ci95": _bootstrap_ci(y, oof, cfg["bootstrap"], seed),
        "fold_auc": fold_auc,
        "per_class_auc": per_class,
        "n_a": int(len(x_a)), "n_b": int(len(x_b)),
    }


def real_control(x_real, groups, cls, cfg=None, seed: int = SEED) -> dict:
    """Split real records into two halves by patient; the detector should not separate them."""
    rng = np.random.default_rng(seed + 7)
    pats = np.unique(groups)
    side = dict(zip(pats, rng.integers(0, 2, len(pats))))
    m = np.array([side[p] for p in groups]) == 1
    return run_detector(x_real[~m], x_real[m], groups[~m], groups[m], cls[~m], cls[m], cfg, seed)


def verdict(auc: float, th) -> str:
    if auc <= th.detector_pass:
        return "PASS"
    if auc >= th.detector_fail:
        return "FAIL"
    return "PARTIAL"
