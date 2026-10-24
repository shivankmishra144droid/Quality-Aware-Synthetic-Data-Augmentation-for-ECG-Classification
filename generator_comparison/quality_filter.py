"""Week 5 quality filter: one quality score per synthetic ECG, then keep the top-k per class.

score = mean of three components, each mapped to [0, 1]:
  typicality  class-conditional: how typical the sample's features are for ITS OWN class, judged
              against real train ECGs of that class (two-sided empirical percentile). Three equally
              weighted groups: rhythm (Lead-II periodicity, peak HR, peak-vs-ACF HR disagreement),
              amplitude (8 per-lead std), spectrum (shape distance to the real class mean).
              Real ECGs average ~0.5 by construction.
  realism     out-of-fold P(real) from the realism detector (k-fold cross-fitting, so no sample is
              scored by a detector that trained on it).
  label       frozen baseline CNN-LSTM's softmax probability of the intended class (scorer only,
              never retrained).
Uses only train-split real ECGs.

Usage: py -3.14 -m generator_comparison.quality_filter <pool.npz> <out_scores.npz> [--folds 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch

from generator_comparison.eval_metrics import CLASS_ORDER, hr_from_acf, hr_from_peaks, qrs_envelope
from generator_comparison.realism_detector import Det
from generator_comparison.tstr import CNNLSTM

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "processed_data"
FS = 100
IND = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6
N_REF = 500


def _logpsd(x12):
    f, p = welch(x12[:, IND], fs=FS, nperseg=256, axis=0)          # (F, 8)
    keep = (f >= 0.5) & (f <= 40)
    p = p[keep] / (p[keep].sum(0, keepdims=True) + 1e-12)
    return np.log10(p + 1e-12)


def features(X: np.ndarray):
    """Per-sample scalar features (N, 11) and log-PSD (N, F, 8)."""
    feats, psds = [], []
    for x in X:
        env = qrs_envelope(x[:, 1].astype(np.float64))
        hp = hr_from_peaks(env)
        ha, per = hr_from_acf(env)
        dis = abs(hp - ha) / ha if np.isfinite(hp) and np.isfinite(ha) else 1.0
        feats.append([per, hp if np.isfinite(hp) else 0.0, dis, *x[:, IND].std(0)])
        psds.append(_logpsd(x))
    return np.asarray(feats), np.asarray(psds)


class Reference:
    """Per-class real feature distributions from train ECGs."""

    def __init__(self, n_ref=N_REF, seed=0):
        X = np.load(DATA / "X_train.npy", mmap_mode="r")
        y = np.load(DATA / "y_train.npy")
        rng = np.random.default_rng(seed)
        self.feat, self.psd_mean, self.psd_dist = {}, {}, {}
        for c in range(4):
            idx = np.sort(rng.choice(np.where(y == c)[0], min(n_ref, int((y == c).sum())), replace=False))
            f, p = features(np.asarray(X[idx]))
            mean = p.mean(0)
            self.feat[c] = f
            self.psd_mean[c] = mean
            self.psd_dist[c] = np.abs(p - mean).mean(axis=(1, 2))

    def typicality(self, f, p, c):
        """Two-sided empirical percentile per feature (1 = median of real class, 0 = beyond all)."""
        cols = np.column_stack([f, np.abs(p - self.psd_mean[c]).mean(axis=(1, 2))])
        ref = np.column_stack([self.feat[c], self.psd_dist[c]])
        pct = np.stack([(ref[:, j][None] < cols[:, j][:, None]).mean(1) for j in range(cols.shape[1])], 1)
        two_sided = 1 - 2 * np.abs(pct - 0.5)
        two_sided[:, -1] = 1 - pct[:, -1]          # spectral distance: smaller is better (one-sided)
        # Equal weight per feature GROUP, so the 8 per-lead stds do not outvote the 3 rhythm features.
        rhythm, amp, spec = two_sided[:, :3].mean(1), two_sided[:, 3:11].mean(1), two_sided[:, 11]
        return (rhythm + amp + spec) / 3


def label_prob(X, y):
    sd = torch.load(ROOT.parent / "baseline" / "cnn_lstm_baseline.pt", map_location="cpu", weights_only=False)
    net = CNNLSTM()
    net.load_state_dict(sd["model_state_dict"])
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out.append(torch.softmax(net(torch.from_numpy(X[i:i + 512].astype(np.float32))), 1).numpy())
    p = np.concatenate(out)
    return p[np.arange(len(y)), y]


def realism_oof(X, y, folds=5, seed=0, epochs=10):
    """Out-of-fold P(real) for each synthetic sample (class-matched real train ECGs as negatives)."""
    Xr_all = np.load(DATA / "X_train.npy", mmap_mode="r")
    yr_all = np.load(DATA / "y_train.npy")
    rng = np.random.default_rng(seed)
    n_c = [min(int((yr_all == c).sum()), int((y == c).sum())) for c in range(4)]
    ridx = np.sort(np.concatenate([rng.choice(np.where(yr_all == c)[0], n_c[c], replace=False) for c in range(4)]))
    Xr = np.asarray(Xr_all[ridx], np.float32)
    fold_s = rng.permutation(len(X)) % folds
    fold_r = rng.permutation(len(Xr)) % folds
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_real = np.zeros(len(X))
    for k in range(folds):
        xa = np.concatenate([Xr[fold_r != k], X[fold_s != k].astype(np.float32)])
        la = np.concatenate([np.zeros((fold_r != k).sum()), np.ones((fold_s != k).sum())]).astype(np.float32)
        torch.manual_seed(seed + k)
        net = Det().to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        lossf = torch.nn.BCEWithLogitsLoss()
        Xt, Lt = torch.from_numpy(xa), torch.from_numpy(la)
        for _ in range(epochs):
            net.train()
            for b in np.array_split(rng.permutation(len(xa)), max(1, len(xa) // 64)):
                opt.zero_grad()
                lossf(net(Xt[b].to(dev)), Lt[b].to(dev)).backward()
                opt.step()
        net.eval()
        te = np.where(fold_s == k)[0]
        with torch.no_grad():
            for i in range(0, len(te), 256):
                b = te[i:i + 256]
                p_real[b] = 1 - torch.sigmoid(net(torch.from_numpy(X[b].astype(np.float32)).to(dev))).cpu().numpy()
    return p_real


def score_pool(X, y, folds=5):
    ref = Reference()
    f, p = features(X)
    typ = np.zeros(len(X))
    for c in range(4):
        m = y == c
        typ[m] = ref.typicality(f[m], p[m], c)
    lab = label_prob(X, y)
    rea = realism_oof(X, y, folds=folds)
    return {"typicality": typ, "label": lab, "realism": rea, "score": (typ + lab + rea) / 3}


def select(scores, y, k_per_class):
    """Indices of the top-k scoring samples per class."""
    keep = [np.where(y == c)[0][np.argsort(-scores[y == c])[:k_per_class]] for c in range(4)]
    return np.sort(np.concatenate(keep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("out")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    d = np.load(a.pool)
    s = score_pool(d["X"], d["y"], folds=a.folds)
    np.savez_compressed(a.out, y=d["y"], **s)
    for c in range(4):
        m = d["y"] == c
        print(CLASS_ORDER[c], {k: round(float(np.median(v[m])), 3) for k, v in s.items()})
