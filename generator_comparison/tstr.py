"""TSTR: Train on Synthetic, Test on Real.

Trains the baseline CNN-LSTM architecture (fresh weights, never the frozen baseline/ model) on a
class-balanced training set from one source, then evaluates on the REAL validation split (the
test split stays untouched). TRTR (same-size real train subset) is the reference.
Same recipe for every source: N/class, Adam 1e-3, wd 1e-4, batch 64, 25 epochs, unweighted CE
(training sets are balanced), last epoch (no selection on val). 3 seeds.

Run from repo root: py -3.14 -m generator_comparison.tstr
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from generator_comparison.eval_metrics import CLASS_ORDER

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "processed_data"
OUT = ROOT / "tstr"
N_PER_CLASS = 440                    # SBRAD has 446 real train ECGs
SEEDS = (0, 1, 2)
EPOCHS = 25
SOURCES = {
    "beat_oracle": ROOT / "beat_templates" / "beat_oracle_big.npz",
    "vpred5k": OUT / "pool_vpred5k.npz",
    "v1": OUT / "pool_v1.npz",
    "beat_gen": OUT / "pool_beatgen.npz",          # generated templates + same assembly as the oracle
}


class CNNLSTM(nn.Module):  # identical to 03_CNN_LSTM_Baseline.ipynb
    def __init__(self, n_leads=12, n_classes=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


def balanced_subset(X, y, n, rng):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in range(4)])
    return np.asarray(X[np.sort(idx)], np.float32), y[np.sort(idx)]


def train_eval(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = CNNLSTM()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    for _ in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // 64)):
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(torch.from_numpy(Xva[i:i + 512])).argmax(1)
                          for i in range(0, len(Xva), 512)]).numpy()
    return {
        "macro_f1": float(f1_score(yva, pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(yva, pred)),
        "f1": dict(zip(CLASS_ORDER, map(float, f1_score(yva, pred, average=None, labels=range(4))))),
        "recall": dict(zip(CLASS_ORDER, map(float, recall_score(yva, pred, average=None, labels=range(4))))),
        "confusion": confusion_matrix(yva, pred, labels=range(4)).tolist(),
    }


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    Xva = np.load(DATA / "X_val.npy").astype(np.float32)
    yva = np.load(DATA / "y_val.npy")
    Xtr_all = np.load(DATA / "X_train.npy", mmap_mode="r")
    ytr_all = np.load(DATA / "y_train.npy")
    sets = {"real (TRTR)": None}
    sets.update({k: p for k, p in SOURCES.items() if p.exists()})
    results = {}
    for name, path in sets.items():
        runs = []
        for s in SEEDS:
            rng = np.random.default_rng(100 + s)
            if path is None:
                Xtr, ytr = balanced_subset(Xtr_all, ytr_all, N_PER_CLASS, rng)
            else:
                d = np.load(path)
                Xtr, ytr = balanced_subset(d["X"], d["y"], N_PER_CLASS, rng)
            runs.append(train_eval(Xtr, ytr, Xva, yva, s))
            print(name, "seed", s, "macroF1 %.3f" % runs[-1]["macro_f1"], flush=True)
        mf = [r["macro_f1"] for r in runs]
        results[name] = {
            "macro_f1_mean": float(np.mean(mf)), "macro_f1_std": float(np.std(mf)),
            "balanced_acc_mean": float(np.mean([r["balanced_acc"] for r in runs])),
            "f1_mean": {c: float(np.mean([r["f1"][c] for r in runs])) for c in CLASS_ORDER},
            "recall_mean": {c: float(np.mean([r["recall"][c] for r in runs])) for c in CLASS_ORDER},
            "runs": runs,
        }
    OUT.mkdir(exist_ok=True)
    (OUT / "tstr_results.json").write_text(json.dumps({
        "recipe": {"n_per_class": N_PER_CLASS, "epochs": EPOCHS, "seeds": SEEDS, "eval": "real val split",
                   "model": "CNN-LSTM (baseline architecture, fresh weights)"},
        "results": results}, indent=2), encoding="utf-8")
    lines = ["| train on | macro-F1 (val) | bal. acc | F1 SR / AFIB / STACH / SBRAD | recall SR / AFIB / STACH / SBRAD |",
             "|---|---|---|---|---|"]
    for name, r in results.items():
        f, rc = r["f1_mean"], r["recall_mean"]
        lines.append(f"| {name} | {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} | {r['balanced_acc_mean']:.3f} | "
                     + " / ".join(f"{f[c]:.2f}" for c in CLASS_ORDER) + " | "
                     + " / ".join(f"{rc[c]:.2f}" for c in CLASS_ORDER) + " |")
    table = "\n".join(lines)
    (OUT / "tstr_results.md").write_text("# TSTR results\n\n" + table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
