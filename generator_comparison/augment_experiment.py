"""Phase 3: main experiment. Same CNN-LSTM recipe as the frozen baseline, three conditions:
  real          real train only
  random        real train + synthetic minority ECGs drawn at random from the pool
  filtered      real train + the top-scoring synthetic minority ECGs (quality_filter `score`)
  filtered_nolabel  as filtered, but ranked by `score_nolabel` (no frozen-baseline component), so a
                gain cannot come only from picking samples the baseline already agrees with
  oversample    control (added after the main analysis): REAL minority ECGs duplicated to the same added
                counts, so the effect of synthetic content is separated from more minority examples /
                changed class weights
Synthetic counts are identical in `random` and `filtered`: each minority class is doubled
(+1056 AFIB, +568 STACH, +446 SBRAD at ratio 1.0). SR gets no synthetic data.

Recipe (baseline/config.json): Adam lr 1e-3, wd 1e-4, batch 64, max 25 epochs, early stopping
patience 7 and ReduceLROnPlateau(0.5, patience 3) on val macro-F1, weighted CE with
w_c = n_total / (4 n_c) from that condition's training labels. Val selects the epoch; test is
reported per run, and per-sample test probabilities are saved for the statistics step.

Usage: py -3.14 -m generator_comparison.augment_experiment <pool.npz> <scores.npz> [--seeds 5] [--ratio 1.0]
       [--conditions real,random,filtered] [--max-epochs 25] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score

from generator_comparison.eval_metrics import CLASS_ORDER
from generator_comparison.quality_filter import select
from generator_comparison.tstr import CNNLSTM

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "processed_data"
OUT = ROOT / "experiment"
MINORITY = (1, 2, 3)


def synthetic_counts(y_train, ratio):
    return {c: int(round(ratio * (y_train == c).sum())) for c in MINORITY}


FILTER_KEYS = {"filtered": "score", "filtered_nolabel": "score_nolabel"}


def pick(cond, pool_y, scores, counts, seed):
    """scores: dict of per-sample score arrays from quality_filter."""
    """Indices into the pool for one condition. random: uniform per class; filtered: top-k."""
    if cond == "real":
        return np.array([], dtype=np.int64)
    if cond == "random":
        rng = np.random.default_rng(1000 + seed)
        return np.concatenate([rng.choice(np.where(pool_y == c)[0], counts[c], replace=False) for c in MINORITY])
    if cond in FILTER_KEYS:   # deterministic: the same top-k for every seed
        top = select(scores[FILTER_KEYS[cond]], pool_y, max(counts.values()))
        return np.concatenate([top[pool_y[top] == c][:counts[c]] for c in MINORITY])
    raise ValueError(cond)


def predict(net, X, dev):
    net.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(net(torch.from_numpy(X[i:i + 512]).to(dev)), 1).cpu()
                          for i in range(0, len(X), 512)]).numpy()


def train_one(Xtr, ytr, Xva, yva, seed, dev, max_epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    w = len(ytr) / (4.0 * np.bincount(ytr, minlength=4))
    net = CNNLSTM().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // 64)):
            opt.zero_grad()
            lossf(net(Xt[b].to(dev)), yt[b].to(dev)).backward()
            opt.step()
        f1 = f1_score(yva, predict(net, Xva, dev).argmax(1), average="macro")
        sched.step(f1)
        hist.append(round(float(f1), 4))
        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 7:
                break
    net.load_state_dict(best_state)
    return net, {"best_val_macro_f1": float(best), "best_epoch": best_ep, "val_history": hist,
                 "class_weights": dict(zip(CLASS_ORDER, map(float, w)))}


def metrics(y, prob):
    pred = prob.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean()),
            "f1": dict(zip(CLASS_ORDER, map(float, f1_score(y, pred, average=None, labels=range(4))))),
            "recall": dict(zip(CLASS_ORDER, map(float, recall_score(y, pred, average=None, labels=range(4)))))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("scores")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--conditions", default="real,random,filtered")
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()

    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = np.load(DATA / "X_train.npy").astype(np.float32)
    ytr = np.load(DATA / "y_train.npy").astype(np.int64)
    Xva, yva = np.load(DATA / "X_val.npy").astype(np.float32), np.load(DATA / "y_val.npy").astype(np.int64)
    Xte, yte = np.load(DATA / "X_test.npy").astype(np.float32), np.load(DATA / "y_test.npy").astype(np.int64)
    pool = np.load(a.pool)
    pX, py = pool["X"].astype(np.float32), pool["y"].astype(np.int64)
    scores = dict(np.load(a.scores))
    assert len(scores["score"]) == len(py) and np.array_equal(scores["y"], py), "scores do not match the pool"
    counts = synthetic_counts(ytr, a.ratio)
    for c in MINORITY:
        assert (py == c).sum() >= counts[c], f"pool too small for {CLASS_ORDER[c]}"
    cfg = {"pool": a.pool, "scores": a.scores, "ratio": a.ratio, "synthetic_per_class":
           {CLASS_ORDER[c]: n for c, n in counts.items()}, "seeds": a.seeds, "max_epochs": a.max_epochs,
           "device": str(dev)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("CONFIG", json.dumps(cfg), flush=True)

    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    for cond in a.conditions.split(","):
        for seed in range(a.seeds):
            key = f"{cond}/seed{seed}"
            if key in results:
                print("skip (done)", key)
                continue
            if cond == "oversample":
                # Control: duplicate REAL minority ECGs to the same added counts (deterministic). Separates
                # "more minority examples / changed class weights" from "synthetic content".
                sel = np.array([], dtype=np.int64)
                dup = np.concatenate([np.resize(np.where(ytr == c)[0], counts[c]) for c in MINORITY])
                Xa, ya = np.concatenate([Xtr, Xtr[dup]]), np.concatenate([ytr, ytr[dup]])
            else:
                sel = pick(cond, py, scores, counts, seed)
                Xa = np.concatenate([Xtr, pX[sel]]) if len(sel) else Xtr
                ya = np.concatenate([ytr, py[sel]]) if len(sel) else ytr
            t0 = time.time()
            net, info = train_one(Xa, ya, Xva, yva, seed, dev, a.max_epochs)
            pte = predict(net, Xte, dev)
            np.save(out / f"test_prob_{cond}_seed{seed}.npy", pte)
            results[key] = {**info, "n_synthetic": int(len(sel)), "n_added": int(len(ya) - len(ytr)), "val": metrics(yva, predict(net, Xva, dev)),
                            "test": metrics(yte, pte), "sec": round(time.time() - t0, 1)}
            (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            r = results[key]
            # Test numbers are saved but not printed: they are read once, in the statistics step.
            print(f"{key}: val mF1 {r['val']['macro_f1']:.3f} | val SBRAD F1 {r['val']['f1']['SBRAD']:.3f} "
                  f"| ep {r['best_epoch']} | {r['sec']}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
