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

Study 2 (report/PREREGISTRATION_2.md) adds, all off by default so earlier tags are unchanged:
  --steps S        fixed budget: exactly S optimizer steps of batch 64 for every condition (no early stopping),
                   val every --eval-every steps, best-val state kept (train_steps)
  --deterministic  deterministic CUDA / cuDNN kernels (set_deterministic)
  --val-only       never loads the test split, saves no test predictions (checks before registration)

Usage: py -3.14 -m generator_comparison.augment_experiment <pool.npz> <scores.npz> [--seeds 5] [--ratio 1.0]
       [--conditions real,random,filtered] [--max-epochs 25] [--tag NAME]
       [--steps S --eval-every 250] [--deterministic] [--val-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    """Indices into the pool for one condition. random: uniform per class; filtered: top-k per class.
    scores: dict of per-sample score arrays from quality_filter."""
    if cond == "real":
        return np.array([], dtype=np.int64)
    if cond == "random":
        rng = np.random.default_rng(1000 + seed)
        return np.concatenate([rng.choice(np.where(pool_y == c)[0], counts[c], replace=False) for c in MINORITY])
    if cond in FILTER_KEYS:   # deterministic: the same top-k for every seed
        # Each class gets its own k. (Before 2026-10-28 this took top-max(k) and sliced by POOL position, so
        # STACH / SBRAD were not their top-k: 54 % / 38 % overlap. Results in experiment/main and
        # experiment/confirmatory were produced with that bug.)
        return select(scores[FILTER_KEYS[cond]], pool_y, counts)
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


def train_steps(Xtr, ytr, Xva, yva, seed, dev, steps, eval_every=250, batch=64):
    """Fixed optimisation budget: exactly `steps` optimizer steps of `batch` samples, whatever the training-set
    size, so augmented conditions get no extra updates. Val macro-F1 every `eval_every` steps; LR halves after
    3 evaluations without improvement; the best-val state is kept. No early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    w = len(ytr) / (4.0 * np.bincount(ytr, minlength=4))
    net = CNNLSTM().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    order, pos = rng.permutation(len(Xt)), 0
    best, best_state, best_step, hist = -1.0, None, 0, []
    for step in range(1, steps + 1):
        if pos + batch > len(order):          # new pass over the data; the partial last batch is dropped
            order, pos = rng.permutation(len(Xt)), 0
        b = order[pos:pos + batch]
        pos += batch
        net.train()
        opt.zero_grad()
        lossf(net(Xt[b].to(dev)), yt[b].to(dev)).backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            f1 = f1_score(yva, predict(net, Xva, dev).argmax(1), average="macro")
            sched.step(f1)
            hist.append(round(float(f1), 4))
            if f1 > best:
                best, best_step = f1, step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, {"best_val_macro_f1": float(best), "best_step": best_step, "steps": steps, "eval_every": eval_every,
                 "val_history": hist, "class_weights": dict(zip(CLASS_ORDER, map(float, w)))}


# Config fields that must match for a resumed run to reuse finished results (val_only, seeds, conditions may differ).
RECIPE_KEYS = ("pool_sha256", "scores_sha256", "ratio", "max_epochs", "steps", "eval_every", "deterministic")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def set_deterministic():
    """Deterministic kernels; must run before the first CUDA op (cuBLAS reads the workspace setting then)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y, prob):
    pred = prob.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "accuracy": float((pred == y).mean()),
            "f1": dict(zip(CLASS_ORDER, map(float, f1_score(y, pred, average=None, labels=range(4))))),
            "recall": dict(zip(CLASS_ORDER, map(float, recall_score(y, pred, average=None, labels=range(4)))))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("scores")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0, help="first seed (confirmatory run uses 100)")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--conditions", default="real,random,filtered")
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--steps", type=int, default=0, help="fixed optimizer-step budget (0: epoch recipe)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--val-only", action="store_true", help="never load the test split")
    a = ap.parse_args(argv)

    if a.deterministic:
        set_deterministic()
        if not torch.cuda.is_available():   # e.g. the GPU dropped out after sleep: a CPU run is neither comparable nor feasible
            raise SystemExit("--deterministic run needs CUDA, and torch.cuda.is_available() is False. Not falling back to CPU.")
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = np.load(DATA / "X_train.npy").astype(np.float32)
    ytr = np.load(DATA / "y_train.npy").astype(np.int64)
    Xva, yva = np.load(DATA / "X_val.npy").astype(np.float32), np.load(DATA / "y_val.npy").astype(np.int64)
    if not a.val_only:
        Xte, yte = np.load(DATA / "X_test.npy").astype(np.float32), np.load(DATA / "y_test.npy").astype(np.int64)
    pool = np.load(a.pool)
    pX, py = pool["X"].astype(np.float32), pool["y"].astype(np.int64)
    scores = dict(np.load(a.scores))
    assert len(scores["score"]) == len(py) and np.array_equal(scores["y"], py), "scores do not match the pool"
    counts = synthetic_counts(ytr, a.ratio)
    for c in MINORITY:
        assert (py == c).sum() >= counts[c], f"pool too small for {CLASS_ORDER[c]}"
    cfg = {"pool": a.pool, "scores": a.scores, "ratio": a.ratio, "synthetic_per_class":
           {CLASS_ORDER[c]: n for c, n in counts.items()}, "seeds": a.seeds, "seed_start": a.seed_start, "max_epochs": a.max_epochs,
           "steps": a.steps, "eval_every": a.eval_every, "deterministic": a.deterministic, "val_only": a.val_only,
           "pool_sha256": sha256(a.pool), "scores_sha256": sha256(a.scores), "device": str(dev)}
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    if results and (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        changed = [k for k in RECIPE_KEYS if k in old and old[k] != cfg[k]]   # keys missing in old configs: not checked
        if changed:
            raise SystemExit(f"{out} has finished runs made with a different {changed}; resuming would mix them. "
                             "Use a new --tag.")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("CONFIG", json.dumps(cfg), flush=True)

    for cond in a.conditions.split(","):
        for seed in range(a.seed_start, a.seed_start + a.seeds):
            key = f"{cond}/seed{seed}"
            if key in results and (a.val_only or "test" in results[key]):   # a val-only entry is redone in a full run
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
            if a.steps:
                net, info = train_steps(Xa, ya, Xva, yva, seed, dev, a.steps, a.eval_every)
            else:
                net, info = train_one(Xa, ya, Xva, yva, seed, dev, a.max_epochs)
            results[key] = {**info, "n_synthetic": int(len(sel)), "n_added": int(len(ya) - len(ytr)),
                            "val": metrics(yva, predict(net, Xva, dev))}
            if not a.val_only:
                pte = predict(net, Xte, dev)
                np.save(out / f"test_prob_{cond}_seed{seed}.npy", pte)
                results[key]["test"] = metrics(yte, pte)
            results[key]["sec"] = round(time.time() - t0, 1)
            (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            r = results[key]
            when = f"step {r['best_step']}" if a.steps else f"ep {r['best_epoch']}"
            # Test numbers are saved but not printed: they are read once, in the statistics step.
            print(f"{key}: val mF1 {r['val']['macro_f1']:.3f} | val SBRAD F1 {r['val']['f1']['SBRAD']:.3f} "
                  f"| {when} | {r['sec']}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
