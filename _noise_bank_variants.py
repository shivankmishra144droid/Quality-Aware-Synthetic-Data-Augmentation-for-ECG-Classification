"""Part B of report/PREREGISTRATION_3.md: build the real-noise-bank variants (CPU, no training).

Donors: REAL train ECGs, drawn exactly as _beat_ablation.py does (rng 7, up to min(500, n_c // 2) per class, so
disjoint real ECGs remain for the detector); every donor index goes into src_idx.
Generated templates: the `templates` arrays stored in existing pool npz files (default: tstr/pool_beatgen.npz,
which has all four classes, then pools2/pool_1.npz for more minority templates). A template is used once per
variant; if a class's bank is smaller than its donor count, it is reused and that is recorded.

Registered variants (synthesize: donor + alpha * (T' - own template) at the donor's own R-peaks):
  N0            random template, no alignment, alpha 1     (should reproduce ablation T_rand, AUC ~0.95)
  N1            random template, aligned,      alpha 1
  N2            nearest template, aligned,     alpha 1
  N1a<A>/N2a<A> as N1/N2 with alpha A in 0.25, 0.50, 0.75
  W:<base>      a variant that passed the gate, plus a time warp to a slower within-class rate (--warp-from)

Output: generator_comparison/noise_bank/<variant>.npz with X (N,1000,12), y, src_idx, donor_corr, alpha,
template_reused (gitignored, regenerable).
Usage:  py -3.14 _noise_bank_variants.py [--variants N0,N1,...] [--warp-from N2a0.50]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generator_comparison.noise_bank import align, donor_corr, donor_parts, nearest_template, synthesize, time_warp
from synthetic_generation.leads import reconstruct_12, to_independent_8

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "processed_data"
OUT = ROOT / "generator_comparison" / "noise_bank"
DEFAULT_BANKS = [ROOT / "generator_comparison" / "tstr" / "pool_beatgen.npz",
                 ROOT / "generator_comparison" / "pools2" / "pool_1.npz"]
ALPHAS = (0.25, 0.50, 0.75)
REGISTERED = ["N0", "N1", "N2"] + [f"N{b}a{a:.2f}" for b in (1, 2) for a in ALPHAS]
# Class mean-RR bounds (s) used by the assembly rhythm model (synthetic_generation/assembly.py MEAN_RR_BOUNDS);
# a warped donor must stay inside its class.
RR_BOUNDS = {0: (0.6, 1.0), 1: (0.3, 2.0), 2: (0.3, 0.6), 3: (1.0, 2.0)}
WARP_MAX = 1.25


def parse(variant: str) -> dict:
    """'N2a0.50' -> {'mode': 'nearest', 'align': True, 'alpha': 0.5}."""
    if variant == "N0":
        return {"mode": "random", "align": False, "alpha": 1.0}
    base, _, a = variant.partition("a")
    if base not in ("N1", "N2"):
        raise ValueError(f"unknown variant {variant}")
    return {"mode": "random" if base == "N1" else "nearest", "align": True, "alpha": float(a) if a else 1.0}


def load_banks(paths) -> dict:
    tpl, ys, used = [], [], []
    for p in paths:
        if Path(p).exists():
            d = np.load(p)
            tpl.append(d["templates"].astype(np.float64))
            ys.append(d["y"].astype(np.int64))
            used.append(str(p))
    if not tpl:
        raise SystemExit(f"no template bank found in {list(map(str, paths))}: build one with _beat_gen_pool.py")
    t, y = np.concatenate(tpl), np.concatenate(ys)
    missing = [c for c in range(4) if not (y == c).any()]
    if missing:
        raise SystemExit(f"template banks {used} have no templates for classes {missing}")
    return {"templates": t, "y": y, "sources": used}


def donors(X, y, seed=7):
    """Exactly _beat_ablation.py's donor draw; returns a list of (train index, class, parts)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(4):
        idx = rng.choice(np.where(y == c)[0], min(500, int((y == c).sum()) // 2), replace=False)
        for i in idx:
            p = donor_parts(to_independent_8(X[i:i + 1])[0])
            if p is not None:
                out.append((int(i), c, p))
    return out


def build(variant, dons, bank, seed=0):
    spec = parse(variant)
    rng = np.random.default_rng(seed)
    xs, ys, src, alphas, reused, x_don = [], [], [], [], [], []
    for c in range(4):
        cls_idx = np.flatnonzero(bank["y"] == c)
        avail = np.ones(len(cls_idx), bool)
        for i, cc, p in (d for d in dons if d[1] == c):
            if not avail.any():
                avail[:] = True                                   # bank exhausted: allow reuse, record it
                reused.append(i)
            if spec["mode"] == "random":
                j = int(rng.choice(np.flatnonzero(avail)))
            else:
                j = nearest_template(bank["templates"][cls_idx], p["tpl"], avail)
            avail[j] = False
            t = bank["templates"][cls_idx[j]]
            if spec["align"]:
                t, _ = align(t, p["tpl"])
            x8 = synthesize(p, t, spec["alpha"])
            xs.append(x8)
            x_don.append(p["x8"])
            ys.append(c)
            src.append(i)
            alphas.append(spec["alpha"])
    x12 = reconstruct_12(np.stack(xs).astype(np.float32))
    d12 = reconstruct_12(np.stack(x_don).astype(np.float32))
    return {"X": x12, "y": np.asarray(ys, np.int64), "src_idx": np.asarray(src, np.int64),
            "donor_corr": donor_corr(x12, d12).astype(np.float32), "alpha": np.asarray(alphas, np.float32),
            "template_reused": np.asarray(sorted(set(reused)), np.int64)}


def warp_variant(base_npz: Path, X, y, seed=0):
    """Time-warp every sample of a passing variant to a slower rate that keeps its class mean-RR bounds."""
    d = np.load(base_npz)
    rng = np.random.default_rng(seed)
    xs, keep, rs = [], [], []
    for n, (x12, c, i) in enumerate(zip(d["X"], d["y"], d["src_idx"])):
        x8 = to_independent_8(x12[None])[0].astype(np.float64)
        p = donor_parts(to_independent_8(X[i:i + 1])[0])
        rr = p["rr_mean"] if p is not None else None
        if rr is None:
            continue
        s_max = min(WARP_MAX, RR_BOUNDS[int(c)][1] / rr)
        if s_max <= 1.02:
            continue                                          # cannot slow it and stay in class: left out
        s = float(rng.uniform(1.02, s_max))
        w, _ = time_warp(x8, p["r"], s)
        xs.append(w)
        keep.append(n)
        rs.append(s)
    keep = np.asarray(keep)
    x12 = reconstruct_12(np.stack(xs).astype(np.float32))
    donors12 = np.asarray(X[d["src_idx"][keep]], np.float32)
    return {"X": x12, "y": d["y"][keep], "src_idx": d["src_idx"][keep],
            "donor_corr": donor_corr(x12, donors12).astype(np.float32), "alpha": d["alpha"][keep],
            "warp_s": np.asarray(rs, np.float32), "template_reused": d["template_reused"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(REGISTERED))
    ap.add_argument("--banks", nargs="+", default=[str(p) for p in DEFAULT_BANKS])
    ap.add_argument("--warp-from", default=None, help="variant id that passed the gate (adds W:<id>)")
    a = ap.parse_args(argv)
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    y = np.load(DATA / "y_train.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.warp_from:
        res = warp_variant(OUT / f"{a.warp_from}.npz", X, y)
        np.savez_compressed(OUT / f"W_{a.warp_from}.npz", **res)
        print(f"W_{a.warp_from}: {len(res['y'])} samples, median donor_corr {np.median(res['donor_corr']):.3f}")
        return
    bank = load_banks(a.banks)
    dons = donors(X, y)
    meta = {"banks": bank["sources"], "n_donors": {int(c): sum(d[1] == c for d in dons) for c in range(4)},
            "bank_per_class": {int(c): int((bank["y"] == c).sum()) for c in range(4)}}
    for v in a.variants.split(","):
        res = build(v, dons, bank)
        np.savez_compressed(OUT / f"{v}.npz", **res)
        meta[v] = {"n": int(len(res["y"])), "median_donor_corr": float(np.median(res["donor_corr"])),
                   "n_template_reused_donors": int(len(res["template_reused"]))}
        print(v, json.dumps(meta[v]), flush=True)
    (OUT / "variants.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
