"""Phase 6 (and 4a): sample with label vectors, convert to our 12-lead z-space, write a pool.

Class-conditional sampling for our 4 classes = conditioning on 71-label vectors of real *train*
records of that class (the model was trained on multi-hot SCP labels, as in the paper).

  python -m full_ecg_diffusion.sample --ckpt generator_comparison/sssd_full/ours_train/ours_last.pt \
      --n-per-class 256 --out generator_comparison/sssd_full/ours_pool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from synthetic_generation.utils import calc_diffusion_hyperparams, hyperparams_to_device, sampling_label

from .config import CLASS_ORDER, DIFFUSION, SEED, add_path_args, paths_from_args
from .convert import foreign_to_z
from .nets import build_net, load_checked, safe_load
from .ptbxl import OurArrays, label_vectors, load_database, norm_stats


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def sample_from_labels(net, labels: np.ndarray, device, batch_size: int = 8, seed: int = SEED,
                       n_channels: int = 8) -> np.ndarray:
    """labels (N, L) -> (N, n_channels, 1000) in the model's own units/channel order. Full T-step DDPM."""
    dh = hyperparams_to_device(calc_diffusion_hyperparams(**DIFFUSION), device)
    net.eval()
    out = []
    for k, s in enumerate(range(0, len(labels), batch_size)):
        torch.manual_seed(seed + k)
        cond = torch.from_numpy(np.ascontiguousarray(labels[s:s + batch_size])).float().to(device)
        x = sampling_label(net, (len(cond), n_channels, 1000), dh, cond=cond)
        out.append(x.cpu().numpy())
        print(f"  sampled {min(s + batch_size, len(labels))}/{len(labels)}", flush=True)
    return np.concatenate(out)


def model_layout(model_channels, scale_to_mV: float) -> dict:
    return {"axis": "channels_first", "channels": list(model_channels), "units": "model", "scale_to_mV": float(scale_to_mV)}


def write_pool(out_dir: Path, x_z: np.ndarray, y: np.ndarray, extra: pd.DataFrame, generator: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_synthetic.npy", x_z.astype(np.float32))
    np.save(out_dir / "y_synthetic.npy", y.astype(np.int64))
    meta = extra.copy()
    meta.insert(0, "sample_id", [f"syn_{i:06d}" for i in range(len(y))])
    meta["class_id"] = y
    meta["class_name"] = [CLASS_ORDER[c] for c in y]
    meta["generator"] = generator
    meta["sampling_rate"] = 100
    meta["num_leads"] = 12
    meta.to_csv(out_dir / "metadata.csv", index=False)


def train_label_vectors(paths, label_order, n_per_class: int, seed: int):
    """Draw label vectors from our train records of each class (with replacement if needed)."""
    arrays = OurArrays(paths)
    db = load_database(paths)
    tr = arrays.meta[arrays.meta["split"] == "train"]
    rng = np.random.default_rng(seed)
    ids, ys = [], []
    for c in range(len(CLASS_ORDER)):
        pool = tr.loc[tr["label"] == c, "ecg_id"].to_numpy()
        ids.append(rng.choice(pool, n_per_class, replace=len(pool) < n_per_class))
        ys.append(np.full(n_per_class, c))
    ids, ys = np.concatenate(ids), np.concatenate(ys)
    return label_vectors(db, ids, label_order), ys, ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n-per-class", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    ck = safe_load(args.ckpt)
    net = build_net(ck["wavenet_config"])
    rep = load_checked(net, ck["model_state_dict"])
    if not rep["ok"]:
        raise SystemExit(f"checkpoint does not match the network: {rep}")
    device = pick_device(args.device)
    net.to(device)
    labels, y, src = train_label_vectors(paths, list(ck["label_order"]), args.n_per_class, SEED)
    x8 = sample_from_labels(net, labels, device, args.batch_size, SEED, len(ck["model_channels"]))
    mean, std = norm_stats(paths)
    x_z = foreign_to_z(x8, model_layout(ck["model_channels"], ck["scale_to_mV"]), mean, std)
    out = args.out or paths.out_root / "ours_pool"
    write_pool(out, x_z, y, pd.DataFrame({"label_source_ecg_id": src, "ckpt_iter": int(ck["iter"])}),
               "SSSD-ECG 36x256 retrained on our train split")
    (out / "sample_config.json").write_text(json.dumps({"ckpt": str(args.ckpt), "iter": int(ck["iter"]),
                                                        "n_per_class": args.n_per_class, "seed": SEED}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
