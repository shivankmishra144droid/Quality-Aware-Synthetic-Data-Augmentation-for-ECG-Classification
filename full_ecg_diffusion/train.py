"""Phase 5: retrain SSSD-ECG with the paper recipe; only the training data changes.

Data = our train split (no val/test patient, so none of judge_reserved), put into the published
model's channels and units, conditioned on 71-label multi-hot SCP vectors in the published order.
Recipe = paper: 36x256, T=200, Adam 2e-4, batch 6, 100k iterations, checkpoint every 4000.
Micro-batching (gradient accumulation) and gradient checkpointing only change memory, not maths.

  # 1) build the training bundle once (also what you upload to Kaggle)
  python -m full_ecg_diffusion.train --export-bundle generator_comparison/sssd_full/bundle
  # 2) train (resumes automatically from the newest checkpoint in --out)
  python -m full_ecg_diffusion.train --bundle generator_comparison/sssd_full/bundle --micro-batch 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from synthetic_generation.utils import calc_diffusion_hyperparams, hyperparams_to_device, training_loss_label

from .config import (DIFFUSION, SEED, SSSD_MODEL_CHANNELS_DEFAULT, TRAINS, WAVENETS, add_path_args,
                     paths_from_args)
from .convert import z_to_foreign
from .nets import build_net, n_params, safe_load
from .ptbxl import OurArrays, label_vectors, load_database, norm_stats


def export_bundle(paths, out: Path, model_channels: list[str]) -> dict:
    insp = json.loads(paths.inspection.read_text(encoding="utf-8"))
    label_order = insp["label_order"]
    scale = float(insp["layout"]["scale_to_mV"])
    arrays = OurArrays(paths)
    tr = arrays.meta[arrays.meta["split"] == "train"]
    ids = tr["ecg_id"].to_numpy()
    x_z, y, _ = arrays.get(ids)
    lay = {"axis": "channels_first", "channels": model_channels, "scale_to_mV": scale}
    mean, std = norm_stats(paths)
    x = z_to_foreign(x_z, lay, mean, std)
    labels = label_vectors(load_database(paths), ids, label_order)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "labels.npy", labels)
    np.save(out / "ecg_ids.npy", ids)
    np.save(out / "y4.npy", y)
    meta = {"source": "our train split only (patient-level; excludes all val/test patients)",
            "n": int(len(ids)), "model_channels": model_channels, "scale_to_mV": scale,
            "units": insp["layout"]["units"], "label_order": label_order,
            "x_sha256": hashlib.sha256((out / "x.npy").read_bytes()).hexdigest()}
    (out / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"bundle: {x.shape} labels {labels.shape} -> {out}")
    return meta


def latest_ckpt(out: Path, tag: str):
    best, it = None, -1
    for p in out.glob(f"{tag}_*.pt"):
        m = re.fullmatch(rf"{tag}_(\d+)\.pt", p.name)
        if m and int(m.group(1)) > it:
            best, it = p, int(m.group(1))
    return best


def train(bundle: Path, out: Path, preset: str, device: torch.device, micro_batch: int | None,
          grad_ckpt: bool, resume: Path | None = None, n_iters: int | None = None, tag: str = "ours") -> Path:
    meta = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    x = np.load(bundle / "x.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(bundle / "labels.npy", allow_pickle=False)
    cfg = dict(TRAINS[preset])
    if n_iters:
        cfg["n_iters"] = n_iters
    wavenet = WAVENETS[preset]
    bs = cfg["batch_size"]
    mb = micro_batch or bs
    if bs % mb:
        raise SystemExit(f"--micro-batch {mb} must divide batch size {bs}")

    torch.manual_seed(SEED)
    net = build_net(wavenet, checkpointing=grad_ckpt).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(SEED)
    it = 0
    out.mkdir(parents=True, exist_ok=True)
    resume = resume or latest_ckpt(out, tag)
    if resume is not None:
        ck = safe_load(resume)
        net.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        it = int(ck["iter"])
        rng = np.random.default_rng([SEED, it])
        torch.manual_seed(SEED + it)
        print(f"resumed {resume} at iter {it}")

    dh = hyperparams_to_device(calc_diffusion_hyperparams(**DIFFUSION), device)
    loss_fn = nn.MSELoss()
    hist_path = out / "history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else {"log": []}
    hist.update(preset=preset, wavenet=wavenet, train_cfg=cfg, micro_batch=mb, grad_checkpointing=grad_ckpt,
                n_params=n_params(net), bundle=str(bundle), bundle_sha256=meta["x_sha256"], device=str(device))

    def save(path):
        torch.save({"iter": it, "model_state_dict": net.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "wavenet_config": wavenet, "diffusion_config": DIFFUSION, "train_cfg": cfg,
                    "model_channels": meta["model_channels"], "scale_to_mV": meta["scale_to_mV"],
                    "label_order": meta["label_order"], "data": meta["source"]}, path)

    order, pos = rng.permutation(len(x)), 0  # epoch-wise shuffling, drop_last like the paper's DataLoader
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train()
    while it < cfg["n_iters"]:
        if pos + bs > len(order):
            order, pos = rng.permutation(len(x)), 0
        b = np.sort(order[pos:pos + bs])
        pos += bs
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for s in range(0, bs, mb):
            sb = b[s:s + mb]
            audio = torch.from_numpy(np.asarray(x[sb])).float().to(device)
            lab = torch.from_numpy(labels[sb]).float().to(device)
            loss = training_loss_label(net, loss_fn, audio, lab, dh) * (len(sb) / bs)
            loss.backward()
            tot += float(loss.item())
        opt.step()
        it += 1
        if it % cfg["log_every"] == 0 or it == 1:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
            hist["log"].append({"iter": it, "loss": tot, "sec": time.time() - t0, "peak_gb": peak})
            print(f"iter {it:6d} loss {tot:.4f} {time.time() - t0:8.0f}s peak {peak:.2f} GB", flush=True)
            hist_path.write_text(json.dumps(hist, indent=1), encoding="utf-8")
        if it % cfg["ckpt_every"] == 0:
            save(out / f"{tag}_{it}.pt")
    last = out / f"{tag}_last.pt"
    save(last)
    hist_path.write_text(json.dumps(hist, indent=1), encoding="utf-8")
    print("done", last)
    return last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--export-bundle", type=Path, default=None, help="write the training bundle here and exit")
    ap.add_argument("--model-channels", default=",".join(SSSD_MODEL_CHANNELS_DEFAULT),
                    help="must equal the published model's channels (confirmed in phase 4a)")
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preset", choices=list(TRAINS), default="paper", help="'tiny' = CPU smoke test only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--n-iters", type=int, default=None, help="override (smoke tests)")
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    if args.export_bundle:
        return export_bundle(paths, args.export_bundle, args.model_channels.split(","))
    if args.bundle is None:
        raise SystemExit("--bundle is required (create it with --export-bundle)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda" and args.preset != "tiny":
        raise SystemExit("paper-size training on CPU is not supported; use a GPU or Kaggle (kaggle_train.ipynb)")
    return train(args.bundle, args.out or paths.out_root / "ours_train", args.preset, device,
                 args.micro_batch, args.grad_checkpointing, args.resume, args.n_iters)


if __name__ == "__main__":
    main()
