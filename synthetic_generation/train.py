"""Train adapted SSSD-ECG from scratch on X_train/y_train only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import DIFFUSION_CONFIG, ROOT, SEED, WAVENET_4GB, WAVENET_CONFIG
from .data import TrainECG8
from .model import SSSD_ECG
from .utils import (
    calc_diffusion_hyperparams,
    count_params,
    hyperparams_to_device,
    set_seed,
    training_loss_label,
)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Full SSSD training must not run on CPU. "
            f"torch={torch.__version__}. Install a CUDA build of PyTorch, then retry."
        )
    return torch.device("cuda")


def select_wavenet() -> dict:
    """Official 36x256 peaks ~8.2 GB in a probe on this 4 GB laptop (shared memory)."""
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram < 8.0:
        print(f"GPU VRAM {vram:.1f} GB; using WAVENET_4GB (8x64). Official 36x256 needs ~8 GB.")
        return WAVENET_4GB
    print(f"GPU VRAM {vram:.1f} GB; using official 36x256.")
    return WAVENET_CONFIG


def train_loop(
    x12,
    y,
    train_cfg: dict,
    tag: str,
    wavenet_config: dict | None = None,
    resume_path=None,
    ckpt_dir=None,
    history_path=None,
):
    device = require_cuda()
    set_seed(SEED)
    wavenet_config = wavenet_config or select_wavenet()
    ds = TrainECG8(x12, y)
    if train_cfg.get("balanced_sampling", False):
        # Equal expected draws per class (with replacement); default off keeps v1..cfg runs reproducible.
        yi = np.asarray(y).astype(np.int64)
        w = 1.0 / np.bincount(yi, minlength=4)[yi]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(yi), replacement=True)
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"], sampler=sampler, drop_last=True)
    else:
        loader = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)

    dh = hyperparams_to_device(calc_diffusion_hyperparams(**DIFFUSION_CONFIG), device)
    n_iter = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        used = ckpt["wavenet_config"]
        net = SSSD_ECG(**used).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        opt = torch.optim.Adam(net.parameters(), lr=train_cfg["learning_rate"])
        if ckpt.get("optimizer_state_dict") is not None:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        n_iter = int(ckpt["iter"])
        print("resume", resume_path, "from iter", n_iter)
    else:
        try:
            net = SSSD_ECG(**wavenet_config).to(device)
            _probe = torch.zeros(train_cfg["batch_size"], 8, 1000, device=device)
            _lab = torch.zeros(train_cfg["batch_size"], 4, device=device)
            _lab[:, 0] = 1
            with torch.amp.autocast("cuda", enabled=bool(train_cfg.get("use_amp", True))):
                _ = training_loss_label(net, nn.MSELoss(), _probe, _lab, dh)
            torch.cuda.empty_cache()
            used = wavenet_config
            print("using wavenet", {k: used[k] for k in ("num_res_layers", "res_channels", "in_channels", "label_embed_classes")})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if wavenet_config is WAVENET_CONFIG:
                print("OOM on official 36x256; falling back to WAVENET_4GB (8 layers, 64 channels). T=200 unchanged.")
                return train_loop(x12, y, train_cfg, tag, WAVENET_4GB)
            raise
        opt = torch.optim.Adam(net.parameters(), lr=train_cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    use_amp = bool(train_cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()
    peak_mem = 0.0
    torch.cuda.reset_peak_memory_stats()
    net.train()
    data_iter = iter(loader)

    while n_iter < train_cfg["n_iters"]:
        try:
            audio, label = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            audio, label = next(data_iter)
        audio = audio.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = training_loss_label(
                net, loss_fn, audio, label, dh,
                label_dropout_p=float(train_cfg.get("label_dropout_p", 0.0)),
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        n_iter += 1
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 ** 3))
        if n_iter % train_cfg["log_every"] == 0 or n_iter == 1:
            rec = {"iter": n_iter, "loss": float(loss.item()), "sec": time.time() - t0, "peak_gb": peak_mem}
            history.append(rec)
            print(f"{tag} iter {n_iter:5d}  loss {rec['loss']:.4f}  peak {peak_mem:.2f} GB")
        if n_iter % train_cfg["ckpt_every"] == 0:
            path = ckpt_dir / f"{tag}_{n_iter}.pt"
            torch.save({"iter": n_iter, "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": opt.state_dict(), "train_cfg": train_cfg,
                        "wavenet_config": used, "diffusion_config": DIFFUSION_CONFIG}, path)
            print("saved", path)

    last = ckpt_dir / f"{tag}_last.pt"
    torch.save({"iter": n_iter, "model_state_dict": net.state_dict(),
                "optimizer_state_dict": opt.state_dict(), "train_cfg": train_cfg,
                "wavenet_config": used, "diffusion_config": DIFFUSION_CONFIG}, last)
    hist_path = Path(history_path) if history_path is not None else ROOT / "history.json"
    payload = {"tag": tag, "n_params": count_params(net), "peak_gb": peak_mem,
               "wavenet_config": used, "history": history}
    if hist_path.exists():
        prev = json.loads(hist_path.read_text(encoding="utf-8"))
        if not isinstance(prev, dict):
            prev = {}
        prev[tag] = payload
        hist_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    else:
        hist_path.write_text(json.dumps({tag: payload}, indent=2), encoding="utf-8")
    return net, history, peak_mem, last
