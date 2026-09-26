"""Train adapted SSSD-ECG from scratch on X_train/y_train only."""
from __future__ import annotations

import json
import math
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
    eval_fn=None,
):
    """eval_fn(net, ema_or_None, n_iter) -> dict, called every train_cfg["eval_every"] steps."""
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

    diffusion_config = train_cfg.get("diffusion_config", DIFFUSION_CONFIG)
    dh = hyperparams_to_device(calc_diffusion_hyperparams(**diffusion_config), device)
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

    # Optional knobs, all default off so earlier runs are unchanged.
    ema_decay = train_cfg.get("ema_decay")
    ema = None
    if ema_decay:  # deepcopy fails on weight_norm modules; rebuild and copy weights instead
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        ema = SSSD_ECG(**used).to(device)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)  # keep data order / noise identical to a no-EMA run
        ema.load_state_dict(net.state_dict())
        ema.eval().requires_grad_(False)
    if ema is not None and resume_path is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])
    accum = int(train_cfg.get("grad_accum", 1))
    base_lr = train_cfg["learning_rate"]
    warmup = int(train_cfg.get("warmup_iters", 0))
    cosine = bool(train_cfg.get("cosine_decay", False))

    def lr_at(it):
        if warmup and it < warmup:
            return base_lr * (it + 1) / warmup
        if cosine:
            frac = (it - warmup) / max(1, train_cfg["n_iters"] - warmup)
            return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        return base_lr

    def state(it):
        d = {"iter": it, "model_state_dict": net.state_dict(),
             "optimizer_state_dict": opt.state_dict(), "train_cfg": train_cfg,
             "wavenet_config": used, "diffusion_config": diffusion_config}
        if ema is not None:
            d["ema_state_dict"] = ema.state_dict()
        return d

    history = []
    evals = []
    t0 = time.time()
    peak_mem = 0.0
    torch.cuda.reset_peak_memory_stats()
    net.train()
    data_iter = iter(loader)

    while n_iter < train_cfg["n_iters"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(n_iter)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            try:
                audio, label = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                audio, label = next(data_iter)
            audio = audio.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = training_loss_label(
                    net, loss_fn, audio, label, dh,
                    label_dropout_p=float(train_cfg.get("label_dropout_p", 0.0)),
                    parameterization=train_cfg.get("parameterization", "eps"),
                )
            scaler.scale(loss / accum).backward()
            loss_sum += float(loss.item())
        loss = torch.tensor(loss_sum / accum)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            with torch.no_grad():
                d = ema_decay
                if train_cfg.get("ema_warmup", False):  # standard (1+n)/(10+n) ramp; flushes random init early
                    d = min(ema_decay, (1 + n_iter) / (10 + n_iter))
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1.0 - d)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
        n_iter += 1
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 ** 3))
        if n_iter % train_cfg["log_every"] == 0 or n_iter == 1:
            rec = {"iter": n_iter, "loss": float(loss.item()), "sec": time.time() - t0, "peak_gb": peak_mem,
                   "lr": opt.param_groups[0]["lr"],
                   "device_used_gb": (lambda f, t: (t - f) / 1024 ** 3)(*torch.cuda.mem_get_info())}
            history.append(rec)
            print(f"{tag} iter {n_iter:5d}  loss {rec['loss']:.4f}  peak {peak_mem:.2f} GB  device {rec['device_used_gb']:.2f} GB", flush=True)
        if eval_fn is not None and n_iter % train_cfg.get("eval_every", 10 ** 9) == 0:
            net.eval()
            rec = {"iter": n_iter, **eval_fn(net, ema, n_iter)}
            net.train()
            evals.append(rec)
            print(f"{tag} EVAL {json.dumps(rec)}", flush=True)
        if n_iter % train_cfg["ckpt_every"] == 0:
            path = ckpt_dir / f"{tag}_{n_iter}.pt"
            torch.save(state(n_iter), path)
            print("saved", path)

    last = ckpt_dir / f"{tag}_last.pt"
    torch.save(state(n_iter), last)
    hist_path = Path(history_path) if history_path is not None else ROOT / "history.json"
    payload = {"tag": tag, "n_params": count_params(net), "peak_gb": peak_mem,
               "wavenet_config": used, "history": history, "evals": evals}
    if hist_path.exists():
        prev = json.loads(hist_path.read_text(encoding="utf-8"))
        if not isinstance(prev, dict):
            prev = {}
        prev[tag] = payload
        hist_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    else:
        hist_path.write_text(json.dumps({tag: payload}, indent=2), encoding="utf-8")
    return net, history, peak_mem, last
