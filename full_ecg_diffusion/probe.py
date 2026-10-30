"""Phase 4b: time the paper-config training step on this GPU and project 100k iterations.

Tries micro-batches 6/3/2/1 (gradient accumulation to batch 6), with and without gradient
checkpointing. Recommends the fastest config whose peak stays under 90% of VRAM (above that,
Windows spills into shared memory and slows ~20x, as the Week-4 16x128 run did).

  python -m full_ecg_diffusion.probe --budget-hours 96
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from synthetic_generation.utils import calc_diffusion_hyperparams, hyperparams_to_device, training_loss_label

from .config import DIFFUSION, TRAINS, WAVENETS, add_path_args, paths_from_args
from .nets import build_net, n_params
from .report import write_json


def time_config(preset, micro, ckpt, device, steps):
    wavenet, cfg = WAVENETS[preset], TRAINS[preset]
    bs = cfg["batch_size"]
    net = build_net(wavenet, checkpointing=ckpt).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    dh = hyperparams_to_device(calc_diffusion_hyperparams(**DIFFUSION), device)
    lossf = nn.MSELoss()
    n_lab = wavenet["label_embed_classes"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for k in range(steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(bs // micro):
            x = torch.randn(micro, wavenet["in_channels"], 1000, device=device)
            lab = (torch.rand(micro, n_lab, device=device) < 0.05).float()
            (training_loss_label(net, lossf, x, lab, dh) * (micro / bs)).backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if k:  # first step is warm-up
            times.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")
    del net, opt
    return float(np.median(times)), peak, n_params(build_net(wavenet))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--preset", choices=list(WAVENETS), default="paper")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--budget-hours", type=float, default=96.0)
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if device.type == "cuda" else float("nan")
    n_iters = TRAINS[args.preset]["n_iters"]
    rows = []
    for ckpt in (False, True):
        for micro in (6, 3, 2, 1):
            if TRAINS[args.preset]["batch_size"] % micro:
                continue
            try:
                sec, peak, npar = time_config(args.preset, micro, ckpt, device, args.steps)
                fits = device.type != "cuda" or peak <= 0.9 * vram
                rows.append({"micro_batch": micro, "grad_checkpointing": ckpt, "sec_per_iter": sec, "peak_gb": peak,
                             "fits": bool(fits), "projected_hours": sec * n_iters / 3600, "n_params": npar})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"micro_batch": micro, "grad_checkpointing": ckpt, "oom": True})
            print(rows[-1], flush=True)
    ok = [r for r in rows if r.get("fits")]
    best = min(ok, key=lambda r: r["sec_per_iter"]) if ok else None
    rec = ("train locally: " + f"--micro-batch {best['micro_batch']}" + (" --grad-checkpointing" if best["grad_checkpointing"] else "")
           if best and best["projected_hours"] <= args.budget_hours
           else "too slow or does not fit: use full_ecg_diffusion/kaggle_train.ipynb")
    res = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "vram_gb": vram, "preset": args.preset, "n_iters": n_iters, "budget_hours": args.budget_hours,
           "configs": rows, "best": best, "recommendation": rec}
    out = paths.out_root / "probe.json"
    write_json(out, res)
    print(json.dumps({"best": best, "recommendation": rec}, indent=1))
    return res


if __name__ == "__main__":
    main()
