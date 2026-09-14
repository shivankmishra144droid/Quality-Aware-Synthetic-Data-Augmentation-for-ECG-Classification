"""Diffusion helpers. Official SSSD util.py used .cuda(); we take an explicit device."""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_diffusion_hyperparams(T: int, beta_0: float, beta_T: float) -> dict:
    beta = torch.linspace(beta_0, beta_T, T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    beta_tilde = beta.clone()
    for t in range(1, T):
        beta_tilde[t] = beta[t] * (1 - alpha_bar[t - 1]) / (1 - alpha_bar[t])
    sigma = torch.sqrt(beta_tilde)
    return {
        "T": T,
        "Beta": beta,
        "Alpha": alpha,
        "Alpha_bar": alpha_bar,
        "Sigma": sigma,
    }


def hyperparams_to_device(dh: dict, device: torch.device) -> dict:
    out = {}
    for k, v in dh.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def calc_diffusion_step_embedding(diffusion_steps: torch.Tensor, dim_in: int) -> torch.Tensor:
    assert dim_in % 2 == 0
    half = dim_in // 2
    device = diffusion_steps.device
    embed = math.log(10000) / (half - 1)
    embed = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -embed)
    embed = diffusion_steps * embed
    return torch.cat((torch.sin(embed), torch.cos(embed)), dim=1)


def std_normal(size, device: torch.device) -> torch.Tensor:
    return torch.randn(size, device=device)


def training_loss_label(net, loss_fn, audio, label, dh: dict, label_dropout_p: float = 0.0):
    """Official training_loss_label + optional CFG label dropout. audio (B,8,1000), label (B,4)."""
    T, alpha_bar = dh["T"], dh["Alpha_bar"]
    b = audio.shape[0]
    device = audio.device
    t = torch.randint(T, size=(b, 1, 1), device=device)
    z = std_normal(audio.shape, device)
    x_t = torch.sqrt(alpha_bar[t]) * audio + torch.sqrt(1 - alpha_bar[t]) * z
    if label_dropout_p > 0:
        drop = torch.rand(b, device=device) < label_dropout_p
        label = label.clone()
        label[drop] = 0.0  # null one-hot -> zero embedding
    eps = net((x_t, label, t.view(b, 1)))
    return loss_fn(eps, z)


@torch.no_grad()
def sampling_label(net, size, dh: dict, cond: torch.Tensor, guidance_w: float | None = None) -> torch.Tensor:
    """Reverse diffusion. size = (B, 8, 1000). cond = (B, 4) float one-hot.
    If guidance_w is set, classifier-free guidance:
    eps = eps_uncond + w * (eps_cond - eps_uncond). Null cond is the zero vector.
    """
    T, alpha, alpha_bar, sigma = dh["T"], dh["Alpha"], dh["Alpha_bar"], dh["Sigma"]
    device = cond.device
    x = std_normal(size, device)
    uncond = torch.zeros_like(cond) if guidance_w is not None else None
    for t in range(T - 1, -1, -1):
        steps = torch.full((size[0], 1), t, device=device, dtype=torch.float32)
        eps = net((x, cond, steps))
        if guidance_w is not None:
            eps_u = net((x, uncond, steps))
            eps = eps_u + guidance_w * (eps - eps_u)
        x = (x - (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t]) * eps) / torch.sqrt(alpha[t])
        if t > 0:
            x = x + sigma[t] * std_normal(size, device)
    return x


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
