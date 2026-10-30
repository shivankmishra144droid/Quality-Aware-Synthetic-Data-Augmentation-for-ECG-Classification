"""Build the official SSSD-ECG network (imported, unmodified) and load weights safely.

`checkpointing=True` swaps in gradient checkpointing per residual block: same parameters, same
state_dict keys and same maths, only less activation memory (for small GPUs).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from synthetic_generation.model import SSSD_ECG, Residual_group, swish
from synthetic_generation.utils import calc_diffusion_step_embedding


class CheckpointedResidualGroup(Residual_group):
    def forward(self, input_data):
        noise, label_embed, diffusion_steps = input_data
        t_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        t_embed = swish(self.fc_t1(t_embed))
        t_embed = swish(self.fc_t2(t_embed))
        h = noise
        skip = 0
        for block in self.residual_blocks:
            if self.training and torch.is_grad_enabled():
                h, skip_n = checkpoint(lambda a, b, c, blk=block: blk((a, b, c)), h, label_embed, t_embed,
                                       use_reentrant=False)
            else:
                h, skip_n = block((h, label_embed, t_embed))
            skip = skip + skip_n
        return skip * math.sqrt(1.0 / self.num_res_layers)


def _materialize_buffers(net: nn.Module) -> None:
    """S4 registers B/P/w as broadcast (stride-0) views; recent torch refuses to load_state_dict
    into overlapping memory. A contiguous copy holds identical values, so the maths is unchanged."""
    for mod in net.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_contiguous():
                mod._buffers[name] = buf.contiguous()


def build_net(wavenet: dict, checkpointing: bool = False) -> SSSD_ECG:
    net = SSSD_ECG(**wavenet)
    _materialize_buffers(net)
    if checkpointing:
        net.residual_layer.__class__ = CheckpointedResidualGroup
    return net


def safe_load(path: Path) -> dict:
    """torch.load with weights_only=True: tensors and plain containers only, no arbitrary code."""
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    return obj


def state_dict_of(obj: dict) -> dict:
    return obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj


def _remap_weight_norm(sd: dict, target_keys: set) -> dict:
    """Old nn.utils.weight_norm keys (weight_g/weight_v) <-> parametrizations (original0/original1)."""
    out = {}
    for k, v in sd.items():
        if k in target_keys:
            out[k] = v
            continue
        alt = None
        if k.endswith(".weight_g"):
            alt = k[:-len(".weight_g")] + ".parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            alt = k[:-len(".weight_v")] + ".parametrizations.weight.original1"
        elif k.endswith(".parametrizations.weight.original0"):
            alt = k[:-len(".parametrizations.weight.original0")] + ".weight_g"
        elif k.endswith(".parametrizations.weight.original1"):
            alt = k[:-len(".parametrizations.weight.original1")] + ".weight_v"
        out[alt if alt in target_keys else k] = v
    return out


def load_checked(net: SSSD_ECG, sd: dict) -> dict:
    """Strict load with a readable report of missing/unexpected keys and shape mismatches."""
    own = net.state_dict()
    sd = _remap_weight_norm(sd, set(own))
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    shape_bad = sorted(k for k in set(own) & set(sd) if tuple(own[k].shape) != tuple(sd[k].shape))
    report = {"n_model_keys": len(own), "n_ckpt_keys": len(sd), "missing": missing[:20], "n_missing": len(missing),
              "unexpected": unexpected[:20], "n_unexpected": len(unexpected),
              "shape_mismatch": [(k, list(own[k].shape), list(sd[k].shape)) for k in shape_bad[:20]],
              "n_shape_mismatch": len(shape_bad)}
    report["ok"] = not (missing or unexpected or shape_bad)
    if report["ok"]:
        net.load_state_dict(sd, strict=True)
    return report


def n_params(net) -> int:
    return sum(p.numel() for p in net.parameters())
