"""Official SSSD_ECG.py (AI4HealthUOL/SSSD-ECG) with local imports.

Only intended change vs official residual/S4/diffusion structure:
label embedding size is set by config (4 classes instead of 71).
"""
import math

import torch
import torch.nn as nn

from .s4 import S4Layer
from .utils import calc_diffusion_step_embedding


def swish(x):
    return x * torch.sigmoid(x)


def _weight_norm(conv):
    # Official uses nn.utils.weight_norm
    if hasattr(nn.utils, "weight_norm"):
        return nn.utils.weight_norm(conv)
    return nn.utils.parametrizations.weight_norm(conv)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = _weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class Residual_block(nn.Module):
    def __init__(
        self, res_channels, skip_channels, diffusion_step_embed_dim_out, in_channels,
        s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm, label_embed_dim=None,
    ):
        super().__init__()
        self.res_channels = res_channels
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.S41 = S4Layer(
            features=2 * self.res_channels, lmax=s4_lmax, N=s4_d_state,
            dropout=s4_dropout, bidirectional=s4_bidirectional, layer_norm=s4_layernorm,
        )
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)
        self.S42 = S4Layer(
            features=2 * self.res_channels, lmax=s4_lmax, N=s4_d_state,
            dropout=s4_dropout, bidirectional=s4_bidirectional, layer_norm=s4_layernorm,
        )
        self.res_conv = _weight_norm(nn.Conv1d(res_channels, res_channels, kernel_size=1))
        nn.init.kaiming_normal_(self.res_conv.weight)
        self.skip_conv = _weight_norm(nn.Conv1d(res_channels, skip_channels, kernel_size=1))
        nn.init.kaiming_normal_(self.skip_conv.weight)
        self.fc_label = nn.Linear(label_embed_dim, 2 * self.res_channels) if label_embed_dim is not None else None

    def forward(self, input_data):
        x, label_embed, diffusion_step_embed = input_data
        h = x
        part_t = self.fc_t(diffusion_step_embed).view([x.shape[0], self.res_channels, 1])
        h = h + part_t
        h = self.conv_layer(h)
        h = self.S41(h.permute(2, 0, 1)).permute(1, 2, 0)
        if self.fc_label is not None:
            label_embed = self.fc_label(label_embed).unsqueeze(2)
            h = h + label_embed
        h = self.S42(h.permute(2, 0, 1)).permute(1, 2, 0)
        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels :, :])
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(
        self, res_channels, skip_channels, num_res_layers,
        diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out,
        in_channels, s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm, label_embed_dim=None,
    ):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.residual_blocks = nn.ModuleList([
            Residual_block(
                res_channels, skip_channels, diffusion_step_embed_dim_out, in_channels,
                s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm, label_embed_dim,
            )
            for _ in range(num_res_layers)
        ])

    def forward(self, input_data):
        noise, label_embed, diffusion_steps = input_data
        t_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        t_embed = swish(self.fc_t1(t_embed))
        t_embed = swish(self.fc_t2(t_embed))
        h = noise
        skip = 0
        for block in self.residual_blocks:
            h, skip_n = block((h, label_embed, t_embed))
            skip = skip + skip_n
        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSD_ECG(nn.Module):
    def __init__(
        self, in_channels, res_channels, skip_channels, out_channels, num_res_layers,
        diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out,
        s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm,
        label_embed_classes=0, label_embed_dim=128,
    ):
        super().__init__()
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        self.embedding = nn.Embedding(label_embed_classes, label_embed_dim) if label_embed_classes > 0 else None
        self.residual_layer = Residual_group(
            res_channels, skip_channels, num_res_layers,
            diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out,
            in_channels, s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm,
            label_embed_dim if label_embed_classes > 0 else None,
        )
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels),
        )

    def forward(self, input_data):
        noise, label, diffusion_steps = input_data
        label_embed = label @ self.embedding.weight if self.embedding is not None else None
        x = self.init_conv(noise)
        x = self.residual_layer((x, label_embed, diffusion_steps))
        return self.final_conv(x)
