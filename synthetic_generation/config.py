"""SSSD-ECG-adapted config. Official source: https://github.com/AI4HealthUOL/SSSD-ECG

Adaptation vs official config_SSSD_ECG.json:
- label_embed_classes: 71 -> 4  (rhythm classes, not SCP multi-label)
- in/out channels stay 8
- T, betas, S4 widths stay official unless a run OOMs
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "processed_data"
OUT_DIR = PROJECT_ROOT / "synthetic_data"

GENERATOR_NAME = "SSSD-ECG-adapted"
GENERATOR_SOURCE = "https://github.com/AI4HealthUOL/SSSD-ECG"
GENERATOR_COMMIT_NOTE = "official SSSD_ECG.py + S4Layer; 4-class one-hot conditioning"

SEED = 42
CLASS_ORDER = ["SR", "AFIB", "STACH", "SBRAD"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}

DIFFUSION_CONFIG = {
    "T": 200,
    "beta_0": 0.0001,
    "beta_T": 0.02,
}

# Official wavenet_config except label_embed_classes
WAVENET_CONFIG = {
    "in_channels": 8,
    "out_channels": 8,
    "num_res_layers": 36,
    "res_channels": 256,
    "skip_channels": 256,
    "diffusion_step_embed_dim_in": 128,
    "diffusion_step_embed_dim_mid": 512,
    "diffusion_step_embed_dim_out": 512,
    "s4_lmax": 1000,
    "s4_d_state": 64,
    "s4_dropout": 0.0,
    "s4_bidirectional": 1,
    "s4_layernorm": 1,
    "label_embed_dim": 128,
    "label_embed_classes": 4,
}

TRAIN_PILOT = {
    "n_iters": 200,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 20,
    "ckpt_every": 200,
    "use_amp": False,
}

TRAIN_FULL = {
    "n_iters": 4000,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "log_every": 50,
    "ckpt_every": 1000,
    "use_amp": False,
}

# Official 36 x 256 does not fit 4 GB; used only if a probe OOM's.
WAVENET_4GB = {
    **WAVENET_CONFIG,
    "num_res_layers": 8,
    "res_channels": 64,
    "skip_channels": 64,
}

# Pilot synthetic pool (not full targets)
PILOT_SAMPLES_PER_CLASS = {
    "SR": 16,
    "AFIB": 32,
    "STACH": 32,
    "SBRAD": 48,
}
