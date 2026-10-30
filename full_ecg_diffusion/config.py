"""Paths, model presets and every pass/fail threshold for the Option-3 phases.

All thresholds live here so the final test (phase 7) can snapshot them into its registration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from synthetic_generation.config import DIFFUSION_CONFIG, WAVENET_CONFIG

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent

SEED = 42
FS = 100
N_T = 1000
CLASS_ORDER = ["SR", "AFIB", "STACH", "SBRAD"]
LEAD_ORDER_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
CHEST_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]

# Channel order of the published SSSD-ECG network output, from their inference.py
# (index_8 = [0,2,3,4,5,6,7,11] over their stored 12-lead order). Phase 1 / 4a verify it from data.
SSSD_MODEL_CHANNELS_DEFAULT = ["I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"]

# ---------------------------------------------------------------- paths


@dataclass
class Paths:
    data_dir: Path = PROJECT_ROOT / "processed_data"
    ptbxl_dir: Path = PROJECT_ROOT / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
    public_dir: Path = PROJECT_ROOT / "archive" / "sssd_ecg_public"
    manifest: Path = PKG_ROOT / "manifests" / "judge_manifest.json"
    out_root: Path = PROJECT_ROOT / "generator_comparison" / "sssd_full"
    # Converted published samples must never feed a classifier, so they stay under gitignored archive/
    public_converted: Path = PROJECT_ROOT / "archive" / "sssd_ecg_public" / "converted"
    inspection: Path = PROJECT_ROOT / "generator_comparison" / "sssd_full" / "public_inspection" / "inspection.json"
    final_dir: Path = PKG_ROOT / "final"

    @property
    def ptbxl_database(self) -> Path:
        return self.ptbxl_dir / "ptbxl_database.csv"

    @property
    def scp_statements(self) -> Path:
        return self.ptbxl_dir / "scp_statements.csv"


def add_path_args(ap: argparse.ArgumentParser) -> None:
    d = Paths()
    ap.add_argument("--data-dir", type=Path, default=d.data_dir, help="processed_data/ (X_*.npy, *_metadata.csv)")
    ap.add_argument("--ptbxl-dir", type=Path, default=d.ptbxl_dir, help="raw PTB-XL 1.0.1 folder")
    ap.add_argument("--public-dir", type=Path, default=d.public_dir, help="downloaded SSSD-ECG files")
    ap.add_argument("--manifest", type=Path, default=d.manifest)
    ap.add_argument("--out-root", type=Path, default=d.out_root)
    ap.add_argument("--public-converted", type=Path, default=d.public_converted)
    ap.add_argument("--inspection", type=Path, default=d.inspection)
    ap.add_argument("--final-dir", type=Path, default=d.final_dir)


def paths_from_args(args) -> Paths:
    return Paths(
        data_dir=args.data_dir, ptbxl_dir=args.ptbxl_dir, public_dir=args.public_dir,
        manifest=args.manifest, out_root=args.out_root, public_converted=args.public_converted,
        inspection=args.inspection, final_dir=args.final_dir,
    )


# ---------------------------------------------------------------- models

# Published SSSD-ECG: 36 x 256, 71-label multi-hot conditioning, T=200.
PAPER_WAVENET = {**WAVENET_CONFIG, "label_embed_classes": 71}
# Smoke-test only (CPU self-test). Never used for a result.
TINY_WAVENET = {**PAPER_WAVENET, "num_res_layers": 2, "res_channels": 16, "skip_channels": 16,
                "s4_d_state": 8, "diffusion_step_embed_dim_mid": 64, "diffusion_step_embed_dim_out": 64,
                "label_embed_dim": 16}
WAVENETS = {"paper": PAPER_WAVENET, "tiny": TINY_WAVENET}

DIFFUSION = dict(DIFFUSION_CONFIG)  # T=200, beta 1e-4 -> 0.02, same as the paper

# Paper train_config: Adam 2e-4, 100k iters, batch 6 (train.py hardcodes 6), ckpt every 4000.
PAPER_TRAIN = {"n_iters": 100_000, "batch_size": 6, "learning_rate": 2e-4,
               "ckpt_every": 4000, "log_every": 100}
TINY_TRAIN = {"n_iters": 3, "batch_size": 2, "learning_rate": 2e-4, "ckpt_every": 2, "log_every": 1}
TRAINS = {"paper": PAPER_TRAIN, "tiny": TINY_TRAIN}

# ---------------------------------------------------------------- thresholds


@dataclass
class Thresholds:
    # Detector (real vs synthetic AUC)
    detector_pass: float = 0.70
    detector_fail: float = 0.95
    control_max: float = 0.60          # real-vs-real and conversion controls must stay below this
    # Conversion control (phase 2), z units
    exact_max_abs: float = 1e-4        # channels present in the published layout
    derived_median_abs: float = 0.02   # limb leads rebuilt from two limb leads (PTB-XL quantisation)
    derived_min_corr: float = 0.995
    # Feature judge (KS distance, pooled over classes)
    feature_median_ks: float = 0.15
    feature_max_ks: float = 0.30
    # Rhythm checks
    detectable_ratio: float = 0.95
    hr_tol_bpm: float = 10.0
    sbrad_max_hr: float = 60.0
    stach_min_hr: float = 100.0
    afib_irregularity_ratio: float = 0.5   # syn (AFIB/SR rr_cv) must reach this share of the real ratio
    # Copying
    copy_corr: float = 0.95
    copy_frac_margin: float = 0.01
    copy_mean_margin: float = 0.05
    # Phase-1 sanity gate
    phase1_label_rank_corr: float = 0.8

    def as_dict(self) -> dict:
        return dict(self.__dict__)


THRESHOLDS = Thresholds()

# Evaluation sizes (CPU friendly)
EVAL_MAX_PER_CLASS = 1000
DETECTOR = {"folds": 5, "epochs": 12, "batch_size": 64, "lr": 1e-3, "bootstrap": 1000}
COPY_REF_MAX = 6000

SEALED_PART = "judge_reserved"
