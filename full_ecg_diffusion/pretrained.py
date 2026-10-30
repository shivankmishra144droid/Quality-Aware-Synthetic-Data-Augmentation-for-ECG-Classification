"""Phase 4a: load the published SSSD-ECG weights and reproduce their published samples.

Download (manually, you approve it) the model from figshare DOI 10.6084/m9.figshare.21922875
(https://figshare.com/s/81834b24a4711c2a5c55) into archive/sssd_ecg_public/weights/.
Weights are opened with torch.load(weights_only=True) only.

Reproduction = our sampler + their weights + the same label vectors must give samples the
detector cannot tell from their published ones (AUC <= 0.60), with matching features. That proves
our network copy, sampler and conversion are right before we train anything.

  python -m full_ecg_diffusion.pretrained --n-per-class 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .checks.beats import batch_beats
from .checks.detector import run_detector
from .checks.features import feature_check, feature_matrix
from .config import (CLASS_ORDER, SEED, SSSD_MODEL_CHANNELS_DEFAULT, THRESHOLDS, WAVENETS, add_path_args,
                     paths_from_args)
from .convert import foreign_to_z, infer_channels, reference_from_real
from .nets import build_net, load_checked, n_params, safe_load, state_dict_of
from .ptbxl import OurArrays, norm_stats
from .report import write_json
from .sample import model_layout, pick_device, sample_from_labels, write_pool


def find_weights(public_dir: Path) -> Path:
    c = sorted(p for p in public_dir.rglob("*") if p.suffix in {".pkl", ".pt", ".pth"} and p.is_file())
    if not c:
        raise SystemExit(f"no .pkl/.pt weights under {public_dir}; download them first (see module doc)")
    # Prefer the 100000-iteration file the official inference.py loads by default.
    best = [p for p in c if p.stem == "100000"]
    return best[0] if best else c[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--wavenet", choices=list(WAVENETS), default="paper", help="'tiny' is for the self-test only")
    ap.add_argument("--model-channels", default=",".join(SSSD_MODEL_CHANNELS_DEFAULT))
    ap.add_argument("--n-per-class", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny detector budget for smoke tests only")
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    out_dir = paths.out_root / "phase4a_repro"
    out_dir.mkdir(parents=True, exist_ok=True)

    wpath = args.weights or find_weights(paths.public_dir)
    net = build_net(WAVENETS[args.wavenet])
    rep = load_checked(net, state_dict_of(safe_load(wpath)))
    rep.update(weights=str(wpath), n_params=n_params(net), wavenet=args.wavenet)
    print("load", {k: rep[k] for k in ("ok", "n_model_keys", "n_ckpt_keys", "n_missing", "n_unexpected", "n_shape_mismatch")})
    write_json(out_dir / "load_report.json", rep)
    if not rep["ok"]:
        raise SystemExit("weights do not match SSSD_ECG(36x256, 71 labels); see load_report.json")
    if args.load_only:
        return rep

    insp = json.loads(paths.inspection.read_text(encoding="utf-8"))
    conv = paths.public_converted
    x_pub = np.load(conv / "X_synthetic.npy", mmap_mode="r", allow_pickle=False)
    y_pub = np.load(conv / "y_synthetic.npy", allow_pickle=False)
    lab_pub = np.load(conv / "labels71.npy", allow_pickle=False)
    idx = np.concatenate([np.flatnonzero(y_pub == c)[:args.n_per_class] for c in range(len(CLASS_ORDER))])

    device = pick_device(args.device)
    net.to(device)
    model_channels = args.model_channels.split(",")
    x8 = sample_from_labels(net, lab_pub[idx], device, args.batch_size, SEED + 4, len(model_channels))

    # Which leads did the network actually produce? Infer from the samples, report against the default.
    mean, std = norm_stats(paths)
    arrays = OurArrays(paths)
    tr = arrays.meta.loc[arrays.meta["split"] == "train", "ecg_id"].to_numpy()
    x_ref, _, _ = arrays.get(np.random.default_rng(SEED).choice(tr, min(2000, len(tr)), replace=False))
    inferred = infer_channels(x8, reference_from_real(x_ref, mean, std))
    x_z = foreign_to_z(x8, model_layout(model_channels, insp["layout"]["scale_to_mV"]), mean, std)
    y_rep = y_pub[idx]
    pool = conv / "repro_pool"
    write_pool(pool, x_z, y_rep, pd.DataFrame({"label_source_global_row": idx}), "SSSD-ECG published weights, our sampler")

    from .evaluate import QUICK_DETECTOR

    x_p = np.asarray(x_pub[idx], dtype=np.float32)
    det = run_detector(x_p, x_z, np.arange(len(idx)), np.arange(len(idx)), y_rep, y_rep,
                       QUICK_DETECTOR if args.quick else None)
    bp, br = batch_beats(x_p), batch_beats(x_z)
    fp, names = feature_matrix(x_p, bp)
    fr, _ = feature_matrix(x_z, br)
    feats = feature_check(fp, y_rep, fr, y_rep, names, THRESHOLDS)
    res = {
        "weights": str(wpath), "n": int(len(idx)), "model_channels_used": model_channels,
        "model_channels_inferred_from_samples": inferred["channels"], "inference_margin": inferred["margin"],
        "channels_agree": inferred["channels"] == model_channels,
        "detector_published_vs_ours": {"auc": det["auc"], "auc_ci95": det["auc_ci95"]},
        "features_published_vs_ours": {"median_ks": feats["median_ks"], "max_ks": feats["max_ks"]},
        "pass": bool(det["auc"] <= THRESHOLDS.control_max and inferred["channels"] == model_channels),
        "note": "reproduction is distributional (different RNG), not bitwise",
    }
    write_json(out_dir / "repro_report.json", res)
    print(json.dumps({k: res[k] for k in ("channels_agree", "detector_published_vs_ours", "pass")}, default=str))
    print("PHASE 4a", "PASS" if res["pass"] else "FAIL", "->", out_dir / "repro_report.json")
    return res


if __name__ == "__main__":
    main()
