"""Phases 3 and 6: run every realism check on a synthetic pool against real patients.

Phase 3 (published SSSD-ECG samples vs folds 9-10, i.e. patients it never saw):
  python -m full_ecg_diffusion.evaluate --pool archive/sssd_ecg_public/converted \
      --real part:dev_unseen_by_both part:dev_unseen_by_sssd --copy-ref part:sssd_train_ref \
      --layout generator_comparison/sssd_full/public_inspection/inspection.json \
      --out generator_comparison/sssd_full/phase3_public
Phase 6 (our model; real must exclude our train split):
  python -m full_ecg_diffusion.evaluate --pool generator_comparison/sssd_full/ours_pool \
      --real part:dev_unseen_by_both --copy-ref split:train --layout <inspection.json> \
      --out generator_comparison/sssd_full/phase6_ours

A pool dir holds X_synthetic.npy (N, 1000, 12) in our z-space and y_synthetic.npy (N,).
Verdict bands: detector AUC <= 0.70 PASS, >= 0.95 FAIL, in between PARTIAL (stop and ask).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checks.beats import batch_beats, rhythm_check
from .checks.copying import copying_check
from .checks.detector import real_control, run_detector, verdict
from .checks.features import feature_check, feature_matrix
from .config import COPY_REF_MAX, DETECTOR, EVAL_MAX_PER_CLASS, SEED, THRESHOLDS, add_path_args, paths_from_args
from .convert import harmonize_real
from .ptbxl import OurArrays, norm_stats
from .realsets import class_matched, load_real, resolve_ids
from .report import markdown, plots, write_json

QUICK_DETECTOR = {"folds": 3, "epochs": 2, "batch_size": 32, "lr": 1e-3, "bootstrap": 50}


def load_pool(pool_dir: Path):
    x = np.load(pool_dir / "X_synthetic.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(pool_dir / "y_synthetic.npy", allow_pickle=False).astype(int)
    if x.ndim != 3 or x.shape[1:] != (1000, 12) or len(x) != len(y):
        raise ValueError(f"pool {pool_dir}: X {x.shape}, y {y.shape}; expected (N,1000,12) and (N,)")
    return x, y


def evaluate(paths, pool_dir: Path, real_specs, copy_ref_spec: str, out_dir: Path, layout: dict | None,
             title: str, max_per_class: int = EVAL_MAX_PER_CLASS, detector_cfg=None,
             allow_sealed: bool = False, seed: int = SEED, th=THRESHOLDS, notes=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = OurArrays(paths)
    mean, std = norm_stats(paths)
    x_syn_all, y_syn_all = load_pool(pool_dir)
    x_real_all, y_real_all, pid_all, _ = load_real(real_specs, paths, arrays, allow_sealed)

    ir, isyn = class_matched(y_real_all, y_syn_all, max_per_class, seed)
    x_real, y_real, pid = np.asarray(x_real_all[ir]), y_real_all[ir], pid_all[ir]
    x_syn, y_syn = np.asarray(x_syn_all[isyn], dtype=np.float32), y_syn_all[isyn]
    if layout is not None:
        x_real = harmonize_real(x_real, layout, mean, std)
    print(f"[{title}] real {len(x_real)} vs synthetic {len(x_syn)} (class-matched, cap {max_per_class}/class)")

    rb, sb = batch_beats(x_real), batch_beats(x_syn)
    rhythm = rhythm_check(rb, y_real, sb, y_syn, th)
    print(f"  rhythm pass={rhythm['pass']} calibrated={rhythm['detector_calibrated_on_real']}")
    fr, names = feature_matrix(x_real, rb)
    fs, _ = feature_matrix(x_syn, sb)
    feats = feature_check(fr, y_real, fs, y_syn, names, th, seed)
    print(f"  features median KS {feats['median_ks']:.3f} max {feats['max_ks']:.3f}")

    det = run_detector(x_real, x_syn, pid, np.arange(len(x_syn)), y_real, y_syn, detector_cfg, seed)
    print(f"  detector AUC {det['auc']:.3f} CI {det['auc_ci95']}")
    ctrl = real_control(x_real, pid, y_real, detector_cfg, seed)
    print(f"  real-vs-real control AUC {ctrl['auc']:.3f}")

    ref_ids = np.array(resolve_ids([copy_ref_spec], paths, arrays))
    rng = np.random.default_rng(seed)
    ref_ids = rng.choice(ref_ids, min(len(ref_ids), COPY_REF_MAX), replace=False)
    x_ref, _, _ = arrays.get(ref_ids)
    if layout is not None:
        x_ref = harmonize_real(x_ref, layout, mean, std)
    copy = copying_check(x_syn, x_real, x_ref, th)
    print(f"  copying pass={copy['pass']}")

    dv = verdict(det["auc"], th)
    control_ok = ctrl["auc"] <= th.control_max
    all_ok = dv == "PASS" and feats["pass"] and rhythm["pass"] and copy["pass"] and control_ok
    overall = "PASS" if all_ok else ("FAIL" if dv == "FAIL" else ("PARTIAL" if dv == "PARTIAL" else "FAIL_CHECKS"))
    notes = list(notes or [])
    if not control_ok:
        notes.append(f"real-vs-real control AUC {ctrl['auc']:.3f} > {th.control_max}: detector result unreliable")
    if not rhythm["detector_calibrated_on_real"]:
        notes.append("R-peak detector failed its calibration on real ECGs; rhythm verdict is not trustworthy")
    if detector_cfg is not None and detector_cfg != DETECTOR:
        notes.append(f"non-default detector config {detector_cfg} (smoke test only)")
    res = {
        "title": title, "pool": str(pool_dir), "real_specs": list(real_specs), "copy_ref": copy_ref_spec,
        "harmonized": layout is not None, "n_per_class": {int(c): int((y_syn == c).sum()) for c in np.unique(y_syn)},
        "thresholds": th.as_dict(), "detector": det, "control": ctrl, "detector_verdict": dv,
        "rhythm": rhythm, "features": feats, "copying": copy, "overall": overall, "notes": notes,
    }
    write_json(out_dir / "realism_report.json", res)
    (out_dir / "realism_report.md").write_text(markdown(res), encoding="utf-8")
    plots(out_dir, x_real, y_real, x_syn, y_syn, rb, sb)
    print(f"  OVERALL {overall} -> {out_dir / 'realism_report.md'}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--real", nargs="+", default=["part:dev_unseen_by_both"])
    ap.add_argument("--copy-ref", default="part:sssd_train_ref")
    ap.add_argument("--layout", type=Path, default=None, help="inspection.json; harmonise real through the same layout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-per-class", type=int, default=EVAL_MAX_PER_CLASS)
    ap.add_argument("--quick", action="store_true", help="tiny detector budget for smoke tests only")
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    layout = json.loads(args.layout.read_text(encoding="utf-8"))["layout"] if args.layout else None
    return evaluate(paths, args.pool, args.real, args.copy_ref, args.out, layout,
                    args.title or args.pool.name, args.max_per_class, QUICK_DETECTOR if args.quick else None)


if __name__ == "__main__":
    main()
