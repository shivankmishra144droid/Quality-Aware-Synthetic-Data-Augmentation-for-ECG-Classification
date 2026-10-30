"""Phase 2: prove the format conversion is exact, so a unit or lead mix-up cannot fake a 1.00.

  (a) raw PTB-XL records -> put into the inferred SSSD layout -> our converter -> must equal our
      X_* rows (channels present in the layout to <= 1e-4 z; rebuilt limb leads close, since PTB-XL
      stores each limb lead with its own quantisation).
  (b) lead-order inference must recover known random channel permutations of real data.
  (c) detector: converted real vs our real (disjoint patients) must stay <= 0.60 AUC.

  python -m full_ecg_diffusion.conversion_control
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .checks.detector import run_detector
from .config import LEAD_ORDER_12, SEED, THRESHOLDS, add_path_args, paths_from_args
from .convert import foreign_to_z, infer_channels, reference_from_real, z_to_foreign
from .ptbxl import OurArrays, load_raw_record, norm_stats
from .realsets import class_matched
from .report import write_json


def raw_to_layout(raw_mv: np.ndarray, layout: dict) -> np.ndarray:
    """(N, 1000, 12) raw mV -> foreign array exactly as the published files would store it."""
    idx = [LEAD_ORDER_12.index(c) for c in layout["channels"]]
    cf = np.transpose(raw_mv[:, :, idx], (0, 2, 1)) / layout["scale_to_mV"]
    return cf if layout["axis"] == "channels_first" else np.transpose(cf, (0, 2, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--n-exact", type=int, default=200)
    ap.add_argument("--n-detector-per-class", type=int, default=150)
    ap.add_argument("--quick", action="store_true", help="tiny detector budget for smoke tests only")
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    th = THRESHOLDS
    insp = json.loads(paths.inspection.read_text(encoding="utf-8"))
    layout = insp["layout"]
    arrays = OurArrays(paths)
    mean, std = norm_stats(paths)
    meta = arrays.meta
    rng = np.random.default_rng(SEED)

    # (a) exactness
    ids = rng.choice(meta["ecg_id"].to_numpy(), min(args.n_exact, len(meta)), replace=False)
    raw = np.stack([load_raw_record(paths, meta.loc[i, "filename_lr"]) for i in ids])
    ours, _, _ = arrays.get(ids)
    conv = foreign_to_z(raw_to_layout(raw, layout), layout, mean, std)
    diff = np.abs(conv.astype(np.float64) - ours)
    present = [LEAD_ORDER_12.index(c) for c in layout["channels"]]
    rebuilt = [j for j in range(12) if j not in present]
    exact_max = float(diff[:, :, present].max())
    a = {"n": int(len(ids)), "present_max_abs_z": exact_max, "present_ok": exact_max <= th.exact_max_abs}
    if rebuilt:
        corr = [float(np.corrcoef(conv[:, :, j].ravel(), ours[:, :, j].ravel())[0, 1]) for j in rebuilt]
        med = float(np.median(diff[:, :, rebuilt]))
        a.update(rebuilt_leads=[LEAD_ORDER_12[j] for j in rebuilt], rebuilt_median_abs_z=med,
                 rebuilt_p99_abs_z=float(np.percentile(diff[:, :, rebuilt], 99)), rebuilt_min_corr=min(corr),
                 rebuilt_ok=med <= th.derived_median_abs and min(corr) >= th.derived_min_corr)
    else:
        a["rebuilt_ok"] = True
    a["pass"] = bool(a["present_ok"] and a["rebuilt_ok"])
    print("(a) exactness", a)

    # (b) permutation recovery on real data
    ref = reference_from_real(ours, mean, std)
    trials = []
    tr_ids = rng.choice(meta["ecg_id"].to_numpy(), min(1000, len(meta)), replace=False)
    x_tr, _, _ = arrays.get(tr_ids)
    for t in range(3):
        perm = list(rng.permutation(layout["channels"]))
        lay = {**layout, "channels": perm, "axis": "channels_first"}
        got = infer_channels(z_to_foreign(x_tr, lay, mean, std), ref)
        trials.append({"true": perm, "inferred": got["channels"], "ok": got["channels"] == perm, "margin": got["margin"]})
    b = {"trials": trials, "pass": all(t["ok"] for t in trials)}
    print("(b) permutation recovery", [t["ok"] for t in trials])

    # (c) detector: converted-from-raw real vs our real, disjoint patients
    pats = meta["patient_id"].unique()
    half = set(rng.choice(pats, len(pats) // 2, replace=False))
    side = meta["patient_id"].isin(half).to_numpy()
    ma, mb = meta[side], meta[~side]
    ia, ib = class_matched(ma["label"].to_numpy(), mb["label"].to_numpy(), args.n_detector_per_class, SEED)
    ra, rb = ma.iloc[ia], mb.iloc[ib]
    raw_a = np.stack([load_raw_record(paths, f) for f in ra["filename_lr"]])
    x_a = foreign_to_z(raw_to_layout(raw_a, layout), layout, mean, std)
    x_b, _, _ = arrays.get(rb["ecg_id"].to_numpy())
    from .evaluate import QUICK_DETECTOR

    det = run_detector(x_a, x_b, ra["patient_id"].to_numpy(), rb["patient_id"].to_numpy(),
                       ra["label"].to_numpy(), rb["label"].to_numpy(), QUICK_DETECTOR if args.quick else None)
    c = {"auc": det["auc"], "auc_ci95": det["auc_ci95"], "pass": det["auc"] <= th.control_max, "detail": det}
    print(f"(c) detector converted-real vs our-real AUC {det['auc']:.3f}")

    res = {"layout": layout, "exactness": a, "permutation_recovery": b, "detector_control": c,
           "pass": bool(a["pass"] and b["pass"] and c["pass"])}
    out = paths.inspection.parent / "conversion_control.json"
    write_json(out, res)
    print("PHASE 2", "PASS" if res["pass"] else "FAIL", "->", out)
    if not res["pass"]:
        sys.exit(1)
    return res


if __name__ == "__main__":
    main()
