"""Phase 1: inspect the published SSSD-ECG synthetic data, infer its format, convert the 4-class part.

Download (manually, you approve it) the figshare dataset DOI 10.6084/m9.figshare.21922947
(https://figshare.com/s/43df16e4a50e4dd0a0c5) and extract it under archive/sssd_ecg_public/.
Nothing here downloads or unpacks archives; .npy/.npz are read with allow_pickle=False.

  python -m full_ecg_diffusion.public_data            # inspect + convert + HR gate
  python -m full_ecg_diffusion.public_data --channels I,V1,V2,V3,V4,V5,V6,aVF   # override inference

Lead order and units are inferred from the data (not from their docs): limb leads by the exact
Einthoven/Goldberger identities (when >= 3 are present) and every lead by matching the real
PTB-XL inter-lead correlation and amplitude profile. Labels: 71-dim multi-hot, default order =
alphabetical SCP codes (PTB-XL benchmark MultiLabelBinarizer), checked by prevalence rank.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .checks.beats import batch_beats, rhythm_check
from .config import CLASS_ORDER, LEAD_ORDER_12, SEED, THRESHOLDS, add_path_args, paths_from_args
from .convert import foreign_to_z, infer_axis, infer_channels, infer_units, reference_from_real, to_channels_first
from .ptbxl import OurArrays, load_database, norm_stats, scp_code_list
from .report import write_json

# The runner-up assignment must score at least this many times worse than the best.
AMBIGUOUS_RATIO = 2.0


def _arrays_in(path: Path):
    """Yield (name, array-like) for .npy and each key of .npz, without unpickling."""
    if path.suffix == ".npy":
        yield path.name, np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            yield f"{path.name}:{k}", z[k]


def discover(public_dir: Path, exclude: Path) -> dict:
    files, archives = [], []
    for p in sorted(public_dir.rglob("*")):
        if not p.is_file() or exclude in p.parents:
            continue
        if p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".7z"}:
            archives.append(str(p))
        if p.suffix in {".npy", ".npz"}:
            for name, arr in _arrays_in(p):
                files.append({"path": str(p), "name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)})
    return {"arrays": files, "unextracted_archives": archives}


def _kind(shape):
    if len(shape) == 3 and 1000 in shape[1:] and (set(shape[1:]) - {1000}) <= {8, 12}:
        return "signals"
    if len(shape) == 2 and shape[1] == 71:
        return "labels"
    return "other"


def pair_files(arrays: list[dict]) -> list[tuple[dict, dict]]:
    sig = [a for a in arrays if _kind(a["shape"]) == "signals"]
    lab = [a for a in arrays if _kind(a["shape"]) == "labels"]
    pairs = []
    for s in sig:
        cands = [l for l in lab if Path(l["path"]).parent == Path(s["path"]).parent and l["shape"][0] == s["shape"][0]]
        want = re.sub(r"(samples|data|signals|ecgs?)", "labels", s["name"], flags=re.I)
        named = [l for l in cands if l["name"] == want]
        if named or len(cands) == 1:
            pairs.append((s, (named or cands)[0]))
    return pairs


def load_entry(e: dict) -> np.ndarray:
    p = Path(e["path"])
    if ":" in e["name"]:
        return np.load(p, allow_pickle=False)[e["name"].split(":", 1)[1]]
    return np.load(p, mmap_mode="r", allow_pickle=False)


def four_class(labels: np.ndarray, label_order: list[str]) -> np.ndarray:
    """Our rule: exactly one of SR/AFIB/STACH/SBRAD present -> that class, else -1."""
    cols = [label_order.index(c) for c in CLASS_ORDER]
    hit = labels[:, cols] > 0.5
    y = np.full(len(labels), -1, dtype=int)
    one = hit.sum(axis=1) == 1
    y[one] = hit[one].argmax(axis=1)
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--channels", default=None, help="comma list overriding the inferred channel order")
    ap.add_argument("--units", choices=["mV", "uV"], default=None, help="override inferred units")
    ap.add_argument("--max-per-class", type=int, default=3000, help="cap of converted records per class")
    ap.add_argument("--hr-check-per-class", type=int, default=300)
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    th = THRESHOLDS
    out_dir = paths.inspection.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    found = discover(paths.public_dir, paths.public_converted)
    if found["unextracted_archives"] and not found["arrays"]:
        raise SystemExit(f"only archives found {found['unextracted_archives']}: extract them under {paths.public_dir} first")
    pairs = pair_files(found["arrays"])
    if not pairs:
        raise SystemExit(f"no (signals, 71-label) array pairs found under {paths.public_dir}; see inspection of arrays:\n"
                         + json.dumps(found, indent=1))
    print(f"{len(pairs)} signal/label pairs, e.g. {pairs[0][0]['name']} {pairs[0][0]['shape']} + {pairs[0][1]['name']}")

    # Real reference (our train split, z -> mV) for lead/unit inference.
    arrays = OurArrays(paths)
    mean, std = norm_stats(paths)
    rng = np.random.default_rng(SEED)
    tr_ids = arrays.meta.loc[arrays.meta["split"] == "train", "ecg_id"].to_numpy()
    ref_ids = rng.choice(tr_ids, min(2000, len(tr_ids)), replace=False)
    x_ref, y_ref, _ = arrays.get(ref_ids)
    ref = reference_from_real(x_ref, mean, std)

    axis = infer_axis(pairs[0][0]["shape"])
    sample = np.concatenate([np.asarray(load_entry(s)[:500]) for s, _ in pairs[:4]])
    x_cf = to_channels_first(sample.astype(np.float64), axis)
    inf = infer_channels(x_cf, ref)
    channels = args.channels.split(",") if args.channels else inf["channels"]
    units = infer_units(x_cf, channels, ref["std_mV"])
    unit_name = args.units or units["units"]
    if unit_name is None:
        raise SystemExit(f"cannot infer units (std ratio vs real mV {units['std_ratio_vs_real_mV']:.3g}); pass --units")
    layout = {"axis": axis, "channels": channels, "units": unit_name, "scale_to_mV": 1.0 if unit_name == "mV" else 1e-3}
    ratio = (inf["second_score"] / max(inf["score"], 1e-12)) if inf["second_score"] is not None else float("inf")
    inf["second_to_best_ratio"] = ratio
    ambiguous = ratio < AMBIGUOUS_RATIO and not args.channels
    print("layout", layout, f"runner-up/best score ratio {ratio:.1f} (runner-up {inf['second_channels']})")

    # Labels: order + prevalence check against PTB-XL.
    label_order = scp_code_list(paths)
    if len(label_order) != 71:
        raise SystemExit(f"scp_statements has {len(label_order)} codes, expected 71")
    labels = np.concatenate([np.asarray(load_entry(l)) for _, l in pairs]).astype(np.float32)
    db = load_database(paths)
    db_prev = np.array([db["scp_dict"].apply(lambda d, c=c: c in d).mean() for c in label_order])
    rank_corr = float(spearmanr(labels.mean(axis=0), db_prev).correlation)
    y_all = four_class(labels, label_order)

    # Convert the 4-class part, capped per class.
    keep = []
    for c in range(len(CLASS_ORDER)):
        idx = np.flatnonzero(y_all == c)
        keep.append(np.sort(rng.choice(idx, min(len(idx), args.max_per_class), replace=False)))
    keep = np.sort(np.concatenate(keep))
    offsets = np.cumsum([0] + [s["shape"][0] for s, _ in pairs])
    xs, rows = [], []
    for k, (s, _) in enumerate(pairs):
        sel = keep[(keep >= offsets[k]) & (keep < offsets[k + 1])] - offsets[k]
        if len(sel):
            xs.append(foreign_to_z(np.asarray(load_entry(s)[sel]), layout, mean, std))
            rows += [{"source": s["name"], "row": int(r), "global_row": int(r + offsets[k])} for r in sel]
    x_conv = np.concatenate(xs)
    y_conv = y_all[keep]
    conv = paths.public_converted
    conv.mkdir(parents=True, exist_ok=True)
    np.save(conv / "X_synthetic.npy", x_conv)
    np.save(conv / "y_synthetic.npy", y_conv)
    np.save(conv / "labels71.npy", labels[keep])
    meta = pd.DataFrame(rows)
    meta["class_id"] = y_conv
    meta["class_name"] = [CLASS_ORDER[c] for c in y_conv]
    meta["generator"] = "SSSD-ECG published (figshare 21922947)"
    meta["never_for_classifier"] = True
    meta.to_csv(conv / "metadata.csv", index=False)

    # HR gate: same R-peak detector on real (calibration) and converted published samples.
    def per_class_take(y, n):
        return np.sort(np.concatenate([np.flatnonzero(y == c)[:n] for c in range(len(CLASS_ORDER))]))

    ir, isyn = per_class_take(y_ref, args.hr_check_per_class), per_class_take(y_conv, args.hr_check_per_class)
    rh = rhythm_check(batch_beats(x_ref[ir]), y_ref[ir], batch_beats(x_conv[isyn]), y_conv[isyn], th)
    syn_hr = {n: rh["syn"][n]["median_hr"] for n in CLASS_ORDER}
    hr_ok = bool(rh["detector_calibrated_on_real"] and rh["checks"]["class_ordering"])

    gates = {
        "pairs_found": True,
        "layout_unambiguous": not ambiguous,
        "units_inferred": units["units"] is not None or args.units is not None,
        "label_order_prevalence_rank_corr": rank_corr >= th.phase1_label_rank_corr,
        "hr_sensible": hr_ok,
    }
    res = {
        "found": found, "pairs": [[s["name"], l["name"]] for s, l in pairs],
        "layout": layout, "channel_inference": inf, "units": units, "overrides": {"channels": args.channels, "units": args.units},
        "label_order": label_order, "label_rank_corr_vs_ptbxl": rank_corr,
        "n_total": int(len(labels)), "four_class_counts": {CLASS_ORDER[c]: int((y_all == c).sum()) for c in range(4)},
        "converted_counts": {CLASS_ORDER[c]: int((y_conv == c).sum()) for c in range(4)},
        "converted_dir": str(conv),
        "hr_gate": {"syn_median_hr": syn_hr, "real_median_hr": {n: rh["real"][n]["median_hr"] for n in CLASS_ORDER},
                    "detector_calibrated_on_real": rh["detector_calibrated_on_real"]},
        "gates": gates, "pass": all(gates.values()),
        "lead_order_12_of_converted": LEAD_ORDER_12,
    }
    write_json(paths.inspection, res)
    print(json.dumps({"gates": gates, "syn_median_hr": syn_hr, "label_rank_corr": rank_corr}, indent=1, default=str))
    print("PHASE 1", "PASS" if res["pass"] else "FAIL", "->", paths.inspection)
    return res


if __name__ == "__main__":
    main()
