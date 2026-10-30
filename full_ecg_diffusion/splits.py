"""Phase 0: patient-level judge manifest (ecg_ids only, no waveforms).

The published SSSD-ECG model trained on PTB-XL folds 1-8. Only folds 9-10 are unseen by it.
PTB-XL folds and our split are both patient-level, so patient sets stay clean.

Parts (our 4-class records only):
  judge_reserved      50% of patients in folds 9-10 AND our test split (class-stratified). Phase 7 only.
  dev_unseen_by_both  rest of folds 9-10 AND (val or test). Phases 3 and 6.
  dev_unseen_by_sssd  folds 9-10 AND our train split. Phase 3 only (our model trains on these).
  sssd_train_ref      folds 1-8 (what the published model trained on). Copying reference.

  python -m full_ecg_diffusion.splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from .config import CLASS_ORDER, SEED, add_path_args, paths_from_args
from .ptbxl import load_database, load_metadata

UNSEEN_FOLDS = (9, 10)


def parts_hash(parts: dict) -> str:
    canon = json.dumps({k: sorted(v["ecg_ids"]) for k, v in sorted(parts.items())}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def build_manifest(paths, seed: int = SEED) -> dict:
    db = load_database(paths)
    meta = load_metadata(paths)
    meta = meta.join(db[["strat_fold"]], how="left")
    if meta["strat_fold"].isna().any():
        raise ValueError("some of our ecg_ids are missing from ptbxl_database.csv")
    meta["strat_fold"] = meta["strat_fold"].astype(int)
    meta["unseen"] = meta["strat_fold"].isin(UNSEEN_FOLDS)

    # A patient must sit in exactly one fold group; drop any (none expected) that straddle.
    straddle = meta.groupby("patient_id")["unseen"].nunique()
    bad = set(straddle[straddle > 1].index)
    meta = meta[~meta["patient_id"].isin(bad)]

    # Primary class per patient = most frequent label among their records (ties -> lowest id).
    prim = meta.groupby("patient_id")["label"].agg(lambda s: int(s.value_counts().sort_index().idxmax()))

    unseen = meta[meta["unseen"]]
    test_unseen_pat = np.array(sorted(unseen.loc[unseen["split"] == "test", "patient_id"].unique()))
    rng = np.random.default_rng(seed)
    reserved_pat = set()
    for c in range(len(CLASS_ORDER)):
        pats = np.array([p for p in test_unseen_pat if prim[p] == c])
        rng.shuffle(pats)
        reserved_pat.update(pats[: (len(pats) + 1) // 2].tolist())

    def part(df):
        ids = sorted(df["ecg_id"].astype(int).tolist())
        return {
            "ecg_ids": ids,
            "n": len(ids),
            "n_patients": int(df["patient_id"].nunique()),
            "by_class": {CLASS_ORDER[c]: int((df["label"] == c).sum()) for c in range(len(CLASS_ORDER))},
        }

    is_res = unseen["patient_id"].isin(reserved_pat)
    parts = {
        "judge_reserved": part(unseen[is_res]),
        "dev_unseen_by_both": part(unseen[~is_res & unseen["split"].isin(["val", "test"])]),
        "dev_unseen_by_sssd": part(unseen[unseen["split"] == "train"]),
        "sssd_train_ref": part(meta[~meta["unseen"]]),
    }
    db_sha = hashlib.sha256(paths.ptbxl_database.read_bytes()).hexdigest()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "unseen_folds": list(UNSEEN_FOLDS),
        "rule": "patient-level; judge_reserved = 50% of folds 9-10 x test-split patients, class-stratified",
        "ptbxl_database_sha256": db_sha,
        "dropped_straddling_patients": len(bad),
        "parts": parts,
        "parts_sha256": parts_hash(parts),
    }


def load_manifest(paths) -> dict:
    m = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if parts_hash(m["parts"]) != m["parts_sha256"]:
        raise ValueError(f"{paths.manifest} was edited after creation (hash mismatch)")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest (breaks the seal)")
    args = ap.parse_args(argv)
    paths = paths_from_args(args)
    if paths.manifest.exists() and not args.force:
        m = load_manifest(paths)
        print(f"manifest exists and is intact: {paths.manifest} sha {m['parts_sha256'][:12]}")
        return m
    m = build_manifest(paths)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(m, indent=1), encoding="utf-8")
    for k, v in m["parts"].items():
        print(f"{k:20s} n={v['n']:6d} patients={v['n_patients']:6d} {v['by_class']}")
    print("wrote", paths.manifest)
    return m


if __name__ == "__main__":
    main()
