"""Phase 7: the one registered final test on the sealed `judge_reserved` patients. Runs once.

Order: write registration.json (git commit, dirty flag, pool SHA-256s, manifest hash, thresholds,
detector config, command line) -> evaluate each pool on judge_reserved -> final_result.json.
If final_result.json exists it refuses to run. If a run crashed after registering, the rerun is
allowed only when everything registered is byte-identical.

  python -m full_ecg_diffusion.final_test \
      --name ours --pool generator_comparison/sssd_full/ours_pool --copy-ref split:train \
      --layout generator_comparison/sssd_full/public_inspection/inspection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import DETECTOR, PROJECT_ROOT, SEALED_PART, THRESHOLDS, add_path_args, paths_from_args
from .evaluate import evaluate
from .report import write_json
from .splits import load_manifest


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*a) -> str:
    try:
        return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--name", nargs="+", required=True)
    ap.add_argument("--pool", nargs="+", type=Path, required=True)
    ap.add_argument("--copy-ref", nargs="+", required=True)
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args(argv)
    if not (len(args.name) == len(args.pool) == len(args.copy_ref)):
        raise SystemExit("--name, --pool and --copy-ref need the same number of entries")
    paths = paths_from_args(args)
    fd = paths.final_dir
    fd.mkdir(parents=True, exist_ok=True)
    result_path, reg_path = fd / "final_result.json", fd / "registration.json"
    if result_path.exists():
        print(f"REFUSED: the final test already ran ({result_path}). It is run exactly once.")
        sys.exit(2)

    manifest = load_manifest(paths)
    reg = {
        "part": SEALED_PART,
        "manifest_parts_sha256": manifest["parts_sha256"],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "pools": [{"name": n, "dir": str(p), "copy_ref": c, "X_sha256": _sha(p / "X_synthetic.npy"),
                   "y_sha256": _sha(p / "y_synthetic.npy")} for n, p, c in zip(args.name, args.pool, args.copy_ref)],
        "layout": str(args.layout) if args.layout else None,
        "layout_sha256": _sha(args.layout) if args.layout else None,
        "thresholds": THRESHOLDS.as_dict(),
        "detector": DETECTOR,
    }
    if reg_path.exists():
        old = json.loads(reg_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "registered_utc"} != reg:
            print("REFUSED: a different registration already exists; the final test cannot be re-specified.")
            sys.exit(2)
        print("re-running a crashed final test under the identical registration")
    else:
        reg_path.write_text(json.dumps({**reg, "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                       indent=2), encoding="utf-8")
        print("registered", reg_path)

    layout = json.loads(args.layout.read_text(encoding="utf-8"))["layout"] if args.layout else None
    out = {}
    for n, p, c in zip(args.name, args.pool, args.copy_ref):
        r = evaluate(paths, p, [f"part:{SEALED_PART}"], c, fd / n, layout, f"FINAL {n}", allow_sealed=True)
        out[n] = {"overall": r["overall"], "detector_auc": r["detector"]["auc"], "detector_ci95": r["detector"]["auc_ci95"],
                  "rhythm_pass": r["rhythm"]["pass"], "features_pass": r["features"]["pass"],
                  "copying_pass": r["copying"]["pass"], "control_auc": r["control"]["auc"]}
    write_json(result_path, {"registration": str(reg_path), "results": out,
                             "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    print(json.dumps(out, indent=1))
    print("FINAL TEST DONE ->", result_path)


if __name__ == "__main__":
    main()
