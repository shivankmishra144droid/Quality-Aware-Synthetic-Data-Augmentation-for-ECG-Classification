"""End-to-end smoke test of every phase on fake data (CPU, ~10 min). Not a result.

Builds a fake PTB-XL (wfdb records, ptbxl_database.csv, scp_statements.csv), fake processed_data,
fake "published" SSSD files (channels I,V1-V6,aVF in mV, 71-label chunks) and fake weights, then
runs phases 0-7 through their real CLIs and asserts the gates behave.

  python -m full_ecg_diffusion.selftest --work /tmp/fed_selftest
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_ORDER, LEAD_ORDER_12, PROJECT_ROOT, WAVENETS

FS = 100
HR = {"SR": (72, 8), "AFIB": (105, 12), "STACH": (115, 8), "SBRAD": (52, 4)}
CHEST_DIRS = np.array([[-0.6, 0.0, 0.8], [-0.2, 0.0, 1.0], [0.3, 0.1, 0.9], [0.7, 0.2, 0.6], [0.9, 0.2, 0.3], [1.0, 0.2, 0.0]])


def fake_ecg(cls: str, rng) -> np.ndarray:
    """(1000, 12) mV from a 3-D dipole; limb leads satisfy Einthoven/Goldberger exactly."""
    t = np.arange(1000) / FS
    mu, sd = HR[cls]
    rr_mean = 60.0 / max(35.0, rng.normal(mu, sd))
    beats, s = [], rng.uniform(0, rr_mean)
    while s < 10.5:
        beats.append(s)
        s += rr_mean * (rng.uniform(0.65, 1.35) if cls == "AFIB" else rng.normal(1.0, 0.02))
    rot = rng.normal(0, 0.15, (3, 3)) + np.eye(3)
    waves = [  # (offset s, width s, amplitude mV, 3-D direction)
        (-0.16, 0.025, 0.12, [0.6, 0.7, -0.1]), (-0.03, 0.010, -0.10, [0.3, -0.2, 0.6]),
        (0.0, 0.014, 1.10, [0.7, 0.6, -0.4]), (0.035, 0.012, -0.30, [-0.2, 0.3, 0.9]),
        (0.26, 0.055, 0.30, [0.6, 0.5, -0.3]),
    ]
    v = np.zeros((3, len(t)))
    for b in beats:
        for off, w, a, d in waves:
            if cls == "AFIB" and off < -0.1:
                continue
            v += np.outer(rot @ np.array(d), a * np.exp(-0.5 * ((t - b - off) / w) ** 2))
    if cls == "AFIB":
        v[:2] += 0.04 * np.sin(2 * np.pi * rng.uniform(5, 7) * t + rng.uniform(0, 6))
    # Noise and wander enter per recorded lead (I, II, V1-V6); III/aVR/aVL/aVF are derived from I and II,
    # as on a real cart, so the limb identities hold exactly.
    wander = 0.04 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6))
    lead_i = v[0] + wander + rng.normal(0, 0.01, len(t))
    lead_ii = 0.5 * v[0] + 0.866 * v[1] + wander + rng.normal(0, 0.01, len(t))
    limb = [lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2]
    chest = list(CHEST_DIRS @ v + wander + rng.normal(0, 0.01, (6, len(t))))
    return np.stack(limb + chest, axis=1)


def make_fixtures(work: Path, n_patients: int = 1600, n_public: int = 1200, seed: int = 0) -> dict:
    import torch
    import wfdb

    from .nets import build_net

    rng = np.random.default_rng(seed)
    ptb = work / "ptbxl"
    data = work / "processed_data"
    pub = work / "public"
    for d in (ptb, data, pub):
        d.mkdir(parents=True, exist_ok=True)
    codes = CLASS_ORDER + ["NORM"] + [f"D{i:02d}" for i in range(66)]
    pd.DataFrame({"scp_code": codes, "description": codes}).set_index("scp_code").to_csv(ptb / "scp_statements.csv")
    probs = np.array([0.55, 0.15, 0.15, 0.15])
    diag_p = rng.dirichlet(np.ones(67) * 0.3)

    def scp(cls):
        d = {cls: 100.0}
        for c in rng.choice(codes[4:], rng.integers(1, 3), p=diag_p, replace=False):
            d[str(c)] = 100.0
        return d

    rows, ecg_id = [], 0
    for p in range(n_patients):
        cls = CLASS_ORDER[rng.choice(4, p=probs)]
        fold, split = int(rng.integers(1, 11)), str(rng.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15]))
        for _ in range(int(rng.integers(1, 3))):
            ecg_id += 1
            fn = f"records100/{(ecg_id // 1000) * 1000:05d}/{ecg_id:05d}_lr"
            (ptb / fn).parent.mkdir(parents=True, exist_ok=True)
            sig = fake_ecg(cls, rng)
            wfdb.wrsamp(Path(fn).name, fs=FS, units=["mV"] * 12, sig_name=[n.upper() for n in LEAD_ORDER_12],
                        p_signal=sig, fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12,
                        write_dir=str((ptb / fn).parent))
            rows.append({"ecg_id": ecg_id, "patient_id": float(p + 1), "scp_codes": str(scp(cls)), "strat_fold": fold,
                         "filename_lr": fn, "age": 50.0, "sex": 0, "target_rhythm": cls,
                         "label": CLASS_ORDER.index(cls), "split": split})
    db = pd.DataFrame(rows)
    db[["ecg_id", "patient_id", "age", "sex", "scp_codes", "strat_fold", "filename_lr"]].to_csv(ptb / "ptbxl_database.csv", index=False)

    raw = {}
    for split in ("train", "val", "test"):
        m = db[db["split"] == split].reset_index(drop=True)
        raw[split] = np.stack([wfdb.rdsamp(str(ptb / f))[0].astype(np.float32) for f in m["filename_lr"]])
        m[["ecg_id", "patient_id", "target_rhythm", "label", "split", "filename_lr", "age", "sex"]].to_csv(
            data / f"{split}_metadata.csv", index=False)
        np.save(data / f"y_{split}.npy", m["label"].to_numpy())
    mean = raw["train"].mean(axis=(0, 1)).astype(np.float64)
    std = raw["train"].std(axis=(0, 1)).astype(np.float64)
    for split, x in raw.items():
        np.save(data / f"X_{split}.npy", ((x - mean) / std).astype(np.float32))
    (data / "normalization_stats.json").write_text(json.dumps(
        {"lead_order": LEAD_ORDER_12, "mean": mean.tolist(), "std": std.tolist()}), encoding="utf-8")

    # Published-style samples: channels-first (N, 8, 1000), I,V1..V6,aVF, mV; 71-dim labels, sorted codes.
    order = sorted(codes)
    chans = ["I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"]
    idx = [LEAD_ORDER_12.index(c) for c in chans]
    for k in range(0, n_public, 400):
        n = min(400, n_public - k)
        cls = [CLASS_ORDER[rng.choice(4, p=probs)] for _ in range(n)]
        sig = np.stack([fake_ecg(c, rng)[:, idx].T for c in cls]).astype(np.float32)
        lab = np.zeros((n, 71), dtype=np.float32)
        for r, c in enumerate(cls):
            for code in scp(c):
                lab[r, order.index(code)] = 1.0
        np.save(pub / f"{k // 400}_samples.npy", sig)
        np.save(pub / f"{k // 400}_labels.npy", lab)

    (pub / "weights").mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.save({"model_state_dict": build_net(WAVENETS["tiny"]).state_dict()}, pub / "weights" / "tiny.pkl")
    paper_ckpt = work / "paper_random.pkl"
    torch.save({"model_state_dict": build_net(WAVENETS["paper"]).state_dict()}, paper_ckpt)
    return {"ptbxl": ptb, "data": data, "public": pub, "paper_ckpt": paper_ckpt}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--keep", action="store_true", help="reuse fixtures from a previous run")
    args = ap.parse_args(argv)
    work = args.work.resolve()
    if PROJECT_ROOT in work.parents or work == PROJECT_ROOT:
        raise SystemExit("--work must be outside the repository")
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    fx = make_fixtures(work) if not (work / "ptbxl").exists() else {
        "ptbxl": work / "ptbxl", "data": work / "processed_data", "public": work / "public",
        "paper_ckpt": work / "paper_random.pkl"}
    out = work / "out"
    common = ["--data-dir", str(fx["data"]), "--ptbxl-dir", str(fx["ptbxl"]), "--public-dir", str(fx["public"]),
              "--manifest", str(work / "manifest.json"), "--out-root", str(out),
              "--public-converted", str(fx["public"] / "converted"),
              "--inspection", str(out / "public_inspection" / "inspection.json"), "--final-dir", str(out / "final")]
    insp = out / "public_inspection" / "inspection.json"
    results = {}

    from . import (conversion_control, evaluate, final_test, pretrained, probe, public_data, sample, splits,
                   train)
    from .realsets import SealedPartError

    print("\n== phase 0 splits")
    m = splits.main(common)
    p = m["parts"]
    sets = {k: set(v["ecg_ids"]) for k, v in p.items()}
    assert all(not (sets[a] & sets[b]) for a in sets for b in sets if a < b), "parts overlap"
    results["phase0"] = {k: v["n"] for k, v in p.items()}

    print("\n== phase 1 public data")
    r1 = public_data.main(common)
    assert r1["layout"]["channels"] == ["I", "V1", "V2", "V3", "V4", "V5", "V6", "aVF"], r1["layout"]
    assert r1["layout"]["units"] == "mV" and r1["pass"], r1["gates"]
    results["phase1"] = r1["gates"]

    print("\n== phase 2 conversion control")
    r2 = conversion_control.main(common + ["--quick", "--n-detector-per-class", "60"])
    results["phase2"] = {"exact": r2["exactness"]["pass"], "perm": r2["permutation_recovery"]["pass"],
                         "detector_auc": r2["detector_control"]["auc"]}

    print("\n== phase 3 evaluate published-style pool")
    r3 = evaluate.main(common + ["--pool", str(fx["public"] / "converted"), "--real", "part:dev_unseen_by_both",
                                 "part:dev_unseen_by_sssd", "--copy-ref", "part:sssd_train_ref", "--layout", str(insp),
                                 "--out", str(out / "phase3_public"), "--quick"])
    assert r3["rhythm"]["detector_calibrated_on_real"], r3["rhythm"]["real"]
    results["phase3"] = {"overall": r3["overall"], "auc": r3["detector"]["auc"], "rhythm": r3["rhythm"]["pass"],
                         "features": r3["features"]["pass"], "copying": r3["copying"]["pass"]}

    print("\n== bad generator: committed Week-4 pool must fail the rhythm check")
    rb = evaluate.main(common + ["--pool", str(PROJECT_ROOT / "generator_comparison" / "sssd"),
                                 "--real", "part:dev_unseen_by_both", "part:dev_unseen_by_sssd",
                                 "--copy-ref", "part:sssd_train_ref", "--out", str(out / "week4"), "--quick"])
    assert not rb["rhythm"]["pass"], "Week-4 pool should fail rhythm"
    results["week4_pool"] = {"overall": rb["overall"], "auc": rb["detector"]["auc"], "rhythm": rb["rhythm"]["pass"]}

    print("\n== sealed part is refused outside the final test")
    try:
        evaluate.main(common + ["--pool", str(fx["public"] / "converted"), "--real", "part:judge_reserved",
                                "--out", str(out / "nope"), "--quick"])
        raise AssertionError("sealed part was not refused")
    except SealedPartError:
        results["sealed_refused"] = True

    print("\n== phase 4a published weights (tiny stand-in) + paper-size load check")
    r4 = pretrained.main(common + ["--weights", str(fx["public"] / "weights" / "tiny.pkl"), "--wavenet", "tiny",
                                   "--n-per-class", "4", "--quick"])
    r4b = pretrained.main(common + ["--weights", str(fx["paper_ckpt"]), "--load-only"])
    assert r4b["ok"], r4b
    results["phase4a"] = {"tiny_repro_ran": True, "paper_load_ok": r4b["ok"], "paper_n_params": r4b["n_params"]}

    print("\n== phase 4b probe (tiny, CPU)")
    r5 = probe.main(common + ["--preset", "tiny", "--device", "cpu", "--steps", "1"])
    results["phase4b"] = r5["recommendation"]

    print("\n== phase 5 train (tiny, CPU) + resume")
    bundle = out / "bundle"
    train.main(common + ["--export-bundle", str(bundle)])
    tdir = out / "ours_train"
    train.main(common + ["--bundle", str(bundle), "--out", str(tdir), "--preset", "tiny", "--device", "cpu"])
    last = train.main(common + ["--bundle", str(bundle), "--out", str(tdir), "--preset", "tiny", "--device", "cpu",
                                "--n-iters", "5", "--resume", str(tdir / "ours_last.pt")])
    log = json.loads((tdir / "history.json").read_text())["log"]
    assert log[-1]["iter"] == 5, log
    results["phase5"] = {"iters_logged": [r["iter"] for r in log]}

    print("\n== phase 6 sample + evaluate ours")
    pool = out / "ours_pool"
    sample.main(common + ["--ckpt", str(last), "--n-per-class", "6", "--out", str(pool), "--device", "cpu"])
    r6 = evaluate.main(common + ["--pool", str(pool), "--real", "part:dev_unseen_by_both", "--copy-ref", "split:train",
                                 "--layout", str(insp), "--out", str(out / "phase6_ours"), "--quick"])
    results["phase6"] = {"overall": r6["overall"], "auc": r6["detector"]["auc"]}

    print("\n== phase 7 final test runs once")
    ft = common + ["--name", "ours", "--pool", str(pool), "--copy-ref", "split:train", "--layout", str(insp)]
    from . import evaluate as ev

    ev_default = ev.evaluate
    ev.evaluate = lambda *a, **k: ev_default(*a, **{**k, "detector_cfg": ev.QUICK_DETECTOR})  # smoke budget
    final_test.evaluate = ev.evaluate
    final_test.main(ft)
    try:
        final_test.main(ft)
        raise AssertionError("second final test was not refused")
    except SystemExit as e:
        assert e.code == 2
    results["phase7"] = {"ran_once": True, "second_run_refused": True}

    (work / "selftest_results.json").write_text(json.dumps(results, indent=1, default=str))
    print("\nSELFTEST OK\n" + json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
