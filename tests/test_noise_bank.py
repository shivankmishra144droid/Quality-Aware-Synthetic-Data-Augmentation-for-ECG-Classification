"""Part B of report/PREREGISTRATION_3.md (real-noise-bank variants). Fake ECGs only; no real array is read.

Run from repo root: py -3.14 -m pytest tests -q
"""
import json

import numpy as np
import pytest

import _noise_bank_eval as EV
import _noise_bank_variants as V
from generator_comparison import noise_bank as NB
from generator_comparison import realism_detector as RD
from generator_comparison import tstr as TS
from synthetic_generation.leads import LEAD_ORDER_12, _norm_stats, reconstruct_12, to_independent_8


def fake_z(cls, rng):
    """(1000, 12) z-space ECG from the full_ecg_diffusion dipole fixture (limb identities exact)."""
    from full_ecg_diffusion.selftest import fake_ecg
    mean, std = _norm_stats()
    return ((fake_ecg(["SR", "AFIB", "STACH", "SBRAD"][cls], rng) - mean) / std).astype(np.float32)


@pytest.fixture(scope="module")
def donor():
    rng = np.random.default_rng(0)
    x8 = to_independent_8(fake_z(0, rng)[None])[0].astype(np.float64)
    p = NB.donor_parts(x8)
    assert p is not None
    return p


def test_alpha_zero_and_own_template_return_the_donor(donor):
    assert np.allclose(NB.synthesize(donor, donor["tpl"] * 0 + 7.0, 0.0), donor["x8"])
    assert np.allclose(NB.synthesize(donor, donor["tpl"], 1.0), donor["x8"])        # the E identity


def test_align_recovers_a_known_shift(donor):
    moved = NB._shift(donor["tpl"], 2.4)
    lag = NB.estimate_lag(moved, donor["tpl"])
    assert abs(lag + 2.4) < 0.6                                   # shifting back by ~2.4 samples
    back, _ = NB.align(moved, donor["tpl"])
    assert np.corrcoef(back.ravel(), donor["tpl"].ravel())[0, 1] > np.corrcoef(moved.ravel(), donor["tpl"].ravel())[0, 1]


def test_donor_corr_is_one_for_the_donor():
    rng = np.random.default_rng(1)
    x = np.stack([fake_z(c, rng) for c in range(4)])
    assert np.allclose(NB.donor_corr(x, x), 1.0, atol=1e-5)


def test_nearest_template_picks_the_closest(donor):
    rng = np.random.default_rng(2)
    bank = np.stack([rng.normal(size=donor["tpl"].shape) for _ in range(5)] + [donor["tpl"] * 1.1])
    assert NB.nearest_template(bank, donor["tpl"], np.ones(len(bank), bool)) == 5


@pytest.mark.parametrize("s", [1.1, 1.25])
def test_time_warp_slows_the_rate_and_keeps_lead_identities(donor, s):
    w, new_r = NB.time_warp(donor["x8"], donor["r"], s)
    assert w.shape == donor["x8"].shape
    rr_new = np.diff(new_r).mean()
    assert abs(rr_new / np.diff(donor["r"]).mean() - s) < 0.08
    x12 = reconstruct_12(w[None].astype(np.float32))[0]          # derived leads must still be consistent in mV
    mean, std = _norm_stats()
    mv = x12 * std + mean
    i, ii, iii = (mv[:, LEAD_ORDER_12.index(n)] for n in ("I", "II", "III"))
    assert np.abs(iii - (ii - i)).max() < 1e-4


def fake_world(monkeypatch, tmp_path, n_per_class=12):
    rng = np.random.default_rng(3)
    data, out = tmp_path / "data", tmp_path / "nb"
    data.mkdir()
    y = np.repeat(np.arange(4), n_per_class)
    np.save(data / "X_train.npy", np.stack([fake_z(c, rng) for c in y]))
    np.save(data / "y_train.npy", y)
    yv = np.repeat(np.arange(4), 4)
    np.save(data / "X_val.npy", np.stack([fake_z(c, rng) for c in yv]))
    np.save(data / "y_val.npy", yv)
    tpls = []
    for c in np.repeat(np.arange(4), 6):                            # "generated" templates = real ones, perturbed
        p = NB.donor_parts(to_independent_8(fake_z(c, rng)[None])[0])
        tpls.append(p["tpl"] * rng.uniform(0.9, 1.1))
    bank = tmp_path / "bank.npz"
    np.savez(bank, templates=np.stack(tpls), y=np.repeat(np.arange(4), 6), X=np.stack([fake_z(c, rng) for c in
                                                                                       np.repeat(np.arange(4), 6)]))
    for mod in (V, EV):
        monkeypatch.setattr(mod, "DATA", data)
        monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(RD, "DATA", data)
    monkeypatch.setattr(EV, "BEAT_GEN", bank)
    monkeypatch.setattr(EV, "N_TSTR", 3)
    monkeypatch.setattr(TS, "EPOCHS", 1)
    monkeypatch.setattr(TS, "SEEDS", (0,))
    monkeypatch.setattr(EV.SC, "N_REF", 3)
    monkeypatch.setattr(EV.SC, "N_POOL", 2)
    orig_run = RD.run
    monkeypatch.setattr(RD, "run", lambda *a, **k: orig_run(*a, **{**k, "epochs": 1}))
    return out, bank


def test_variants_and_eval_smoke(monkeypatch, tmp_path):
    out, bank = fake_world(monkeypatch, tmp_path)
    V.main(["--variants", "N0,N1,N2a0.50", "--banks", str(bank)])
    for v in ("N0", "N1", "N2a0.50"):
        d = np.load(out / f"{v}.npz")
        assert d["X"].shape[1:] == (1000, 12) and len(d["X"]) == len(d["y"]) == len(d["src_idx"])
        assert np.all(d["donor_corr"] <= 1.0 + 1e-5)
    half = np.load(out / "N2a0.50.npz")["donor_corr"]
    full = np.load(out / "N1.npz")["donor_corr"]
    assert np.median(half) >= np.median(full) - 0.05               # smaller alpha stays closer to the donor
    EV.main(["--variants", "N0,N1,N2a0.50"])
    res = json.loads((out / "results.json").read_text())
    assert set(res["variants"]) == {"N0", "N1", "N2a0.50"}
    assert (out / "results.md").exists() and (out / "realism_novelty.png").exists()
    V.main(["--warp-from", "N2a0.50"])
    assert (out / "W_N2a0.50.npz").exists()
