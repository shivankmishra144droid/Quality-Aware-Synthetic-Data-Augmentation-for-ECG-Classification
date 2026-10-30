"""Deviation D-B1 from report/PREREGISTRATION_3.md (approved by the user 2026-10-30): run _noise_bank_eval.py with one
change to reference().

The registered reference takes 256 non-donor ECGs per class, then 32 more for real_ref. SBRAD has 446 train ECGs and
223 are donors, so all 223 non-donors land in the reference and real_ref gets no SBRAD; scorecard.memorized_frac then
fails on an empty array. Here each class's reference size is min(256, non-donors - 32), so real_ref always gets 32.
Only SBRAD changes (reference 191); SR, AFIB and STACH keep 256 + 32. Donors stay out of both sets. Everything else
(_noise_bank_eval.py, gate, outputs) is the frozen code, unchanged.

Usage:  py -3.14 _noise_bank_eval_fix1.py [--variants N0,N1,...]
"""
from __future__ import annotations

import numpy as np

import _noise_bank_eval as E
from generator_comparison.eval_metrics import CLASS_ORDER, lead_ii_hr_pt

SC = E.SC


def reference(X, y, exclude):
    rng = np.random.default_rng(42)
    ref_idx, pool_idx = [], []
    for c in range(4):
        idx = rng.permutation(np.setdiff1d(np.where(y == c)[0], exclude))
        n_ref = min(SC.N_REF, len(idx) - SC.N_POOL)                 # the only change
        ref_idx.append(idx[:n_ref])
        pool_idx.append(idx[n_ref:n_ref + SC.N_POOL])
        print(f"reference class {c}: {n_ref} ref + {len(pool_idx[-1])} real_ref", flush=True)
    ref_idx, pool_idx = np.sort(np.concatenate(ref_idx)), np.sort(np.concatenate(pool_idx))
    x_ref, y_ref = np.asarray(X[ref_idx]), y[ref_idx]
    ref = (x_ref, y_ref, [SC.log_psd(x_ref[y_ref == c]) for c in range(4)],
           {CLASS_ORDER[c]: lead_ii_hr_pt(x_ref[y_ref == c])["median_hr_peaks"] for c in range(4)})
    real_std_lead = x_ref.std(axis=(0, 1))
    real_ref = SC.score_pool(np.asarray(X[pool_idx]), y[pool_idx], ref, real_std_lead, X, y, exclude=pool_idx)
    return ref, real_std_lead, real_ref


E.reference = reference

if __name__ == "__main__":
    E.main()
