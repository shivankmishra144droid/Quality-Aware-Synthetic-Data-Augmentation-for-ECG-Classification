# test/ — scripts not used by the final pipeline

Kept for the record (report Section 5 and the Week-4 notebooks), not needed to reproduce the final results.
The final pipeline is listed in the root `README.md`.

| Script | What it was for |
|---|---|
| `_build_w4.py`, `_build_compare.py` | Regenerated notebooks 04 / 05 (Week 4) |
| `_check_ab.py`, `_run_sssd_v2.py`, `_run_cfg.py` | Week-4 SSSD quality pilots (ckpt2000, 16×128 v3, 16k iters, CFG w=2/4) |
| `preprocess_ptbxl.py` | 7-rhythm label-count helper on the raw CSV (not the frozen 4-class rule) |
| `_rescore_hr.py` | Re-scored all early pools with the fixed heart-rate metric (`hr_rescore/`) |
| `_run_balanced.py` | Class-balanced sampling run (`sssd_balanced/`) |
| `_run_ema.py`, `_run_sched.py`, `_run_budget.py` | EMA, rescaled noise schedule, 80× training budget runs |
| `_run_memorize.py` | Single-ECG memorisation test (eps vs v-prediction) |
| `_run_vpred16.py` | 16×128 v-prediction capacity check |
| `_sample_ckpt.py` | Sample a 32/class pool from any diffusion checkpoint |

These scripts import the project packages and use repo-relative paths, so run them **from the repo root**:

```bash
PYTHONPATH=. py -3.14 test/_run_memorize.py v
```

Several need data or checkpoints that are gitignored (rebuild `processed_data/X_*.npy` from notebook 02; checkpoints
from the corresponding run).
