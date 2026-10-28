# v-prediction run and what led to it (2026-09-26)

All numbers from `generator_comparison/scorecard/` (32/class pools vs 256 real train ECGs/class;
held-out real_ref scores 9/9). Periodicity = median Lead-II QRS-envelope ACF peak over SR/STACH/SBRAD.

## Sequence

| Run | Change | Periodicity | Std | Score |
|---|---|---|---|---|
| real_ref | – | 0.84 | 1.07 | 9/9 |
| v1 / ema_raw | 8x64, 4k iters, batch 1, eps | 0.13-0.17 | ~0.5 | 2/9 |
| ema | + EMA 0.999 | 0.12 | 0.42 | 3/9 |
| sched | + schedule 5e-4->0.1 (old ends with 36% signal at t=T) | 0.07-0.13 | 6-8 (blow-up) | 1-3/9 |
| budget (eps) | + batch 8 x accum 4, 10k steps (~320k ECGs), warmup+cosine | 0.16 | 0.9 | 5/9 (fails valid) |
| vpred @10k | same budget, **v-prediction**, EMA warmup | 0.20-0.21 | 0.72 | 4/9 |
| **vpred @5k (raw)** | same run, step-5000 checkpoint | 0.22 | 0.88 | **6/9** |

Step 5000 was picked from the in-training 8/class evals (same metric family as the scorecard),
so the 6/9 carries a small selection bias.

## Memorization test (`test/_run_memorize.py`, `memorize_test/`)

8x64, one clean SBRAD train ECG (periodicity 0.89), 2000 steps:
- eps: raw samples corr 0.29 with the training ECG, periodicity 0.09 -> cannot memorize ONE ECG.
- v: raw samples corr 0.967, periodicity 0.87, HR 59 -> memorizes.

Trace (eps checkpoint): per-t noise MSE on forward-noised inputs was tiny (0.007 at t=199), but at
t=199 that is worse than the trivial predictor (~3e-5), and along the sampling trajectory the
error grows (0.037 @ t=100 -> 587 @ t=0). An oracle eps reproduces the ECG exactly (corr 1.000),
so the sampler math is correct.

## Conclusions (scope: 8x64 on this data)

- At 8x64, v-prediction is **necessary** (eps cannot even memorize) but **not sufficient**:
  full-data periodicity plateaus at ~0.22 vs 0.84 real. No pool passes the heartbeat check.
- This is not a general claim that eps is wrong: published 36x256 SSSD-ECG uses eps successfully.
- Plateau, not a proven capacity ceiling. Capacity (36x256 = 50.2M params vs 1.53M) is the
  leading untested hypothesis. A 36x256 run with v-pred + rescaled schedule would be the
  SSSD-ECG backbone with a modified diffusion setup, not the published model.

## Process notes

- The eps "budget" run was stopped at ~4.3k steps, but its bash retry loop survived the
  python kill, resumed from step 4000 and trained to 10k concurrently with the memorize tests and
  this run (which is why this run was 2-3x slower). Its results are valid; its 1k-4k evals are
  only in `sssd_budget_run.log`. Future runs: STOP-file wrapper, check bash.exe too.
- Two early CUDA OOMs at ~1-1.5 GB peak were transient (other GPU apps); the identical config
  ran flat at 1.97 GB device memory in isolation.
