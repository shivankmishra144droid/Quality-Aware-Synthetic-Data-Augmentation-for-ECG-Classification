# Step 1 diagnosis (before any v2 training)

Measured on the frozen Week-4 checkpoint `synthetic_generation/checkpoints/full_last.pt` and `generator_comparison/sssd/` pool.

## Conditioning
Not a wiring bug. One-hot vectors are class-correct. `label @ embedding.weight` produces distinct 128-d rows (cosine ≈ 0 except AFIB–SBRAD 0.25). Residual-block `fc_label` adds differ by class (L2 0.33–0.43). Same noisy `x_t` at t=100 yields different `eps` per class (min pairwise L2 0.089). Conditioning reaches every residual block; it is not zeroed.

## Normalization
No denorm mismatch. `X_train` and synthetic both live in Week-2 train-only per-lead z-score space (real std 1.018, syn std 0.495). The amplitude gap is the generator, not a stats-file bug.

## Inference steps
Checkpoint was trained with T=200; sampling already uses all 200 reverse steps. DDPM T cannot be increased on this checkpoint without a new schedule. A skip-step NFE probe (invalid DDPM, diagnostic only) gave std 0.65 / 0.41 / 0.33 at 50 / 100 / 200 NFEs — fewer steps were not more damped. Amplitude failure is not “too few reverse steps.”

## Training length
4000 iters, batch 1. Noise MSE 1.02 → last 0.125 (mean of last 10 logs 0.11 vs first 10 logs 0.67). Minimum 0.045 at iter 2600, then noisy. Not a flat plateau, but also not steadily falling. 4000 updates is a small fraction of 13811 train ECGs. This is the plausible cause of weak class-conditional rate and low amplitude.

## Step 2 choice
**(b) more iterations only.** Resume 4000 → 16000, keep 8×64, T=200, seed 42, 32/class. No capacity change, no sampler change.
