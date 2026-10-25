# Pre-registration: confirmatory filtered-vs-random experiment

Written and committed **before** any confirmatory run. The git commit containing this file, and its timestamp, is the
registration. Nothing below may be changed after the run starts; any deviation is reported as a deviation.

## Background
The exploratory main experiment (`experiment/main/`, seeds 0–9) found quality-filtered synthetic augmentation
+0.010 test macro-F1 above randomly selected synthetic augmentation (8/10 seeds, two-sided Wilcoxon p = 0.065,
paired dz = 0.63). That study was underpowered (≈ 22 seeds needed for 80 % power at that effect size).

## Hypothesis
H1: training the baseline CNN-LSTM on real data plus **quality-filtered** synthetic minority ECGs gives a different
(expected: higher) test macro-F1 than real data plus the same number of **randomly selected** synthetic minority ECGs.
H0: no difference.

## Design (identical to the exploratory run except the seeds)
- Conditions: `filtered` (top-scored by `score`, same subset for every seed) and `random` (uniform per class, new draw
  per seed, rng seed 1000 + seed). No other conditions.
- Pool and scores: `generator_comparison/final_pool/pool.npz` and `scores.npz` (SHA-256 below), ratio 1.0
  (+1,056 AFIB, +568 STACH, +446 SBRAD).
- Training recipe: `generator_comparison/augment_experiment.py` at this commit (baseline recipe, ≤ 25 epochs, early
  stopping and LR schedule on val macro-F1, weighted CE from each condition's labels).
- **Seeds: 100–129 (30 seeds), never used before.** Seed s is used for both conditions (paired design).
- Launch: `bash run_retry.sh _run_confirmatory.py generator_comparison/experiment_confirmatory_run.log`
  (tag `confirmatory`). A run that crashes is rerun with the **same seed**. No run is excluded.

## Stopping rule
Exactly 30 seeds per condition. No interim analysis of test results. The run prints only validation metrics.

## Primary analysis (frozen in `_confirmatory_analysis.py` at this commit)
- Outcome: test macro-F1 per run.
- Test: **two-sided Wilcoxon signed-rank test** on the 30 paired differences (filtered − random), α = 0.05.
- Decision: H1 supported if p < 0.05 **and** the mean difference is > 0. If p < 0.05 with a negative difference,
  random is better. If p ≥ 0.05, the confirmatory study does not support a difference.
- Reported with the decision: mean and median difference, seeds won, paired dz, and a 95 % CI of the mean difference
  from a two-level bootstrap (test patients × seeds, 2,000 resamples, seed 0).

## Secondary (descriptive only, no inference claimed)
Test SBRAD F1 difference, paired t-test p, per-class recall. These do not affect the decision.

## Known limitation, stated in advance
The test split was already analysed in the exploratory study (no fresh test set exists for PTB-XL in this design).
New seeds plus this pre-specified single test are the strongest available safeguard.

## Frozen inputs (SHA-256)
```
a8b7be16cd1ef0e1c7eac4453f3d85054b930d6a29064d3029e425580aaab400  generator_comparison/final_pool/pool.npz
166735d55bfaad402e3bc40978d9757765a3957cd7e4a17a2a329d33b6303416  generator_comparison/final_pool/scores.npz
19a8e764681f0a0a897a7edf9d0f46565798d686948aa1d43ba076801ad979df  generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt
51466880ac3c50a9d6957f1d90be58db72fb2f06a40bb94704ba0751fbfd327e  baseline/cnn_lstm_baseline.pt
```
