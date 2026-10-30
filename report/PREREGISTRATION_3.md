# Pre-registration 3: threshold-tuned control (A) and real-noise-bank variants (B)

Written before any part-A or part-B run. The code named below is frozen with this file. Nothing in study 1 or study 2 is re-run or re-analysed except the explicitly listed retraining, which only adds validation probabilities.

## Why

1. **Part A.** Study 2 (`report/REPORT.md` §8.3) found that random synthetic augmentation beats real-only training by +0.036 test macro-F1 (40/40 units). The per-class table suggests most of that gain is an operating-point shift: SBRAD F1 0.49 → 0.61 while SBRAD recall falls (0.81 → 0.77) and SBRAD AUPRC stays flat (0.761 vs 0.756). §9 names the missing control: a real-only model whose decision thresholds are tuned on validation. Part A runs that control.
2. **Part B.** The synthetic ECGs are detectable (AUC 1.00). Study 2's ablation (`_beat_ablation.py`) showed that a real recording rebuilt from its own template, residual and wander is undetectable (E: 0.51). Swapping in another recording's template at the same R-peaks is caught (T_rand 0.956, T_match 0.947). Algebraically the swap is `donor + (T' − own_template)@r`, one difference waveform repeated identically at every beat. Part B asks whether aligning the generated template to the donor's beat, and scaling the difference down, makes samples undetectable **while they stay new ECGs**.

## Part A: threshold-tuned real-only control

**Runs.**
- `_run_threshold_control.py` retrains conditions `real` and `random` for the 40 study-2 units (pools 1–4, seeds 200–239) with the study-2 recipe unchanged (`augment_experiment.train_steps`: 6,250 steps, batch 64, validation every 250, best-validation state, deterministic CUDA).
- Training sets are rebuilt exactly as `augment_experiment.main` builds them: `pick("random", …)`, real rows first, ratio 1.0.
- Per run it saves validation and test probabilities.
- Each retrained test-probability file is compared with the committed study-2 file of the same run (identical? max abs diff? same predictions?). That comparison is reported but does not change the analysis. The analysis always uses the retrained files, so validation and test probabilities come from the same model.

**Tuning (fixed).**
- Per-class multiplicative weights on the softmax; prediction = argmax_c(p_c · w_c), with w_SR = 1.
- Fit by coordinate ascent of validation macro-F1 over log w ∈ {−3.0, −2.9, …, 3.0} for AFIB, STACH and SBRAD, starting at w = 1, 3 sweeps; ties go to the grid value nearest log w = 0 (`generator_comparison/threshold_tuning.py`).
- Weights are fitted on each run's own validation probabilities and applied unchanged to its test probabilities.

**Comparisons**, test macro-F1, 40 paired units:

| id | comparison | question |
|---|---|---|
| T1 (primary) | random − real_tuned | Does augmentation still beat a threshold-tuned real-only model? |
| T2 | real_tuned − real | What does tuning alone buy? |
| T3 | random_tuned − real_tuned | Augmentation when both are tuned |

**Statistics** (`_threshold_analysis.py`, frozen, runs once).
- Two-sided Wilcoxon signed-rank on each comparison; Holm over T1–T3, α = 0.05.
- TOST for T1 (paired t, ±0.01).
- 95 % CIs from study 2's three-level bootstrap (test patients × pools × seeds, 2,000 resamples, seed 0).
- `share_recovered = mean(real_tuned − real) / mean(random − real)`.

**Decision rule.**
- **AUGMENTATION DOES REAL WORK** if T1 is significant (Holm) and positive.
- **MOSTLY THRESHOLD** if `share_recovered` ≥ 0.5 and T1 is not significant.
- Otherwise **MIXED**.

**Descriptive only.** Per-class recall, precision, SBRAD F1 and AUPRC for the four conditions. A validation-optimism check: tune on one half of the validation patients, score on the other half, both ways; in-sample vs held-out gain.

**Disclosed limits.**
- This is the fourth use of the test split.
- Validation already selects the checkpoint, so tuning on it reuses it; the optimism check measures how much that matters.
- Bit-exact reproduction is only expected on the original machine (RTX 3050 Laptop, torch 2.11.0+cu128, Windows 11).

## Part B: real-noise-bank variants

Train and validation data only; the test split is never loaded.

**Donors.** Real train ECGs drawn exactly as `_beat_ablation.py` draws them: rng 7, up to min(500, n_c // 2) per class, which leaves disjoint real ECGs for the detector. All donor indices are excluded from the detector's real side and from the scorecard references.

**Generated templates.**
- Taken from existing pool files: `tstr/pool_beatgen.npz` (all four classes), then `pools2/pool_1.npz` (more minority templates).
- Each is used once per variant. If a class's bank is smaller than its donor count, templates are reused, and the reuse is recorded.

**Variants** (`generator_comparison/noise_bank.py`, `_noise_bank_variants.py`). Every variant is `donor + α·(T' − own template)@r` at the donor's own R-peaks, rebuilt to 12 leads with `reconstruct_12`.

| id | template T' | aligned to donor beat | α |
|---|---|---|---|
| N0 | random generated | no | 1 |
| N1 | random generated | yes | 1 |
| N2 | nearest generated (max correlation at ±4 samples) | yes | 1 |
| N1a0.25 / 0.50 / 0.75 | as N1 | yes | 0.25 / 0.50 / 0.75 |
| N2a0.25 / 0.50 / 0.75 | as N2 | yes | 0.25 / 0.50 / 0.75 |

The plan said "α sweep on the better of N1/N2". Both are swept instead, so no choice depends on an earlier result.

Alignment means a sub-sample lag from the cross-correlation of 8-lead QRS energy, a linear-interpolation shift, then the same re-zero and taper `_beat_gen_pool.py` applies to generated templates.

**N4 (conditional).**
- Only if any registered variant passes the gate: `_noise_bank_variants.py --warp-from <passing id>` builds `W_<id>`.
- Each sample is slowed by s ~ U(1.02, s_max), with s_max = min(1.25, class upper mean-RR bound / donor mean RR).
- QRS and PR windows stay rigid, ST–T stretches by √s, and TP absorbs the rest.
- Samples that cannot be slowed within their class are left out. `W_<id>` is scored with the same gate.

**Gate** (`_noise_bank_eval.py`). All of these must hold:
1. **Detector:** `realism_detector.run`, seeds 0 and 1, donors excluded, `auc_mean` ≤ 0.70.
2. **Scorecard:** every `scorecard.checks` item passes against a real_ref built like `scorecard.main` (rng 42, 256/class reference, 32/class real_ref, donors kept out). `not_memorized` is computed against non-donor train ECGs.
3. **Novelty:** median whole-signal correlation of each sample to its own donor ≤ 0.95 (the scorecard's memorisation threshold), and the share above 0.98 ≤ 1 %.
4. **TSTR:**
   - Balanced accuracy on the real validation split (`tstr.train_eval`, seeds 0–2) ≥ the same measure for `beat_gen` (`tstr/pool_beatgen.npz`), both at N = 200 per class.
   - The plan's bar was 0.68 at 440 per class. SBRAD donors are capped at 223, so 440 is impossible, and the bar becomes "at least the current generator at the same N". If `pool_beatgen.npz` is missing, the bar is 0.68.

**Outcomes.**
- **Some variant passes:** build a full pool from it, then register a separate study 3 (filtered vs random on that pool). Not part of this registration.
- **None passes:** the result is the realism–novelty curve (detector AUC against median donor correlation), reported as a finding. Stop.

## Procedure (your machine)

Inputs:
- `processed_data/X_{train,val,test}.npy` (notebook 02).
- `generator_comparison/pools2/pool_{1..4}.npz`, rebuilt with `_build_pools2.sh`, with SHA-256 matching `PREREGISTRATION_2.md`.
- `generator_comparison/tstr/pool_beatgen.npz` (`_gen_tstr_pools.py`).

```
py -3.14 -m pytest tests -q
bash run_retry.sh _run_threshold_control.py generator_comparison/threshold_control_run.log   # ~80 min GPU
py -3.14 _threshold_analysis.py                                                               # once
py -3.14 _noise_bank_variants.py                                                              # CPU
py -3.14 _noise_bank_eval.py                                                                  # ~1-2 h
```

After both parts: `report/THRESHOLD_CONTROL.md` and `report/NOISE_BANK.md` are written from the result files. `report/REPORT.md` is left unchanged unless you ask.
