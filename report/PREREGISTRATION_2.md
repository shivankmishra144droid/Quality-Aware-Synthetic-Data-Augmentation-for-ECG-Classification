# Pre-registration 2 (DRAFT, not registered): quality-filtered vs random synthetic augmentation, corrected selection

**Status: DRAFT.** This document becomes a registration only when it is committed, together with the code and
frozen inputs it names, *after* every item in "Preconditions" is done and *before* any run under this protocol.
Until then nothing here is binding, and no part of this study may touch the test split.

## Why a new study
Pre-registration 1 (`report/PREREGISTRATION.md`, seeds 100–129) found no filtered-vs-random difference (p = 0.79).
A code review on 2026-10-28 found that it did not test the stated hypothesis for two of three classes:

1. **Selection bug (protocol deviation in studies 1 and the exploratory run).** `augment_experiment.pick` took the
   top-1,056 per class and then the first `k` *by pool position*. Overlap with the true top-k: AFIB 1.00, STACH 0.54,
   SBRAD 0.38. Mean score of the SBRAD subset used: 0.561 (true top-k 0.722, random 0.500). The "filtered" arm was
   therefore closer to random than to the filter for STACH and SBRAD.
2. **Realism component depended on the CV fold.** Raw log-odds from 5 fold detectors were ranked together; detector
   scales differed (median −13.2 to −10.4), so the top-third share ranged 0.17–0.53 by fold.
3. **Rhythm sampling was unbounded.** Sampled mean RR could fall outside the class definition (true placed-R rate out
   of class ≈ 5 % for STACH and SBRAD). Smaller than first reported: the earlier "26 % of SBRAD ≥ 60 bpm" figure was
   mostly the HR estimator misreading synthetic signals (it is within 10 % of the true rate for only ~56–72 % of
   synthetic STACH/SBRAD).

Study 1's result stands as reported for what was actually run; it is not re-analysed or overwritten. This study tests
the hypothesis as originally intended.

## Preconditions (all must be done and committed before registration)
- [x] P1. `pick` selects the per-class top-k (`select(scores, y, counts)`); regression test
      `tests/test_selection_and_rhythm.py::test_pick_*`.
- [x] P2. Realism ranked within (fold, class) (`quality_filter.realism_within_fold`); test
      `test_realism_within_fold_removes_fold_offset`.
- [x] P3. Rate kept inside the class (`assembly.MEAN_RR_BOUNDS`: SR 0.6–1.0 s, STACH 0.3–0.6, SBRAD 1.0–2.0,
      AFIB 0.3–2.0): the sampled mean RR is rejection-sampled into the bounds, and the whole R-peak series is redrawn
      until the realised mean RR of the placed beats is strictly inside them; tests
      `test_rhythm_mean_rr_within_class_bounds`, `test_realised_rate_of_placed_beats_within_class`.
- [x] P4. **Fixed optimisation budget** (`augment_experiment.train_steps`, `--steps`). Every condition trains for the
      same number of optimizer steps (not epochs), S = 6,250 steps of batch 64 (≈ 25 epochs of the largest
      augmented set), validation every 250 steps, LR halved after 3 evaluations without improvement, no early
      stopping; the state with the best validation macro-F1 is kept. Removes the confound that augmented sets get
      ~15 % more steps per epoch and that many study-1 runs stopped at the 25-epoch cap.
- [x] P5. **Determinism** (`--deterministic`; implemented). *G2 passed 2026-10-29: pool 1, seed 200, filtered, full
      6,250 steps, run twice in fresh folders: all 25 validation evaluations identical (val macro-F1 0.820, best step
      5,500, ~53 s per run). `generator_comparison/pools2/gate.json`.* `torch.use_deterministic_algorithms(True)`,
      `torch.backends.cudnn.deterministic = True`, `benchmark = False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. A rerun of
      one seed must reproduce its validation metrics exactly; this check is recorded before registration. It runs with
      `--val-only` (`_run_study2.py --val-only 1 200 filtered`; the test split is never loaded). Smoke check on the old
      final pool, 500 steps, run twice: identical validation history.
- [x] P6. **Pools.** Build 4 independent pools with the bounded generator (`bash _build_pools2.sh`: `_beat_gen_pool.py ... --seed 11/21/31/41`, spaced so no
      diffusion-noise seed is shared between pools,
      0,3200,1800,1400 each) and score each with the corrected `quality_filter`. SHA-256 of every pool and score file
      recorded below. *Done 2026-10-29: all 4 pools PASS `_check_pools2.py` (exact top-k, exact replay, true rate
      out of class 0 %, no non-finite values); hashes in `generator_comparison/pools2/checks.json`. The rhythm model
      redraws until the realised rate of the placed beats is in class (bounding only the sampled mean left ~5 % of
      STACH out of class in a first build, which was discarded).*
- [x] P7. **Validation gate (validation split only).** *G1 passed 2026-10-29: top − random TSTR balanced accuracy
      +0.049 / +0.041 / +0.035 / +0.013 (pools 1–4; 4 of 4 won, mean +0.035); top − bottom +0.12 to +0.19.
      `generator_comparison/pools2/filter_check_1..4.md`, `gate.json`.* For each pool, run `_filter_check.py` (TSTR on real val:
      top-k vs random-k vs bottom-k, 3 seeds; run and evaluated by `_gate_study2.py`, rule fixed before any gate
      result was seen). Gate: top > random balanced accuracy in **at least 3 of the 4 pools and** a mean
      (top − random) > 0. (A mean-only rule would pass about half the time for a useless filter.) If
      the gate fails, the study is **not run** and that is reported; the filter is not tuned and retried under this
      registration.
- [ ] P8. Frozen analysis script `_study2_analysis.py` and launcher `_run_study2.py`, written and committed before the run, implementing
      exactly the analysis below, refusing to run on an incomplete result set.

## Hypotheses
- **H1 (primary):** filtered ≠ random in test macro-F1 (expected filtered > random).
- **H2:** random synthetic ≠ real-only (re-tests the exploratory "augmentation helps" claim on fresh seeds).
- **H3:** random synthetic ≠ oversampled real (does synthetic content add anything beyond more minority examples?).

## Design
- Conditions: `real`, `oversample`, `random`, `filtered` (no `filtered_nolabel`). Ratio 1.0
  (+1,056 AFIB, +568 STACH, +446 SBRAD); SR gets nothing.
- Units: 4 pools × 10 seeds = **40 paired units**. `pool_1` uses seeds 200–209, `pool_2` 210–219, `pool_3` 220–229,
  `pool_4` 230–239 (never used before); results in `experiment/study2/pool<k>/`. Within a unit all four conditions share the seed (same init and, where sizes match, same batch order).
- `filtered`: per-class top-k of the pool's `score`, fixed within a pool. `random`: uniform per class, rng
  1000 + seed.
- Recipe: as study 1 except P4 (fixed steps) and P5 (determinism). Class-weighted CE from each condition's labels.
- A crashed run is rerun with the same seed. No run is excluded. The run prints validation metrics only.

## Sample size
Smallest effect of interest: **+0.010 test macro-F1** (the exploratory estimate, and about the gap a practitioner would
care about). Planning SD of the paired difference: 0.019 (study 1, filtered − random; only its spread is used, not its
effect). dz = 0.52; paired t power at n = 40 is 0.89 (≈ 0.85 for Wilcoxon at ARE 0.955). Pool-to-pool variation may
raise the SD; that risk is accepted and stated now rather than corrected by adding units later.

## Stopping rule
Exactly 40 units × 4 conditions = 160 runs. No interim look at test metrics.

## Analysis (frozen in P8)
- Outcome: test macro-F1 per run. Paired differences per unit.
- Test: two-sided Wilcoxon signed-rank on the 40 differences for each of H1–H3. **H1 is the single primary test at
  α = 0.05, unadjusted.** H2 and H3 form one secondary family with **Holm correction over H2–H3**, family-wise
  α = 0.05. (Holm over all three would test H1 at 0.0167 and cut its power from 0.89 to 0.78.)
- Decision per hypothesis: supported if its adjusted p (H1: unadjusted) < 0.05; direction from the mean difference.
- Equivalence (H1 only): if not significant, TOST with paired t at bounds ±0.010; "equivalent within ±0.01" may be
  claimed only if both one-sided p < 0.05.
- Reported with every decision: mean and median difference, wins, dz, 95 % CI from a three-level bootstrap (test
  patients × pools × seeds-within-pool, 2,000 resamples, seed 0).
- Sensitivity (descriptive, does not change decisions): per-pool mean differences and a one-sample t-test on the 4
  pool means (treats the pool as the unit; a mixed model was dropped because statsmodels is not in the environment).

## Secondary (descriptive only)
Computed by the frozen analysis script and reported without inference: mean SBRAD F1, per-class recall and per-class
AUPRC (one-vs-rest, from the saved probabilities) per condition, and the SBRAD F1 difference and wins for H1–H3.
The filtered-subset overlap with the true top-k is a check, not a result: it is 1.00 for every pool
(`pools2/checks.json`).

## Known limitations, stated in advance
- **What the validation gate does and does not show.** In TSTR (synthetic minorities only, same synthetic SR in every
  subset), the top-scored subsets raised minority recall and lowered SR recall in all 4 pools (SR recall 0.41–0.52
  top vs 0.56–0.70 random). Part of the filter's validation advantage is therefore a shift toward predicting the
  minority classes. With 11,741 real SR ECGs in the augmented training set this may not carry over; H1 tests whether
  it does. Pool 4's margin (+0.013) is within its seed spread (± 0.03).
- The test split was already analysed in study 1 and the exploratory run. Fresh seeds, fresh pools and one
  pre-specified analysis are the available safeguard; results are not claims about an untouched test set.
- One generator family, one classifier, one ratio. Results do not extend to other ratios or filters; those are new studies.
- The custom patient-level split is not PTB-XL's recommended `strat_fold` split, so absolute numbers are not
  comparable with the literature.

## Frozen inputs (SHA-256)
Paths relative to `generator_comparison/` unless noted. The training run records the pool and score hashes in every
`config.json` and refuses to resume if they change.
```
pools2/pool_1.npz                           bf7daeca6e6ef95d0c6f643885d746246af7037ec5b976560c6361e356cd2141
pools2/pool_2.npz                           e428a704812b61f06c30f4b52762083a0e2d575265faf9c48b4a0136c24bcbb1
pools2/pool_3.npz                           95892db476faacdb7855d76da946f052a604c14c923f85bd486fa5f988b8c7c8
pools2/pool_4.npz                           1d05494af6065e8dbf933d959c89e05450106854a231d95c38c389f108ab6b27
pools2/scores_1.npz                         6c9df2113470c01027a162c96575323052f035021118a406cf62bd6480bf7a8b
pools2/scores_2.npz                         be82eaa0397eb59db72ca45869e2de8d86ef6ad26853ae3676e8ebd3b55d61a9
pools2/scores_3.npz                         e3e5a3d2d4a692258376d9e4f6c057ff919498197c7b51a4107e7618cba66ffa
pools2/scores_4.npz                         2a779e1aeb7a87db29f70b20d08b302769978ed5b862c2f30136d3e451211cda
template_gen/checkpoints/tpl_8x64_last.pt   19a8e764681f0a0a897a7edf9d0f46565798d686948aa1d43ba076801ad979df
beat_templates/assembly_models.pkl          b2b133a6b2289c611accdbdafe0b4f315ea361fb170e2fcfde415507337fa9f5
baseline/cnn_lstm_baseline.pt (repo root)   51466880ac3c50a9d6957f1d90be58db72fb2f06a40bb94704ba0751fbfd327e
```
Generator and baseline hashes are identical to pre-registration 1.

## Environment
Exact reproducibility (P5) holds for this hardware and software only: NVIDIA GeForce RTX 3050 Laptop GPU, driver
591.59, Python 3.14, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19), numpy 2.5.2, scipy 1.17.1, scikit-learn 1.9.0
(Windows 11). On other hardware results will differ slightly; the protocol does not depend on bit-exactness.
