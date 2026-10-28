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
- [x] P3. Mean RR bounded per class (`assembly.MEAN_RR_BOUNDS`: SR 0.6–1.0 s, STACH 0.3–0.6, SBRAD 1.0–2.0,
      AFIB 0.3–2.0), by rejection sampling.
- [x] P4. **Fixed optimisation budget** (`augment_experiment.train_steps`, `--steps`). Every condition trains for the
      same number of optimizer steps (not epochs), S = 6,250 steps of batch 64 (≈ 25 epochs of the largest
      augmented set), validation every 250 steps, LR halved after 3 evaluations without improvement, no early
      stopping; the state with the best validation macro-F1 is kept. Removes the confound that augmented sets get
      ~15 % more steps per epoch and that many study-1 runs stopped at the 25-epoch cap.
- [ ] P5. **Determinism** (`--deterministic`; implemented). `torch.use_deterministic_algorithms(True)`,
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
- [ ] P7. **Validation gate (validation split only).** For each pool, run `_filter_check.py` (TSTR on real val:
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
SBRAD F1, per-class recall, per-class AUPRC (from saved probabilities), and the filtered-subset overlap with true top-k
(must be 1.00; a check, not a result).

## Known limitations, stated in advance
- The test split was already analysed in study 1 and the exploratory run. Fresh seeds, fresh pools and one
  pre-specified analysis are the available safeguard; results are not claims about an untouched test set.
- One generator family, one classifier, one ratio. Results do not extend to other ratios or filters; those are new studies.
- The custom patient-level split is not PTB-XL's recommended `strat_fold` split, so absolute numbers are not
  comparable with the literature.

## Frozen inputs (SHA-256), filled in at registration
```
<pool_1..4.npz>          TBD
<scores_1..4.npz>        TBD
template_gen/checkpoints/tpl_8x64_last.pt   19a8e764681f0a0a897a7edf9d0f46565798d686948aa1d43ba076801ad979df
beat_templates/assembly_models.pkl          TBD
baseline/cnn_lstm_baseline.pt               51466880ac3c50a9d6957f1d90be58db72fb2f06a40bb94704ba0751fbfd327e
```
