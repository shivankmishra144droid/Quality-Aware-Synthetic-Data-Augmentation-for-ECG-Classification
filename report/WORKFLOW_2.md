# Workflow for study 2 (corrected filtered-vs-random study)

The order of work from now until the write-up. The design itself is in `report/PREREGISTRATION_2.md`; this file says
**who does what, in which order, and what has to be true before moving on.**

Ground rules for every phase:
- **The test split is opened only in Phase 5 and 6.** Any script that would compute a test metric before then is a bug.
- Each phase ends with a commit. A phase is done only when its exit check passes.
- If a gate fails, stop and write it down. Do not tweak and retry inside this study.

Steps 1–4 of the pipeline (data, split, baseline, template generator, rhythm models) are **reused unchanged**.

```
Phase 0  Freeze and disclose        ~0.5 h   no GPU
Phase 1  Finish the code            ~2-3 h   no GPU, no test
Phase 2  Build 4 pools + scores     ~1.5 h   GPU, no test
Phase 3  Validation gate            ~1 h     GPU, val only        <- GO / NO-GO
Phase 4  Register                   ~0.5 h   no GPU
Phase 5  Run the study              ~2.5 h   GPU, test saved but not looked at
Phase 6  Analyse (once)             ~0.2 h
Phase 7  Report                     ~2 h
```

---

## Phase 0: Freeze and disclose
**Goal:** lock in the fixes so far and record the old bug honestly before anything new runs.

1. Commit the drafted fixes: `pick`/`select`, within-fold realism, RR bounds, `tests/`, draft `PREREGISTRATION_2.md`,
   this file.
2. Add a "Protocol deviation" note to `report/REPORT.md` §8 / §8.1: the filtered arm was not the true top-k for
   STACH (54 %) and SBRAD (38 %), so study 1 did not test H1 as written. Record the fold-dependent realism score,
   the unbounded rhythm sampling and the HR-estimator reliability. Do not re-run or re-analyse study 1.
3. Note at the top of `report/PREREGISTRATION.md` that a deviation was found later (point to REPORT); leave the
   rest untouched.

**Exit check:** `py -3.14 -m pytest tests -q` passes; commit made.

## Phase 1: Finish the code (no GPU, no test)
**Goal:** everything the study needs exists and is tested before any data is built.

| # | Change | File |
|---|---|---|
| 1.1 | Fixed step budget (P4): train for S = 6,250 steps of batch 64, validate every 250 steps, keep the best-val checkpoint. Same S for every condition. | `augment_experiment.py` |
| 1.2 | Determinism (P5): `use_deterministic_algorithms(True)`, cuDNN deterministic, `benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. | `augment_experiment.py` |
| 1.3 | `--val-only` flag: skip the test forward pass and save no test files. Used in Phase 3. | `augment_experiment.py` |
| 1.4 | Pools: the launcher calls the experiment once per pool; `pool_k` uses seeds 200+10(k−1) … 209+10(k−1), results in `experiment/study2/pool<k>/`. | `_run_study2.py` |
| 1.5 | `_filter_check.py` takes an output path, so the 4 pools do not overwrite each other. | `_filter_check.py` |
| 1.6 | Launcher `_run_study2.py` (for `run_retry.sh`, resumes by skipping finished runs). | new |
| 1.7 | Frozen analysis `_study2_analysis.py`: Wilcoxon for H1–H3 (H1 primary at α 0.05, Holm over H2–H3), TOST ±0.01 for H1, 3-level bootstrap, per-pool sensitivity. Refuses to run unless all 160 runs exist. Test it on **fake** result files only. | new |
| 1.8 | `.gitignore`: `generator_comparison/pools2/*.npz`. | `.gitignore` |
| 1.9 | Tests: fixed step count equal across conditions; `--val-only` writes no test files; analysis script on a fake results file gives the expected decision. | `tests/` |

**Exit check:** all tests pass. A 2-step smoke run with `--val-only` finishes and creates no `test_prob_*` file.

## Phase 2: Build the data (GPU, no test)
**Goal:** 4 independent pools from the frozen generator, scored with the corrected filter.

```bash
bash _build_pools2.sh > generator_comparison/pools2/build.log 2>&1
py -3.14 _check_pools2.py
```
The build makes `pool_1..4.npz` with `--seed 11 / 21 / 31 / 41` (spaced so no diffusion-noise seed is shared between
pools; 11–14 shared seeds across classes) and scores each one; ~8 min per pool in total.

**Checks per pool (`_check_pools2.py`, written to `pools2/checks.json`):**
- `pick('filtered')` overlap with true top-k = 1.00 for all three classes.
- Realism top-third share by fold: max − min < 0.10 (holds by construction after the within-fold ranking).
- Replay: regenerating every ECG from the stored template and the pool seed reproduces the pool exactly.
- True placed-R rate out of class ≤ 2 % (STACH, SBRAD). The rhythm model redraws until the realised rate is in class.
- Sanity: no NaN/inf, |z| > 20 rate near the old pool's 0.14 %.

**Exit check:** all 4 pools PASS; SHA-256 of pools, scores and frozen inputs in `checks.json` for Phase 4.

## Phase 3: Validation gate (val only) — GO / NO-GO
**Goal:** prove on validation data that the corrected filter is worth testing, and that runs repeat exactly.

Everything below is one command, `py -3.14 _gate_study2.py` (writes `generator_comparison/pools2/gate.json`):
1. Filter check per pool (TSTR on real val, 3 seeds, top vs random vs bottom) via `_filter_check.py`.
2. **Gate G1:** top > random balanced accuracy in **at least 3 of 4 pools**, and mean (top − random) > 0.
3. Determinism check: the study recipe (pool 1, seed 200, filtered, `--val-only`) run twice in fresh folders;
   **Gate G2:** identical val metrics and val history.
4. Write the gate results into the draft pre-registration (Preconditions P5–P7).

**If G1 fails:** stop. Report "the corrected filter does not pick more useful samples on validation". No test run.
**If G2 fails:** fix the determinism code (Phase 1) and repeat Phase 3. This does not touch test, so it is allowed.

## Phase 4: Register
1. Fill the SHA-256 table in `PREREGISTRATION_2.md` (4 pools, 4 score files, `assembly_models.pkl`, generator and
   baseline checkpoints) and tick P1–P8.
2. Remove "DRAFT" and commit **the prereg, the launcher and the frozen analysis script together**. That commit and its
   timestamp are the registration.
3. From here on: no edits to `augment_experiment.py`, the launcher, the analysis script or the prereg.

**Exit check:** registration commit hash written down; `git status` clean.

## Phase 5: Run the study
```bash
bash run_retry.sh _run_study2.py generator_comparison/experiment_study2_run.log
```
- 4 conditions × 4 pools × 10 seeds = **160 runs**, about 53 s each at 6,250 steps (measured in G2; ~2.4 h plus
  restarts).
- The log prints validation metrics only. Do not open `results.json` test fields.
- A crash is rerun with the same seed (the launcher does this). No run is excluded. Stop with a `STOP` file if needed.

**Exit check:** 160 entries in `experiment/study2/results.json`; the log has no unhandled errors.

## Phase 6: Analyse (once)
```bash
py -3.14 _study2_analysis.py
```
Run exactly once. The decisions for H1–H3 come from its output as written. Commit the result files immediately,
before reading further into the numbers.

## Phase 7: Report
1. New REPORT section "Study 2": design, the deviation that motivated it, the gate results, and the decisions for H1–H3
   with CIs and the TOST result for H1.
2. Anything looked at after Phase 6 (per-class plots, per-pool breakdowns) is labelled **post hoc**.
3. Update the abstract and conclusions to match, whichever way the result goes. Update the README commands and the
   project memory.

---

## Decision map
```
Phase 3 G1 fails ──► stop, report "filter not useful on val"   (no test run)
Phase 3 G2 fails ──► fix determinism, redo Phase 3
Phase 6 H1 significant, filtered > random      ──► quality-aware selection helps (this setting)
Phase 6 H1 not significant, TOST passes        ──► no gain larger than ±0.01
Phase 6 H1 not significant, TOST fails         ──► inconclusive; state the CI
H2 / H3 are reported the same way, independent of H1.
```
