# Option 3: whole-ECG diffusion at the published scale (gated phases)

New code only. Nothing in `synthetic_generation/`, the notebooks, `baseline/` or `processed_data/` is modified. This package only *imports* the official SSSD-ECG network copy (`synthetic_generation.model`, `s4`, `utils`).

**Why.** The Week-4 whole-ECG runs (8×64 and 16×128, 4k steps at batch 1) produced no regular heartbeat: about 130 bpm in every class at half the real amplitude. The published SSSD-ECG is far bigger (36×256, 100k steps × batch 6) and was never tested here. Its authors released synthetic ECGs and weights (MIT), so we can check whether a model at that scale is realistic **on the CPU**, before spending days training our own.

**Leakage rule.** The published model trained on PTB-XL folds 1–8, which cover about 80% of our val/test patients. Its samples are judged **only against folds 9–10 patients** and are **never** used for a classifier result. Converted copies live under the gitignored `archive/`.

## Phases

| # | Command (`python -m full_ecg_diffusion.<x>`) | Gate |
|---|---|---|
| 0 | `splits` | Writes `manifests/judge_manifest.json` (ecg_ids + hash). Run once, then commit it. |
| 1 | `public_data` | Infers lead order, units and labels from the data. Needs an unambiguous layout, label prevalence rank-corr ≥ 0.8, and sensible class HR (SBRAD < 60 < SR < 100 < STACH). |
| 2 | `conversion_control` | Exact round-trip (≤ 1e-4 z), recovery of a shuffled lead order, and a converted-real vs our-real detector ≤ 0.60. **Any failure stops the pipeline.** |
| 3 | `evaluate --pool archive/sssd_ecg_public/converted --real part:dev_unseen_by_both part:dev_unseen_by_sssd --copy-ref part:sssd_train_ref --layout <inspection.json> --out generator_comparison/sssd_full/phase3_public` | Detector ≤ 0.70 → **PASS**, continue. ≥ 0.95 → **FAIL**, switch to the real-noise-bank fallback. In between → **PARTIAL**, stop and decide. |
| 4a | `pretrained` | Loads the published weights and reproduces their samples. Published vs ours must give a detector ≤ 0.60, and the inferred channels must match. |
| 4b | `probe` | Times the paper step on this GPU and recommends local training or `kaggle_train.ipynb`. |
| 5 | `train --export-bundle …`, then `train --bundle …` | Paper recipe; only the data changes (our train split, so no val/test or judge patients). |
| 6 | `sample --ckpt …`, then `evaluate --pool generator_comparison/sssd_full/ours_pool --real part:dev_unseen_by_both --copy-ref split:train --layout …` | Same checks and bands. |
| 7 | `final_test --name ours --pool … --copy-ref split:train --layout …` | Registers first, runs on `judge_reserved` **once**, and refuses a rerun. |

Default paths: PTB-XL under `archive/ptb-xl-…-1.0.1/`, downloads under `archive/sssd_ecg_public/` (synthetic data: figshare DOI 10.6084/m9.figshare.21922947; weights: 10.6084/m9.figshare.21922875), outputs under `generator_comparison/sssd_full/`. Every CLI takes `--help`, and all path options can be overridden.

## Judge split (patient-level, our 4 classes)

- `judge_reserved`: 50% of the patients in folds 9–10 ∩ our test split, class-stratified, seed 42. **Phase 7 only**; every other entry point refuses it.
- `dev_unseen_by_both`: the rest of folds 9–10 ∩ val/test. Phases 3 and 6.
- `dev_unseen_by_sssd`: folds 9–10 ∩ our train split. Phase 3 only (our model trains on these).
- `sssd_train_ref`: folds 1–8. Reference for the published model's copying check.

## Checks (`checks/`, thresholds in `config.py`)

- **Detector**: a small 1D-CNN telling real from synthetic. Stratified 5-fold, patient-grouped AUC with a bootstrap CI, pooled and per class, plus a real-vs-real control that must stay ≤ 0.60.
- **Feature judge**: KS distance on HR, RR CV, RMSSD, QRS width, R/S amplitude and std per lead, band powers and inter-lead correlations. Pass if the pooled median KS ≤ 0.15 and the max ≤ 0.30. A real-vs-real floor is reported alongside.
- **Rhythm checks**: a multi-lead R-peak detector that must first pass calibration on real ECGs. Then: detectable fraction ≥ 95% of real; class median HR within 10 bpm of real; SBRAD < 60 < SR < 100 < STACH; AFIB more irregular than SR.
- **Copying**: nearest-neighbour correlation of each synthetic ECG to the generator's training records, compared with held-out real ECGs (flag > 0.95).

The real ECGs are passed through the same channel restriction and limb-lead rebuild as the synthetic ones (`--layout`), so a rebuild artifact cannot be the detector's cue.

## Self-test

```bash
python -m full_ecg_diffusion.selftest --work /tmp/fed_selftest
```

This builds a fake PTB-XL, fake published files and fake weights, then runs phases 0–7 through the real CLIs, about 10 minutes on CPU. It checks the inferred layout, conversion exactness, that the committed Week-4 pool fails (detector 1.00, rhythm fail), that the sealed part is refused, that a 36×256 checkpoint loads, that training resumes, and that the final test refuses a second run.

## Notes

- Downloads are opened safely: `.npy` with `allow_pickle=False`, weights with `torch.load(weights_only=True)`. Archives are never auto-extracted.
- The S4 layer registers some buffers as broadcast views. `nets.build_net` makes contiguous copies with identical values so current PyTorch can load checkpoints. Gradient checkpointing (`--grad-checkpointing`) gives bit-identical gradients and only saves memory.
