# Quality-Aware Synthetic Data Augmentation for Robust Imbalanced ECG Classification

Research project on **PTB-XL**: generate class-conditional synthetic 12-lead ECGs and test whether **quality-aware
selection** of those samples improves imbalanced rhythm classification more than random synthetic augmentation.

> Full write-up: [`report/REPORT.md`](report/REPORT.md).

**Result in one paragraph.** Augmenting the minority classes improved test macro-F1 from 0.791 (real only) to
0.820–0.835 and SBRAD F1 from 0.49 to 0.58–0.61 (10 seeds). Quality-filtered vs randomly selected synthetic ECGs looked
slightly better at first (+0.010, p = 0.065), but a **pre-registered confirmatory study with 30 new seeds found no
difference** (+0.0014 macro-F1, 16/30 seeds, p = 0.79). A later code review found that those "filtered" subsets were
not the true top-scored samples for STACH and SBRAD. **Study 2** fixed this and was pre-registered: 4 independent pools ×
10 seeds, 160 runs. Filtered vs random was −0.0041 macro-F1 (16/40, p = 0.058), equivalent within ±0.01 under the
registered test. Random synthetic augmentation beat real-only training (+0.036, 40/40) and duplicating real minority ECGs
(+0.020). At this mix ratio, quality-aware selection does not improve the classifier over random selection
(report Section 8.3).

## Research question

> Can quality-aware selection of synthetic ECG samples improve imbalanced ECG classification compared with random
> synthetic augmentation?

Four rhythm classes: **SR / AFIB / STACH / SBRAD** (SR ≈ 85 %). The frozen real-only CNN-LSTM baseline reaches test
macro-F1 0.783; SBRAD is the weak class (F1 0.497).

## Pipeline (current)

```
PTB-XL 100 Hz ─► patient split + train-only z-score (frozen)
              ─► frozen CNN-LSTM baseline
              ─► beat templates from train ECGs ─► template generator (v-pred SSSD backbone, 8x64, L=70)
              ─► rule-based rhythm assembly (per-class RR model, residual/wander GPs, AFIB irregular RR)
              ─► final synthetic pool (AFIB 3200 / STACH 1800 / SBRAD 1400)
              ─► quality filter (typicality + frozen-baseline label prob + realism detector)
              ─► CNN-LSTM retrained: real | real+oversample | real+random | real+filtered | real+filtered(no label)  x 10 seeds
              ─► paired tests + patient-level bootstrap on the untouched test split
```

The whole-ECG diffusion generator from Week 4 (adapted SSSD-ECG) was investigated in depth and replaced: it produced
no regular heartbeat at this model size, and a classifier trained on its samples alone scores at chance on real data.
See the report, Sections 5–6.

## Setup

1. Python 3.14. Install a CUDA build of torch that matches your driver, then the
   rest: `pip install -r requirements.txt` (versions pinned).
2. Download PTB-XL 1.0.1 to `archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/` and run
   `02_PTBXL_Exploration.ipynb` to rebuild `processed_data/X_*.npy` (gitignored). Labels, metadata and train-only
   normalization stats are committed.
3. GPU steps were run on an RTX 3050 Laptop (4 GB). Long runs use `bash run_retry.sh <script.py> <log>`, which restarts
   from the newest checkpoint; create a file named `STOP` to stop it cleanly.

## Reproduce the main result

```bash
py -3.14 _beat_templates_all.py                     # median beat templates from train ECGs
py -3.14 _beat_oracle.py                            # fit assembly models (+ oracle pool)
bash run_retry.sh _run_template_gen.py generator_comparison/template_gen_run.log   # ~30 min GPU
py -3.14 _beat_gen_pool.py generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt 0,3200,1800,1400 generator_comparison/final_pool/pool.npz --weights ema --seed 7
py -3.14 -m generator_comparison.quality_filter generator_comparison/final_pool/pool.npz generator_comparison/final_pool/scores.npz
bash run_retry.sh _run_main_experiment.py generator_comparison/experiment_main_run.log            # ~25 min GPU
py -3.14 -m generator_comparison.stats --tag main
py -3.14 _report_figures.py
```

## Reproduce study 2 (pre-registered, report Section 8.3)

```bash
bash _build_pools2.sh                                # 4 pools (seeds 11/21/31/41) + scores, ~8 min each
py -3.14 _check_pools2.py                            # per-pool checks + SHA-256 -> pools2/checks.json
py -3.14 _gate_study2.py                             # validation-only gate G1/G2 -> pools2/gate.json
bash run_retry.sh _run_study2.py generator_comparison/experiment_study2_run.log   # 160 runs, ~2.5 h GPU
py -3.14 _study2_analysis.py                         # runs once; refuses if the result file exists
```

## Reproduce the generator comparison (report Sections 6–7)

```bash
bash run_retry.sh _run_vpred.py generator_comparison/sssd_vpred_run.log          # v-pred diffusion 8x64 (~3 h GPU)
py -3.14 _gen_tstr_pools.py 440                                                   # 440/class from v1 and v-pred step 5000
py -3.14 _beat_gen_pool.py generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt 440 generator_comparison/tstr/pool_beatgen.npz --weights ema --scorecard-dir generator_comparison/beat_gen
py -3.14 -m generator_comparison.tstr                                             # TSTR table (CPU)
py -3.14 -m generator_comparison.scorecard                                        # 9-check scorecard
py -3.14 -m generator_comparison.realism_detector generator_comparison/tstr/pool_beatgen.npz beat_gen
py -3.14 _filter_check.py generator_comparison/final_pool/pool.npz generator_comparison/final_pool/scores.npz 440
py -3.14 _beat_ablation.py                                                        # detector ablations (Section 5.3)
```

Evaluation tools: `generator_comparison/scorecard.py` (9-check pool scorecard), `realism_detector.py`
(real-vs-synthetic AUC), `tstr.py` (train on synthetic, test on real), `_filter_check.py` (filter sanity check).

## Frozen artifacts — do not retrain to "improve" later results

| Path | What it is |
|---|---|
| `baseline/cnn_lstm_baseline.pt`, `baseline/metrics.json` | Real-only CNN-LSTM (seed 42), test macro-F1 0.783 |
| `processed_data/*.json`, `*_metadata.csv`, `y_*.npy` | Split, labels, train-only stats |
| `generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt` (gitignored) | Selected template generator |

## Directory map

```
01–05_*.ipynb                 Weeks 1–4 notebooks (audit, preprocessing, baseline, SSSD adaptation, comparison)
baseline/                     Frozen CNN-LSTM + metrics
processed_data/               Split, labels, stats (X_*.npy gitignored)
synthetic_generation/         SSSD backbone (model, S4, train loop with EMA / v-pred / schedules),
                              beats.py (templates), assembly.py (rhythm assembly), leads.py (12-lead derivation)
generator_comparison/
  eval_metrics.py             HR / periodicity metric (lead_ii_hr_pt)
  scorecard.py, realism_detector.py, tstr.py
  quality_filter.py, augment_experiment.py, stats.py
  sssd_*/                     Diffusion experiments (each with metrics / RESULT.md)
  beat_templates/, template_gen/, beat_gen/, final_pool/   Selected generator and its outputs
  tstr/, scorecard/, experiment/                           Results
report/                       REPORT.md + figures
_run_*.py, _beat_*.py         Experiment entry points
```

## Citation — SSSD-ECG

The template generator uses the SSSD-ECG architecture (width reduced, v-prediction, rescaled noise schedule); it is not
the published model.

- Code: [AI4HealthUOL/SSSD-ECG](https://github.com/AI4HealthUOL/SSSD-ECG)
- Alcaraz, J. M. L., & Strodthoff, N. *Diffusion-based conditional ECG generation with structured state space models.*
  Computers in Biology and Medicine, 2023.

PTB-XL: Wagner, P., et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 2020.
