# Quality-Aware Synthetic Data Augmentation for Robust Imbalanced ECG Classification

Eight-week research project on **PTB-XL**: generate class-conditional synthetic 12-lead ECGs and test whether **quality-aware selection** of those samples improves imbalanced rhythm classification versus random synthetic augmentation (or real data only).

## Research question

> Can quality-aware selection of synthetic ECG samples improve imbalanced ECG classification compared with random synthetic augmentation?

Primary experiment: four rhythm classes **SR / AFIB / STACH / SBRAD**. We are **not** inventing a new generative architecture, replacing real data, or claiming SOTA on all seven PTB-XL rhythm codes.

**Status:** Weeks 1–4 are done (audit, preprocess, frozen CNN-LSTM baseline, adapted SSSD-ECG generation, generator comparison, quality-improvement pilots). **Week 5 quality filtering is not done.** Classifier experiments that mix synthetic pools into training have not started.

## Dataset, classes, and split

Source: [PTB-XL v1.0.1](https://physionet.org/content/ptb-xl/1.0.1/) (PhysioNet). We use **100 Hz**, 10 s recordings → shape `(12, 1000)`.

Labels come from SCP rhythm codes in `scp_codes`. A recording is kept only if **exactly one** of {SR, AFIB, STACH, SBRAD} is present (0 conflicts among these four). AFLT / SARRH / SVTAC are out of the primary experiment.

| | Recordings | Patients |
|--|----------:|---------:|
| Labelled 4-class | 19,759 | 17,320 |
| Train (70%) | 13,811 | 12,124 |
| Val (15%) | 2,960 | 2,598 |
| Test (15%) | 2,988 | 2,598 |

Split is **patient-level** (not PTB-XL `strat_fold`, which is recording-level). Train ∩ val ∩ test patient overlap is **0**. Stratification is on each patient’s primary rhythm so class mix is similar across splits (SR ~85% everywhere).

Per-split class counts:

| Split | SR | AFIB | STACH | SBRAD |
|-------|---:|-----:|------:|------:|
| Train | 11,741 | 1,056 | 568 | 446 |
| Val | 2,515 | 224 | 125 | 96 |
| Test | 2,526 | 234 | 133 | 95 |

**Normalization is train-only:** per-lead mean/std fitted on the train split, then applied to val/test (`processed_data/normalization_stats.json`). Do not recompute stats on the full dataset.

## Pipeline

```
PTB-XL (100 Hz) → patient split + train-only z-score
       → frozen real-only CNN-LSTM baseline
       → class-conditional generator (train split only)
       → synthetic pool (npy + metadata)
       → Week 5: quality filter  [NOT DONE]
       → later: train with real + (filtered vs random) synthetic
```

Generation never sees val/test recordings. Independent leads I, II, V1–V6 are modeled; III / aVR / aVL / aVF are reconstructed from Einthoven/Goldberger relations.

## Setup

1. Python 3.10+ recommended. GPU optional for the baseline (trained on CPU); generator training expects **CUDA PyTorch** on a small GPU.
2. Create a venv and install:

```bash
pip install -r requirements.txt
```

Install a CUDA build of `torch` that matches your driver, then the rest of `requirements.txt` (`pandas`, `numpy`, `matplotlib`, `scikit-learn`, `wfdb`, `jupyter`, `einops`, `opt_einsum`).

3. Download PTB-XL 1.0.1 and place it at:

```
archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
```

The raw archive is **not** in this repo (multi-GB). Waveform arrays `processed_data/X_*.npy` are also omitted (GitHub 100 MB limit). Rebuild them with `02_PTBXL_Exploration.ipynb` (dataset finalization sections). Labels, metadata CSVs, and `normalization_stats.json` **are** committed.

4. Run notebooks in order: `01` audit → `02` preprocess → `03` baseline (load frozen weights; do not retrain) → `04` generation → `05` generator comparison.

## Frozen artifacts — do not retrain to “improve” later weeks

These are Baseline V1 / Week 2–3 contracts. Changing them would make synthetic experiments incomparable.

| Path | What it is |
|------|------------|
| `baseline/cnn_lstm_baseline.pt` | Real-only CNN-LSTM (seed 42, best epoch 12) |
| `baseline/metrics.json` | Test Macro-F1 **0.783**, accuracy **0.918** |
| `baseline/class_weights.json` | Weighted CE from **train labels only** |
| `processed_data/*.json`, `*_metadata.csv`, `y_*.npy` | Split, labels, train-only stats |
| `processed_data/dataset_config.json` | Seed 42, 4-class rule, patient split |

Do **not** retune the CNN-LSTM so later synthetic runs look better. Architecture: Conv1d 12→32 k=7 → Conv1d 32→64 k=5 → LSTM 64 → Linear 4 (~47k params). Weak class: SBRAD (F1 0.50); main confusion SR→SBRAD.

## Generator comparison (honest)

Pilot, not a SOTA claim. Four candidates; **only one ran**.

| Generator | Runnable? | Outcome |
|-----------|-----------|---------|
| **SSSD-ECG (adapted)** | Yes | Selected. 8 residual layers × 64 channels (not the paper’s 36×256). Train-only, T=200, batch 1, AMP off. ~1.5M params, ~1 GB peak on RTX 3050 4 GB. 32 samples/class in the comparison pool. |
| TransDiffECG | No | No public code. Paper uses QTDB/LUDB, not PTB-XL. We did not substitute another model. |
| SE-Diff | No | Official repo exists, but needs unpublished latents / VAE weights / simulator prior and OpenAI embeddings. From-scratch 4-class training on our `X_train` is not a fair drop-in; published batch 4096 does not fit 4 GB. |
| FlowECG | No | No public implementation (GitHub search returned 0 repos). We did not reimplement flow matching on the SSSD sampler. PPGFlowECG is a different method and was not used. The paper’s 36×256 backbone also does not fit 4 GB (~8.2 GB in our SSSD-sized probe). |

**Why SSSD:** only leakage-free generator that matches this 4-class, 12-lead, 100 Hz, 4 GB setup.

Known gap after Week 4: synthetic amplitude is damped vs real (syn std ~0.5 vs real ~1.0) and class-conditional heart rate is weakly separated (HR MAE vs real ~29 bpm in the first pool). That is a generator-quality issue, not a stats-file bug.

Quality-improvement pilots (still **not** Week 5 filtering):

- **v2:** more iterations (4000 → 16000) at 8×64 — loss did not keep improving cleanly.
- **ckpt2000:** sample from the 2000-iter snapshot instead of last.
- **v3:** 16×128 capacity, 4000 iters.
- **CFG:** classifier-free guidance training (label dropout 0.1, 8×64, 4000 iters) — training finished; sampling/eval still in progress.

See `generator_comparison/metadata/selection.json` and each generator’s `NOT_RUNNABLE.json` or `metrics.json`.

## Directory map

```
01_PTBXL_Data_Audit.ipynb          Week 1
02_PTBXL_Exploration.ipynb         Week 2 preprocess + saved dataset
03_CNN_LSTM_Baseline.ipynb         Week 3 frozen baseline
04_Synthetic_ECG_Generation.ipynb  Week 4 SSSD adaptation
05_Generator_Comparison_Pilot.ipynb
preprocess_ptbxl.py                Small label-count helper
requirements.txt
baseline/                          Frozen CNN-LSTM + metrics
processed_data/                    Stats, labels, metadata (no X_*.npy in git)
synthetic_generation/              Adapted SSSD-ECG code
synthetic_data/                    Week 4 sample pool (small npy)
generator_comparison/
  sssd/                            Selected generator pool + checkpoint
  transdiff|sediff|flowecg/        Not-runnable records
  sssd_v2|sssd_ckpt2000|sssd_v3|sssd_cfg/  Quality pilots
archive/                           Local PTB-XL only (gitignored)
```

## Reproducibility

- Global seed **42** for split, baseline, and the selected SSSD run (other generator probes used 43/44 where noted).
- Hardware limit: **4 GB RTX 3050**. Official SSSD-ECG width **36×256 does not fit**; we use **8×64**. Do not describe 8×64 as the published architecture.
- Generator batch size 1, AMP off.
- Intermediate generator `.pt` snapshots are gitignored (tens to hundreds of MB). The frozen baseline weights and `generator_comparison/sssd/checkpoint.pt` are kept.

## Week 5 (not done)

Quality filtering of the synthetic pool (statistical / temporal / feature-space scores, then keep vs discard) has **not** been implemented. Downstream CNN-LSTM training with real + synthetic (quality vs random, same ratio) is also not done.

## Citation — SSSD-ECG

This repo adapts the official SSSD-ECG implementation (architecture and diffusion setup; width reduced for 4 GB VRAM):

- Code: [AI4HealthUOL/SSSD-ECG](https://github.com/AI4HealthUOL/SSSD-ECG)
- Alcaraz, J. M. L., & Strodthoff, N. *Diffusion-based conditional ECG generation with structured state space models.* Computers in Biology and Medicine, 2023.

PTB-XL: Wagner, P., et al. *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 2020. PhysioNet.
