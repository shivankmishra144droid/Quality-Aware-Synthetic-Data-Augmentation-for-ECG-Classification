# Final synthetic pool (Phase 1.4)

`pool.npz` (gitignored, ~250 MB): X (6400, 1000, 12) z-scored 12-lead ECGs, y, templates (6400, 8, 70).
3x the synthetic ECGs the main experiment adds, so the quality filter has room to choose:
AFIB 3200 / STACH 1800 / SBRAD 1400 (no SR).

## Generator (frozen)
- Beat templates: v-prediction SSSD-backbone (8x64, s4_lmax 70), trained 15k steps on clean per-recording
  median beats from the train split (`_run_template_gen.py`). Checkpoint
  `generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt`, **EMA weights**.
  Final evals: nearest-real-template corr 0.89-0.93 (real-to-real reference 0.88-0.94), 0% copies (> 0.98).
- Assembly (rule-based, fitted on train only): per-class RR model, residual and wander Gaussian processes,
  respiratory modulation (`synthetic_generation/assembly.py`, models in `beat_templates/assembly_models.pkl`).

## Rebuild
```bash
py -3.14 _beat_templates_all.py
py -3.14 _beat_oracle.py            # fits and saves the assembly models
py -3.14 _beat_gen_pool.py generator_comparison/template_gen/checkpoints/tpl_8x64_last.pt 0,3200,1800,1400 generator_comparison/final_pool/pool.npz --weights ema --seed 7
```

## Known properties (see scorecard / TSTR / detector)
- TSTR (440/class, real val): balanced acc 0.678 vs 0.721 with real templates and 0.873 with real ECGs.
- Scorecard 8/9: typical amplitude ~12% above real (noise and wander scales are sampled independently).
- Realism detector AUC 1.00: the assembled residual is not beat-locked like real ECG residual.
- 9 of 6400 samples exceed |z| = 20 (0.14%; real train ECGs: ~1%).
