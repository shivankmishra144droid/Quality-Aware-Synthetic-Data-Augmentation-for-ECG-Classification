# Balanced-sampling run (single change vs v1)

8x64, 4000 iters, lr 2e-4, batch 1, no CFG. Only change: WeightedRandomSampler (~1000 draws/class instead of ~3400 SR / ~130 SBRAD).

Scored with the fixed metric `lead_ii_hr_pt` (see `generator_comparison/hr_rescore/`).

| | SR | AFIB | STACH | SBRAD | periodicity |
|--|--:|--:|--:|--:|--:|
| real train | 70.3 | 84.6 | 107.9 | 49.0 | 0.78-0.90 (AFIB 0.36) |
| v1 | 74.5 | 53.5 | 42.4 | 66.0 | 0.15 |
| balanced | 37.5 | 47.4 | 42.4 | 47.2 | 0.14 |

Std 0.45 (v1 0.49). Loss last 0.149.

**Verdict: no improvement.** Class imbalance is not the bottleneck. Synthetic Lead II has no regular QRS train
in any pool (peak-vs-ACF agreement 9-22% vs 87-97% real), so per-class HR numbers on synthetics are noise.
The generator fails at beat morphology/rhythm itself, before class conditioning matters.
