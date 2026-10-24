# 16x128 capacity check (2026-09-27)

Question: is model size what limits beat regularity? Only change vs `sssd_vpred` (8x64, 1.53M params):
16x128 (7.2M params, 4.7x). Same effective batch 32 (4 x accum 8), lr schedule, v-prediction,
schedule 5e-4->0.1, EMA warmup. ~9 ECG/s vs ~54 for 8x64, so stopped at step 3000 (STOP file).

## In-training evals (8/class, raw / EMA periodicity; real ~0.84)

| step | 8x64 | 16x128 |
|---|---|---|
| 1000 | 0.17 / 0.21 | 0.21 / 0.19 |
| 1500 | – | 0.22 / 0.25 |
| 2000 | 0.18 / 0.23 | 0.23 / 0.22 |
| 2500 | – | 0.25 / 0.25 |
| 3000 | 0.22 / 0.21 | 0.21 / 0.25 |
| 5000 (8x64 best) | 0.25 / 0.24 | – |

## Scorecard (32/class, step-3000 checkpoint)

| pool | periodicity | std | HR SR/AF/ST/SB | PSD dB | score |
|---|---|---|---|---|---|
| vpred16_ema | **0.26** (best full pool so far) | 0.67 | 55/57/60/33 | 0.7 | 4/9 |
| vpred16_raw | 0.19 | 0.79 | 34/62/44/36 | 1.2 | 4/9 |
| vpred5k_raw (8x64) | 0.22 | 0.88 | 56/62/63/42 | 1.0 | 6/9 |

## Conclusion

4.7x capacity reaches 8x64's best periodicity in about half the steps and separates class heart
rate better (STACH > SR > SBRAD in most evals; EMA pool 60 > 55 > 33), but beat regularity plateaus at
the same ~0.25 instead of climbing toward 0.5. Capacity helps a little; it is not the main blocker at this
budget. Extrapolating to 36x256 is a guess, not a measurement, but this result makes the cloud run a
weak bet for fixing periodicity. Amplitude also drifts low (0.67-0.79) as in the 8x64 run.
