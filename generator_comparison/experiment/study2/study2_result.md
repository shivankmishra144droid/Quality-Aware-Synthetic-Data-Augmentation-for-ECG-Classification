# Study 2 result (pre-registered)

| condition | test macro-F1 (mean of 40) |
|---|---|
| real | 0.8030 |
| oversample | 0.8199 |
| random | 0.8393 |
| filtered | 0.8352 |

| | comparison | mean diff | wins | dz | Wilcoxon p | adjusted p | 95 % CI | decision |
|---|---|---|---|---|---|---|---|---|
| H1 | filtered - random | -0.0041 | 16/40 | -0.30 | 0.0584 | 0.0584 (none (single primary test)) | -0.0116 to +0.0031 | **NOT SUPPORTED; equivalent within +/-0.01 (TOST)** |
| H2 | random - real | +0.0364 | 40/40 | 2.58 | 0.0000 | 0.0000 (Holm over H2-H3) | +0.0264 to +0.0465 | **SUPPORTED: random > real** |
| H3 | random - oversample | +0.0195 | 37/40 | 1.31 | 0.0000 | 0.0000 (Holm over H2-H3) | +0.0084 to +0.0305 | **SUPPORTED: random > oversample** |

H1 TOST (paired t, bounds ±0.01): p_lower 0.0048, p_upper 0.0000.

Sensitivity (descriptive): per-pool mean differences H1 -0.0083, -0.0039, -0.0021, -0.0022 (pool-level t p 0.067); H2 +0.0374, +0.0346, +0.0338, +0.0397 (pool-level t p 0.000); H3 +0.0252, +0.0141, +0.0163, +0.0222 (pool-level t p 0.005).

## Secondary (descriptive only, no inference)

| condition | SBRAD F1 | recall SR / AFIB / STACH / SBRAD | AUPRC SR / AFIB / STACH / SBRAD |
|---|---|---|---|
| real | 0.493 | 0.93 / 0.93 / 0.90 / 0.81 | 0.996 / 0.954 / 0.937 / 0.761 |
| oversample | 0.543 | 0.94 / 0.93 / 0.88 / 0.79 | 0.996 / 0.951 / 0.933 / 0.750 |
| random | 0.610 | 0.96 / 0.92 / 0.91 / 0.77 | 0.997 / 0.958 / 0.937 / 0.756 |
| filtered | 0.604 | 0.95 / 0.93 / 0.90 / 0.77 | 0.997 / 0.952 / 0.939 / 0.758 |

- H1 SBRAD F1 difference: -0.0062, 18/40 wins
- H2 SBRAD F1 difference: +0.1168, 40/40 wins
- H3 SBRAD F1 difference: +0.0666, 34/40 wins
