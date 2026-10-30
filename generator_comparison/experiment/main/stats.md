# Statistics (main)

| condition | test macro-F1 (mean ± sd over seeds) | 95% CI (patients x seeds) | test SBRAD F1 | 95% CI |
|---|---|---|---|---|
| real | 0.791 ± 0.022 | 0.765–0.815 | 0.491 ± 0.073 | 0.410–0.567 |
| oversample | 0.814 ± 0.013 | 0.790–0.836 | 0.545 ± 0.047 | 0.470–0.612 |
| random | 0.820 ± 0.017 | 0.795–0.844 | 0.583 ± 0.039 | 0.507–0.653 |
| filtered | 0.830 ± 0.012 | 0.806–0.853 | 0.605 ± 0.042 | 0.526–0.675 |
| filtered_nolabel | 0.835 ± 0.011 | 0.812–0.858 | 0.610 ± 0.032 | 0.531–0.680 |

| family | comparison | metric | mean diff | dz | Wilcoxon p | Holm p | paired t p | wins | bootstrap 95% CI of diff (patients x seeds) |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | filtered - random | macro_f1 | +0.0099 | 0.63 | 0.0645 | — | 0.0757 | 8/10 | -0.0018 to +0.0213 |
| A: augmentation vs real-only | oversample - real | macro_f1 | +0.0230 | 0.75 | 0.1055 | 0.2109 | 0.0424 | 6/10 | +0.0035 to +0.0414 |
| A: augmentation vs real-only | oversample - real | sbrad_f1 | +0.0539 | 0.56 | 0.1055 | 0.2109 | 0.1084 | 7/10 | -0.0079 to +0.1117 |
| A: augmentation vs real-only | random - real | macro_f1 | +0.0291 | 1.15 | 0.0098 | 0.0391 | 0.0053 | 9/10 | +0.0107 to +0.0466 |
| A: augmentation vs real-only | random - real | sbrad_f1 | +0.0918 | 1.28 | 0.0098 | 0.0391 | 0.0029 | 8/10 | +0.0415 to +0.1429 |
| A: augmentation vs real-only | filtered - real | macro_f1 | +0.0390 | 1.51 | 0.0020 | 0.0156 | 0.0010 | 10/10 | +0.0203 to +0.0577 |
| A: augmentation vs real-only | filtered - real | sbrad_f1 | +0.1140 | 1.28 | 0.0059 | 0.0293 | 0.0029 | 9/10 | +0.0564 to +0.1764 |
| A: augmentation vs real-only | filtered_nolabel - real | macro_f1 | +0.0446 | 1.65 | 0.0039 | 0.0273 | 0.0006 | 9/10 | +0.0253 to +0.0637 |
| A: augmentation vs real-only | filtered_nolabel - real | sbrad_f1 | +0.1191 | 1.49 | 0.0039 | 0.0273 | 0.0011 | 9/10 | +0.0660 to +0.1779 |
| B: synthetic vs duplicated real | random - oversample | macro_f1 | +0.0061 | 0.28 | 0.5566 | 0.5566 | 0.4045 | 7/10 | -0.0100 to +0.0218 |
| B: synthetic vs duplicated real | random - oversample | sbrad_f1 | +0.0378 | 0.62 | 0.0195 | 0.0586 | 0.0798 | 9/10 | -0.0033 to +0.0845 |
| B: synthetic vs duplicated real | filtered - oversample | macro_f1 | +0.0160 | 0.99 | 0.0273 | 0.0586 | 0.0120 | 8/10 | +0.0031 to +0.0296 |
| B: synthetic vs duplicated real | filtered - oversample | sbrad_f1 | +0.0600 | 1.23 | 0.0059 | 0.0293 | 0.0037 | 9/10 | +0.0242 to +0.1005 |
| B: synthetic vs duplicated real | filtered_nolabel - oversample | macro_f1 | +0.0217 | 1.47 | 0.0098 | 0.0391 | 0.0012 | 8/10 | +0.0084 to +0.0351 |
| B: synthetic vs duplicated real | filtered_nolabel - oversample | sbrad_f1 | +0.0651 | 1.39 | 0.0039 | 0.0234 | 0.0017 | 9/10 | +0.0268 to +0.1060 |
| C: filter vs random (secondary) | filtered - random | sbrad_f1 | +0.0222 | 0.53 | 0.1055 | 0.2109 | 0.1300 | 8/10 | -0.0087 to +0.0525 |
| C: filter vs random (secondary) | filtered_nolabel - random | macro_f1 | +0.0156 | 1.06 | 0.0195 | 0.0586 | 0.0086 | 9/10 | +0.0032 to +0.0269 |
| C: filter vs random (secondary) | filtered_nolabel - random | sbrad_f1 | +0.0273 | 0.56 | 0.1055 | 0.2109 | 0.1112 | 7/10 | -0.0070 to +0.0603 |
| exploratory | filtered_nolabel - filtered | macro_f1 | +0.0056 | 0.77 | 0.0371 | — | 0.0375 | 8/10 | -0.0029 to +0.0147 |
| exploratory | filtered_nolabel - filtered | sbrad_f1 | +0.0051 | 0.16 | 0.7695 | — | 0.6250 | 5/10 | -0.0200 to +0.0310 |

Primary comparison power (paired t, two-sided, alpha 0.05): dz = 0.63, power at n = 10: 0.43; seeds needed for 80 % power: 22.
