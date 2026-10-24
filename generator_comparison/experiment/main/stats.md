# Statistics (main)

| condition | test macro-F1 (mean ± sd over seeds) | 95% CI (patient bootstrap) | test SBRAD F1 | 95% CI |
|---|---|---|---|---|
| real | 0.791 ± 0.022 | 0.769–0.811 | 0.491 ± 0.073 | 0.423–0.554 |
| random | 0.820 ± 0.017 | 0.797–0.841 | 0.583 ± 0.039 | 0.510–0.646 |
| filtered | 0.830 ± 0.012 | 0.806–0.851 | 0.605 ± 0.042 | 0.532–0.670 |
| filtered_nolabel | 0.835 ± 0.011 | 0.812–0.856 | 0.610 ± 0.032 | 0.536–0.676 |

| comparison | metric | mean diff | dz | Wilcoxon p | paired t p | wins | bootstrap 95% CI of diff |
|---|---|---|---|---|---|---|---|
| random - real | macro_f1 | +0.0291 | 1.15 | 0.0098 | 0.0053 | 9/10 | +0.0201 to +0.0372 |
| random - real | sbrad_f1 | +0.0918 | 1.28 | 0.0098 | 0.0029 | 8/10 | +0.0652 to +0.1182 |
| filtered - real | macro_f1 | +0.0390 | 1.51 | 0.0020 | 0.0010 | 10/10 | +0.0298 to +0.0481 |
| filtered - real | sbrad_f1 | +0.1140 | 1.28 | 0.0059 | 0.0029 | 9/10 | +0.0869 to +0.1407 |
| filtered_nolabel - real | macro_f1 | +0.0446 | 1.65 | 0.0039 | 0.0006 | 9/10 | +0.0355 to +0.0537 |
| filtered_nolabel - real | sbrad_f1 | +0.1191 | 1.49 | 0.0039 | 0.0011 | 9/10 | +0.0901 to +0.1481 |
| filtered - random | macro_f1 | +0.0099 | 0.63 | 0.0645 | 0.0757 | 8/10 | +0.0050 to +0.0152 |
| filtered - random | sbrad_f1 | +0.0222 | 0.53 | 0.1055 | 0.1300 | 8/10 | +0.0100 to +0.0346 |
| filtered_nolabel - random | macro_f1 | +0.0156 | 1.06 | 0.0195 | 0.0086 | 9/10 | +0.0098 to +0.0216 |
| filtered_nolabel - random | sbrad_f1 | +0.0273 | 0.56 | 0.1055 | 0.1112 | 7/10 | +0.0123 to +0.0419 |
| filtered_nolabel - filtered | macro_f1 | +0.0056 | 0.77 | 0.0371 | 0.0375 | 8/10 | -0.0000 to +0.0110 |
| filtered_nolabel - filtered | sbrad_f1 | +0.0051 | 0.16 | 0.7695 | 0.6250 | 5/10 | -0.0073 to +0.0174 |
