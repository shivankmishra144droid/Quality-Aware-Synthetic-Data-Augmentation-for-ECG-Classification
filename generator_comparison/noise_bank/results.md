# Real-noise-bank variants (pre-registered, part B)

Gate: detector <= 0.7, all scorecard checks, median donor corr <= 0.95 and share > 0.98 <= 0.01, TSTR balanced acc >= 0.686 (N=200/class).

TSTR references: TRTR 0.796, beat_gen 0.686

| variant | n | detector AUC | scorecard | median donor corr | share >0.98 | TSTR bal. acc | pass |
|---|---|---|---|---|---|---|---|
| N0 | 1496 | 0.959 | 9/9 | 0.485 | 0.001 | 0.777 | fail |
| N1 | 1496 | 0.955 | 9/9 | 0.622 | 0.001 | 0.768 | fail |
| N1a0.25 | 1496 | 0.727 | 8/9 | 0.978 | 0.460 | 0.791 | fail |
| N1a0.50 | 1496 | 0.919 | 8/9 | 0.905 | 0.011 | 0.780 | fail |
| N1a0.75 | 1496 | 0.941 | 9/9 | 0.779 | 0.003 | 0.784 | fail |
| N2 | 1496 | 0.859 | 8/9 | 0.882 | 0.006 | 0.771 | fail |
| N2a0.25 | 1496 | 0.552 | 8/9 | 0.993 | 0.913 | 0.807 | fail |
| N2a0.50 | 1496 | 0.679 | 8/9 | 0.970 | 0.223 | 0.806 | fail |
| N2a0.75 | 1496 | 0.793 | 8/9 | 0.933 | 0.018 | 0.767 | fail |

Passing: none.
