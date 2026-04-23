# Edge Link-Inference Strategy Comparison (Random vs High-Degree)

| num_removes | random Delta AUC (mean+/-std) | high_degree Delta AUC (mean+/-std) | random Delta AP (mean+/-std) | high_degree Delta AP (mean+/-std) | abs-gap AUC | abs-gap AP |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | -0.2968 +/- 0.0051 | -0.0115 +/- 0.0015 | -0.2660 +/- 0.0102 | -0.0117 +/- 0.0030 | 0.2853 | 0.2543 |
| 1000 | -0.3096 +/- 0.0125 | -0.0283 +/- 0.0005 | -0.2804 +/- 0.0131 | -0.0277 +/- 0.0021 | 0.2813 | 0.2527 |
| 2000 | -0.3052 +/- 0.0098 | -0.2016 +/- 0.0026 | -0.2797 +/- 0.0077 | -0.1410 +/- 0.0044 | 0.1036 | 0.1387 |
| 5000 | -0.2952 +/- 0.0040 | -0.1667 +/- 0.0004 | -0.2689 +/- 0.0060 | -0.1063 +/- 0.0029 | 0.1284 | 0.1626 |

Interpretation:
- random consistently yields a larger absolute Delta (stronger privacy-risk reduction) than high_degree at all tested scales.
- This supports the claim that deletion strategy changes security outcomes even when utility/efficiency are similar.