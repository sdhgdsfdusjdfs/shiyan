# AIA Data Cleaning Report

- Source rows: 26
- Formal rows kept by config filter: 24
- Rows flagged as outliers (strict joint rule): 0
- Final rows used in paper stats: 24

## Exclusion Rules
- Keep only `num_removes in {500, 1000}`.
- Keep only `sensitive_dim in {0,1,2}`, `seed in {0,1,2}`.
- Keep only rows with `removed_nodes == num_removes`.
- Outlier rule: exclude only if BOTH `delta_auc` and `delta_ap` are group-wise IQR outliers.

## Output Files
- d:\experiment\shiyan\result_exp1\feature_aia_summary\aia_feature_runs_formal_with_outlier_flags.csv
- d:\experiment\shiyan\result_exp1\feature_aia_summary\aia_feature_runs_clean_for_paper.csv
- d:\experiment\shiyan\result_exp1\feature_aia_summary\aia_feature_summary_clean_for_paper.csv