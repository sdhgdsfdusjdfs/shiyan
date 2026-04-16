# Node Unlearning Summary

| nr | strategy | AUC (U/R) | F1 (U/R) | Precision (U/R) | Recall (U/R) | Speedup | MIA (B/A) |
|---:|:---------|:----------|:---------|:----------------|:-------------|--------:|:----------|
| 100 | high_degree | 0.597409 / 0.597409 | 0.031882 / 0.031882 | 0.016722 / 0.016722 | 0.341359 / 0.341359 | 1.940x | 0.5000 / 0.5000 |
| 100 | random | 0.597595 / 0.597594 | 0.031940 / 0.031941 | 0.016752 / 0.016752 | 0.342218 / 0.342218 | 1.676x | 0.5000 / 0.5000 |
| 500 | high_degree | 0.597417 / 0.597419 | 0.031879 / 0.031878 | 0.016720 / 0.016720 | 0.341359 / 0.341359 | 1.568x | 0.5000 / 0.5000 |
| 500 | random | 0.597407 / 0.597407 | 0.031881 / 0.031881 | 0.016721 / 0.016721 | 0.341359 / 0.341359 | 2.316x | 0.5000 / 0.5000 |
| 1000 | high_degree | 0.597316 / 0.597318 | 0.031857 / 0.031857 | 0.016708 / 0.016708 | 0.341359 / 0.341359 | 1.743x | 0.5000 / 0.5000 |
| 1000 | random | 0.597483 / 0.597483 | 0.031879 / 0.031879 | 0.016720 / 0.016720 | 0.341359 / 0.341359 | 1.671x | 0.5000 / 0.5000 |

## Source Files

- nr=100, strategy=high_degree: `d:\experiment\result_exp1\dgraphfin_std_1e-02_lam_1e-03_nr_100_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_high_degree_retrain.pth`
- nr=100, strategy=random: `d:\experiment\result\dgraphfin_std_1e-02_lam_1e-03_nr_100_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_retrain.pth`
- nr=500, strategy=high_degree: `d:\experiment\gnn+unlearn\sgc_unlearn-main\result\dgraphfin_std_1e-02_lam_1e-03_nr_500_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_high_degree_retrain.pth`
- nr=500, strategy=random: `d:\experiment\result\dgraphfin_std_1e-02_lam_1e-03_nr_500_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_retrain.pth`
- nr=1000, strategy=high_degree: `d:\experiment\gnn+unlearn\sgc_unlearn-main\result\dgraphfin_std_1e-02_lam_1e-03_nr_1000_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_high_degree_retrain.pth`
- nr=1000, strategy=random: `d:\experiment\gnn+unlearn\sgc_unlearn-main\result\dgraphfin_std_1e-02_lam_1e-03_nr_1000_K_2_opt_Adam_mode_node_eps_1.0_delta_1e-04_bin_1_random_retrain.pth`