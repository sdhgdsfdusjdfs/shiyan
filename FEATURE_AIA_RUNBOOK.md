# Feature Unlearning AIA Runbook

## What was added

- New script: `gnn+unlearn/sgc_unlearn-main/attribute_inference_eval.py`
- Goal: run a baseline Attribute Inference Attack (AIA) before/after feature unlearning.
- Extra downloads: **not required** (uses existing `torch` + `scikit-learn`).
- `--forget_mode`:
  - `full_vector`: zero all features of removed nodes (may cause score collapse in some sparse/debug settings).
  - `sensitive_dim`: zero only the sensitive feature dimension (recommended for field-level deletion analysis).

## Dataset path

Current canonical DGraphFin raw file:

`D:\experiment\PyG_datasets\DGraphFin\raw\dgraphfin.npz`

`DGraphFin` loader now auto-resolves this root path.

## Quick smoke test (recommended first)

```powershell
cd D:\experiment\shiyan\gnn+unlearn\sgc_unlearn-main
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py `
  --data_dir D:\experiment\shiyan\PyG_datasets `
  --dataset dgraphfin `
  --num_removes 50 `
  --node_delete_strategy random `
  --prop_step 2 `
  --sensitive_dim 0 `
  --forget_mode sensitive_dim `
  --debug_sample_size 3000 `
  --seed 0 `
  --device -1 `
  --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
```

## Formal runs (full graph)

```powershell
cd D:\experiment\shiyan\gnn+unlearn\sgc_unlearn-main

# random strategy
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy random --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 0 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy random --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 1 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy random --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 2 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv

# high_degree strategy
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy high_degree --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 0 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy high_degree --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 1 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
D:\experiment\shiyan\.venv_cuda\Scripts\python.exe .\attribute_inference_eval.py --data_dir .\PyG_datasets --dataset dgraphfin --num_removes 500 --node_delete_strategy high_degree --prop_step 2 --sensitive_dim 0 --forget_mode sensitive_dim --seed 2 --device 0 --out_csv D:\experiment\shiyan\result\aia_feature_runs.csv
```

## Output interpretation

Script prints:

- `[AIA Before] ...`
- `[AIA After ] ...`
- `[AIA DIAG] ...` (score std + unique probabilities)
- `Delta(after-before): ...`

For privacy gain, we usually expect `AUC/AP` to decrease after feature unlearning.
