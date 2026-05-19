# Benchmark Runner

This directory keeps one public experiment entry point:

```text
benchmarks/run_benchmark.py
```

It is a lightweight benchmark runner for LG-FGT and common tabular baselines.
It supports fixed train/validation/test splits, multiple seeds, optional sample
caps, regression target transforms for deep models, and optional early stopping
for neural baselines.

## Supported Datasets

```text
covertype
adult_income
bank_marketing
california_housing
bike_sharing
abalone
wine_quality_red
airfoil_self_noise
energy_efficiency
qsar_fish_toxicity
breast_cancer
diabetes
electricity
helena
higgs
jannis
phoneme
wine_classification
synthetic_classification
synthetic_regression
```

The synthetic datasets are smoke tests only. They should not be used as
benchmark evidence.

## Supported Models

Core baselines:

- `logistic_regression`
- `ridge`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `xgboost`
- `lightgbm`
- `catboost`

Deep tabular models:

- `category_embedding`
- `ft_transformer`
- `tab_transformer`
- `gandalf`
- `feature_gate_transformer`
- `feature_gate_transformer_adaptive`

LG-FGT ablations:

- `feature_gate_transformer_no_gating`
- `feature_gate_transformer_no_branch_gate`
- `feature_gate_transformer_no_raw_shortcut`
- `feature_gate_transformer_local_only`
- `feature_gate_transformer_global_only`
- `feature_gate_transformer_no_ensemble`
- `feature_gate_transformer_cls_interaction_focus`
- `feature_gate_transformer_cls_conservative`

## Commands

Fast smoke test:

```powershell
python benchmarks/run_benchmark.py --dataset synthetic_classification --models logistic_regression feature_gate_transformer_adaptive --sample-size 1000 --max-epochs 3 --accelerator cpu
```

Single classification dataset:

```powershell
python benchmarks/run_benchmark.py --dataset adult_income --models logistic_regression random_forest ft_transformer feature_gate_transformer_adaptive --sample-size 50000 --max-epochs 30 --accelerator gpu --seeds 42 43 44
```

Single regression dataset with standardized neural targets:

```powershell
python benchmarks/run_benchmark.py --dataset california_housing --models ridge random_forest ft_transformer feature_gate_transformer_adaptive --sample-size 15000 --max-epochs 30 --target-transform standard --accelerator gpu --seeds 42 43 44
```

Resume an interrupted run:

```powershell
python benchmarks/run_benchmark.py --dataset adult_income --models logistic_regression feature_gate_transformer_adaptive --sample-size 50000 --max-epochs 30 --resume
```

## Outputs

By default, outputs are written under `benchmark_outputs/`, which is ignored by
Git. A run directory contains raw per-model results, leaderboards, progress
metadata, and environment information.

Do not commit downloaded datasets, local result folders, training logs, or model
checkpoints to the public repository.
