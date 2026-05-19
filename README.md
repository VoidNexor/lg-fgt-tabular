# LG-FGT: Local-Global Feature-Gate Transformer for Tabular Prediction

This repository is a research fork of `pytorch-tabular` with an added
Feature Gate Transformer model and a reproducible benchmark scaffold for
small-to-medium tabular prediction.

It is not an official release of the upstream `pytorch-tabular` project.

The project is best read as an open-source research prototype rather than as a
claim of universal state-of-the-art performance. Tree ensembles such as
CatBoost, LightGBM, XGBoost, ExtraTrees, and HistGradientBoosting remain strong
reference points for many tabular tasks. LG-FGT explores whether a compact
train-from-scratch neural model can combine local MLP features, transformer
interactions, controlled feature gating, and diagnostic feature-importance
signals in one practical tabular-learning pipeline.

## What Is Included

- `src/pytorch_tabular/models/feature_gate_transformer/`: LG-FGT model and
  configuration.
- `benchmarks/run_benchmark.py`: benchmark runner for classification and
  regression datasets.
- `tests/test_feature_gate_transformer.py`: smoke tests for the LG-FGT model,
  embedding transformer compatibility, and adaptive gating controls.

## Model Idea

LG-FGT builds a tabular representation through four parts:

1. Feature tokenization for continuous and categorical inputs.
2. Optional sample-wise feature gating for adaptive feature weighting.
3. A local branch for flattened token features and a global branch for
   transformer-based feature interactions.
4. Controlled fusion with optional branch gating, raw-token shortcut, and
   small fusion ensembles for selected settings.

The implementation also exposes diagnostic importance signals from the feature
gates and token-pooling weights. These signals are meant for model inspection,
not as causal explanations.

## Installation

Create an environment with Python 3.10 or newer, then install the repository in
editable mode:

```bash
pip install -e .[extra]
```

Optional benchmark baselines require their own packages:

```bash
pip install xgboost lightgbm catboost
```

## Quick Usage

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FeatureGateTransformerConfig

data_config = DataConfig(
    target=["target"],
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    normalize_continuous_features=True,
)

model_config = FeatureGateTransformerConfig(
    task="classification",
    input_embed_dim=32,
    num_heads=4,
    num_attn_blocks=2,
    use_feature_gating=True,
    use_local_branch=True,
    use_global_branch=True,
)

trainer_config = TrainerConfig(max_epochs=30, batch_size=1024)
optimizer_config = OptimizerConfig()

model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=trainer_config,
    optimizer_config=optimizer_config,
)
model.fit(train=train_df, validation=valid_df)
metrics = model.evaluate(test_df)
predictions = model.predict(test_df)
```

## Benchmark Examples

Run a fast smoke test:

```powershell
python benchmarks/run_benchmark.py --dataset synthetic_classification --models logistic_regression feature_gate_transformer_adaptive --sample-size 1000 --max-epochs 3 --accelerator cpu
```

Run a single real dataset:

```powershell
python benchmarks/run_benchmark.py --dataset adult_income --models logistic_regression random_forest ft_transformer feature_gate_transformer_adaptive --sample-size 50000 --max-epochs 30 --accelerator gpu --seeds 42 43 44
```

Raw experiment outputs are intentionally ignored by Git. Keep large logs,
checkpoints, downloaded datasets, and local result folders outside the public
repository unless you deliberately prepare a release artifact.

## Tests

```bash
pytest tests/test_feature_gate_transformer.py
```

The original `pytorch-tabular` test suite is still present, but this fork's
main added tests are the LG-FGT tests above.

## Project Status

This is a research-engineering project. The current evidence supports a
conservative interpretation:

- LG-FGT is useful as a compact deep tabular baseline with local-global feature
  fusion and inspectable gating signals.
- It can be competitive among common deep tabular baselines on selected
  small-to-medium datasets.
- It should not be advertised as universally stronger than tree ensembles,
  TabPFN-style foundation models, or recent large-scale tabular systems.

## Project Origin

This repository builds on the open-source `pytorch-tabular` project and keeps
the original MIT license. The LG-FGT model and benchmark scaffold were added in
this fork for academic experimentation.

## Citation

This fork does not currently provide a formal academic citation. If you use the
base library, cite the upstream `pytorch-tabular` project as appropriate. If you
use LG-FGT from this repository, cite the repository URL and release version
once a public release is created.
