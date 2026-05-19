#!/usr/bin/env python
"""Tests for the feature gate transformer model."""

import numpy as np
import pandas as pd
from skbase.utils.git_diff import _is_module_changed
import pytest

from pytorch_tabular import TabularModel
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import FeatureGateTransformerConfig


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
@pytest.mark.parametrize("multi_target", [True, False])
@pytest.mark.parametrize(
    "continuous_cols",
    [["AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
def test_regression(regression_data, multi_target, continuous_cols, categorical_cols):
    (train, test, target) = regression_data
    if len(continuous_cols) + len(categorical_cols) == 0:
        return
    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="regression",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        fusion_ensemble_size=2,
        fusion_ensemble_dropout=0.05,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    result = tabular_model.evaluate(test)
    assert "test_mean_squared_error" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("continuous_cols", [[f"feature_{i}" for i in range(54)]])
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
def test_classification(classification_data, multi_target, continuous_cols, categorical_cols):
    (train, test, target) = classification_data
    data_config = DataConfig(
        target=target + ["feature_53"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="classification",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        use_branch_fusion_gate=False,
        fusion_ensemble_size=2,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    result = tabular_model.evaluate(test)
    assert "test_accuracy" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]


def test_embedding_transformer(regression_data):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=["AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        categorical_cols=["HouseAgeBin"],
    )
    model_config = FeatureGateTransformerConfig(
        task="regression",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        use_feature_gating=False,
        use_global_branch=False,
        use_raw_shortcut=False,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    transformer = CategoricalEmbeddingTransformer(tabular_model)
    train_transform = transformer.fit_transform(train)
    embed_cols = [col for col in train_transform.columns if "HouseAgeBin_embed_dim" in col]
    assert len(train["HouseAgeBin"].unique()) + 1 == len(transformer._mapping["HouseAgeBin"].keys())
    assert all(val.shape[0] == len(embed_cols) for val in transformer._mapping["HouseAgeBin"].values())


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
def test_auto_disable_gating_on_wide_inputs():
    n_rows = 256
    train = {
        **{f"feature_{i}": np.random.randn(n_rows) for i in range(24)},
        "feature_0_cat": np.random.choice(["a", "b", "c"], size=n_rows),
        "target": np.random.choice([0, 1], size=n_rows),
    }
    train = pd.DataFrame(train)
    data_config = DataConfig(
        target=["target"],
        continuous_cols=[f"feature_{i}" for i in range(24)],
        categorical_cols=["feature_0_cat"],
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="classification",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        auto_disable_gating_on_wide_inputs=True,
        gating_max_tokens=16,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    backbone = tabular_model.model.backbone
    assert backbone._use_feature_gating is False
    assert backbone.feature_gating_active_ is False
    np.testing.assert_allclose(backbone.feature_importance_, np.ones(25))
    assert backbone.token_pool_feature_importance_ is not None
    assert len(backbone.token_pool_feature_importance_) == 25


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
def test_auto_disable_gating_on_multiclass_inputs():
    n_rows = 180
    train = pd.DataFrame(
        {
            "num_0": np.random.randn(n_rows),
            "num_1": np.random.randn(n_rows),
            "cat_0": np.random.choice(["a", "b", "c"], size=n_rows),
            "target": np.tile([0, 1, 2], n_rows // 3),
        }
    )
    data_config = DataConfig(
        target=["target"],
        continuous_cols=["num_0", "num_1"],
        categorical_cols=["cat_0"],
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="classification",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        auto_disable_gating_on_multiclass_inputs=True,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    backbone = tabular_model.model.backbone
    assert backbone._use_feature_gating is False
    assert backbone.feature_gating_active_ is False
    np.testing.assert_allclose(backbone.feature_importance_, np.ones(3))
    assert backbone.token_pool_feature_importance_ is not None
    assert len(backbone.token_pool_feature_importance_) == 3


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
def test_auto_regression_ensemble_on_categorical_inputs():
    n_rows = 128
    train = pd.DataFrame(
        {
            "num": np.random.randn(n_rows),
            "cat_0": np.random.choice(["a", "b", "c"], size=n_rows),
            "cat_1": np.random.choice(["d", "e", "f"], size=n_rows),
            "cat_2": np.random.choice(["g", "h", "i"], size=n_rows),
            "cat_3": np.random.choice(["j", "k", "l"], size=n_rows),
            "target": np.random.randn(n_rows),
        }
    )
    data_config = DataConfig(
        target=["target"],
        continuous_cols=["num"],
        categorical_cols=["cat_0", "cat_1", "cat_2", "cat_3"],
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="regression",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        fusion_ensemble_size=1,
        auto_enable_categorical_regression_ensemble=True,
        categorical_regression_ensemble_min_tokens=4,
        categorical_regression_ensemble_size=2,
        categorical_regression_ensemble_dropout=0.03,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    backbone = tabular_model.model.backbone
    assert backbone._fusion_ensemble_size == 2
    assert backbone.fusion_input_scales.shape[0] == 2


@pytest.mark.skipif(
    not _is_module_changed("pytorch_tabular.models.feature_gate_transformer"),
    reason="run test only if feature_gate_transformer module is changed",
)
def test_auto_simplify_categorical_regression():
    n_rows = 128
    train = pd.DataFrame(
        {
            "num": np.random.randn(n_rows),
            "cat_0": np.random.choice(["a", "b", "c"], size=n_rows),
            "cat_1": np.random.choice(["d", "e", "f"], size=n_rows),
            "cat_2": np.random.choice(["g", "h", "i"], size=n_rows),
            "cat_3": np.random.choice(["j", "k", "l"], size=n_rows),
            "target": np.random.randn(n_rows),
        }
    )
    data_config = DataConfig(
        target=["target"],
        continuous_cols=["num"],
        categorical_cols=["cat_0", "cat_1", "cat_2", "cat_3"],
        normalize_continuous_features=True,
    )
    model_config = FeatureGateTransformerConfig(
        task="regression",
        input_embed_dim=8,
        num_attn_blocks=1,
        num_heads=2,
        feature_gate_hidden_dim=16,
        mlp_layers="32-16",
        fusion_layers="16",
        use_feature_gating=True,
        use_branch_fusion_gate=True,
        fusion_ensemble_size=2,
        fusion_ensemble_dropout=0.03,
        auto_simplify_categorical_regression=True,
        categorical_regression_simplify_min_tokens=4,
    )
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    backbone = tabular_model.model.backbone
    assert backbone._use_feature_gating is False
    assert backbone.feature_gating_active_ is False
    assert backbone.branch_gate is None
    assert backbone._fusion_ensemble_size == 1
    assert backbone.fusion_input_scales is None
