import argparse
import json
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import urlopen
from zipfile import ZipFile

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

warnings.filterwarnings(
    "ignore",
    message=r"Starting from v1\.9\.0, `tensorboardX` has been removed as a dependency.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The '.*_dataloader' does not have many workers which may be a bottleneck.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The number of training batches \(\d+\) is smaller than the logging interval.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Checkpoint directory .* exists and is not empty\.",
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM.* was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message=r"DataFrame is highly fragmented.*",
)

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing, fetch_openml, load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import (
    AutoIntConfig,
    CategoryEmbeddingModelConfig,
    DANetConfig,
    FeatureGateTransformerConfig,
    FTTransformerConfig,
    GANDALFConfig,
    NodeConfig,
    TabNetModelConfig,
    TabTransformerConfig,
)
from pytorch_tabular.utils import load_covertype_dataset, make_mixed_dataset

SUPPORTED_DATASETS = [
    "covertype",
    "adult_income",
    "bank_marketing",
    "california_housing",
    "bike_sharing",
    "abalone",
    "wine_quality_red",
    "airfoil_self_noise",
    "energy_efficiency",
    "qsar_fish_toxicity",
    "breast_cancer",
    "diabetes",
    "electricity",
    "helena",
    "higgs",
    "jannis",
    "phoneme",
    "wine_classification",
    "synthetic_classification",
    "synthetic_regression",
]


@dataclass
class DatasetBundle:
    name: str
    task: str
    frame: pd.DataFrame
    target: str
    categorical_cols: List[str]
    continuous_cols: List[str]
    description: str


def build_feature_gate_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    common_kwargs = dict(
        task=task,
        num_heads=4,
        feature_gate_hidden_dim=32,
        branch_gate_hidden_dim=32,
        num_attn_blocks=2 if task == "classification" else 1,
        feature_gate_dropout=0.05 if task == "classification" else 0.04,
        residual_gate_scale=0.35 if task == "classification" else 0.25,
        mlp_layers="128-64",
        fusion_layers="64",
        dropout=0.05 if task == "classification" else 0.04,
        use_branch_fusion_gate=(task == "classification"),
        use_feature_gating=True,
        use_raw_shortcut=True,
        fusion_ensemble_size=4 if task == "classification" else 1,
        fusion_ensemble_dropout=0.02 if task == "classification" else 0.0,
        learning_rate=1e-3 if task == "classification" else 8e-4,
        seed=seed,
    )
    return FeatureGateTransformerConfig(**common_kwargs)


def build_feature_gate_ablation_config(task: str, seed: int, **overrides) -> FeatureGateTransformerConfig:
    config = build_feature_gate_config(task, seed)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def build_feature_gate_tuned_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_config(task, seed)
    if task == "classification":
        config.use_branch_fusion_gate = False
    else:
        config.use_feature_gating = False
        config.fusion_ensemble_size = 2
        config.fusion_ensemble_dropout = 0.03
    return config


def build_feature_gate_auto_classification_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_config(task, seed)
    if task == "classification":
        config.use_branch_fusion_gate = False
        config.auto_disable_gating_on_multiclass_inputs = True
    return config


def build_feature_gate_cls_interaction_focus_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_adaptive_config(task, seed)
    if task == "classification":
        config.use_local_branch = False
        config.use_branch_fusion_gate = False
    return config


def build_feature_gate_cls_conservative_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_adaptive_config(task, seed)
    if task == "classification":
        config.use_feature_gating = False
        config.use_branch_fusion_gate = False
        config.fusion_ensemble_size = 1
        config.fusion_ensemble_dropout = 0.0
    return config


def build_feature_gate_adaptive_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_config(task, seed)
    if task == "classification":
        config.use_branch_fusion_gate = False
        config.auto_disable_gating_on_multiclass_inputs = True
    else:
        config.use_feature_gating = True
        config.use_branch_fusion_gate = True
        config.fusion_ensemble_size = 2
        config.fusion_ensemble_dropout = 0.03
        config.auto_enable_categorical_regression_ensemble = False
        config.auto_simplify_categorical_regression = True
        config.categorical_regression_simplify_min_tokens = 4
    return config


def build_feature_gate_reg_numeric_wide_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_adaptive_config(task, seed)
    if task == "regression":
        config.feature_gate_dropout = 0.02
        config.dropout = 0.02
        config.mlp_layers = "256-128-64"
        config.fusion_layers = "128-64"
        config.use_batch_norm = True
        config.learning_rate = 1e-3
        config.fusion_ensemble_size = 2
        config.fusion_ensemble_dropout = 0.02
        config.auto_enable_categorical_regression_ensemble = False
        config.auto_simplify_categorical_regression = False
    return config


def build_feature_gate_reg_numeric_wide_no_raw_config(task: str, seed: int) -> FeatureGateTransformerConfig:
    config = build_feature_gate_reg_numeric_wide_config(task, seed)
    if task == "regression":
        config.use_raw_shortcut = False
    return config


DL_MODEL_FACTORIES: Dict[str, Callable[[str, int], object]] = {
    "category_embedding": lambda task, seed: CategoryEmbeddingModelConfig(
        task=task,
        layers="256-128-64",
        seed=seed,
    ),
    "ft_transformer": lambda task, seed: FTTransformerConfig(
        task=task,
        num_attn_blocks=4,
        num_heads=4,
        seed=seed,
    ),
    "feature_gate_transformer": build_feature_gate_config,
    "feature_gate_transformer_tuned": build_feature_gate_tuned_config,
    "feature_gate_transformer_cls_auto": build_feature_gate_auto_classification_config,
    "feature_gate_transformer_adaptive": build_feature_gate_adaptive_config,
    "feature_gate_transformer_cls_interaction_focus": build_feature_gate_cls_interaction_focus_config,
    "feature_gate_transformer_cls_conservative": build_feature_gate_cls_conservative_config,
    "feature_gate_transformer_reg_numeric_wide": build_feature_gate_reg_numeric_wide_config,
    "feature_gate_transformer_reg_numeric_wide_no_raw": build_feature_gate_reg_numeric_wide_no_raw_config,
    "feature_gate_transformer_no_gating": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_feature_gating=False,
    ),
    "feature_gate_transformer_no_branch_gate": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_branch_fusion_gate=False,
    ),
    "feature_gate_transformer_no_raw_shortcut": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_raw_shortcut=False,
    ),
    "feature_gate_transformer_local_only": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_global_branch=False,
        use_raw_shortcut=False,
    ),
    "feature_gate_transformer_global_only": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_local_branch=False,
    ),
    "feature_gate_transformer_no_ensemble": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        fusion_ensemble_size=1,
        fusion_ensemble_dropout=0.0,
    ),
    "feature_gate_transformer_cls_no_gating_no_branch": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_feature_gating=False,
        use_branch_fusion_gate=False,
    ),
    "feature_gate_transformer_cls_no_gating_no_branch_no_raw": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_feature_gating=False,
        use_branch_fusion_gate=False,
        use_raw_shortcut=False,
    ),
    "feature_gate_transformer_reg_branch_gate": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_branch_fusion_gate=True,
    ),
    "feature_gate_transformer_reg_ensemble2": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        fusion_ensemble_size=2,
        fusion_ensemble_dropout=0.03,
    ),
    "feature_gate_transformer_reg_soft_gate": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        residual_gate_scale=0.1,
        gate_temperature=1.5,
        feature_gate_dropout=0.0,
        fusion_ensemble_size=2,
        fusion_ensemble_dropout=0.03,
    ),
    "feature_gate_transformer_reg_legacy": lambda task, seed: build_feature_gate_ablation_config(
        task,
        seed,
        use_branch_fusion_gate=True,
        fusion_ensemble_size=2,
        fusion_ensemble_dropout=0.03,
    ),
    "tab_transformer": lambda task, seed: TabTransformerConfig(
        task=task,
        num_attn_blocks=4,
        num_heads=4,
        seed=seed,
    ),
    "gandalf": lambda task, seed: GANDALFConfig(
        task=task,
        gflu_stages=6,
        seed=seed,
    ),
    "tabnet": lambda task, seed: TabNetModelConfig(
        task=task,
        n_d=16,
        n_a=16,
        n_steps=3,
        seed=seed,
    ),
    "autoint": lambda task, seed: AutoIntConfig(
        task=task,
        attn_embed_dim=32,
        num_heads=4,
        num_attn_blocks=3,
        deep_layers=True,
        layers="128-64",
        seed=seed,
    ),
    "node": lambda task, seed: NodeConfig(
        task=task,
        num_layers=1,
        num_trees=512,
        depth=6,
        seed=seed,
    ),
    "danet": lambda task, seed: DANetConfig(
        task=task,
        n_layers=8,
        abstlay_dim_1=32,
        seed=seed,
    ),
}

DEFAULT_MODELS = {
    "classification": [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "category_embedding",
        "ft_transformer",
        "feature_gate_transformer",
        "tab_transformer",
        "gandalf",
    ],
    "regression": [
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "category_embedding",
        "ft_transformer",
        "feature_gate_transformer",
        "tab_transformer",
        "gandalf",
    ],
}

OPTIONAL_BASELINE_MODELS = {
    "classification": ["xgboost", "lightgbm", "catboost"],
    "regression": ["xgboost", "lightgbm", "catboost"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight benchmark on tabular datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run. If omitted, uses task-specific defaults.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row subsample before splitting.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction from the full dataset.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Optional fraction of the training split to keep for low-data sensitivity runs.",
    )
    parser.add_argument(
        "--train-subsample-seed",
        type=int,
        default=42,
        help="Random seed used when --train-fraction is less than 1.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Model seeds to run.")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed data split seed.")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs for deep models.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for deep models.")
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=400,
        help=(
            "Number of CatBoost iterations. Keep the default for formal benchmarks; "
            "lower it for lightweight probe runs on high-cardinality multiclass datasets."
        ),
    )
    parser.add_argument(
        "--deep-early-stopping",
        action="store_true",
        help="Enable valid_loss early stopping and load-best checkpointing for deep models.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Patience for --deep-early-stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0005,
        help="Minimum valid_loss improvement for --deep-early-stopping.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["none", "standard", "yeo-johnson"],
        default="none",
        help="Optional target transform for deep regression models.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Trainer accelerator for deep models.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Trainer devices for deep models.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing raw_results.csv by skipping already successful model/seed runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to benchmark_outputs/<dataset>_<timestamp>.",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset cache directory.")
    return parser.parse_args()


def ensure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("benchmark_outputs") / f"{args.dataset}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    return out_dir


def download_with_cache(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    with urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def read_csv_cached(url: str, destination: Path, **kwargs) -> pd.DataFrame:
    local_path = download_with_cache(url, destination)
    return pd.read_csv(local_path, **kwargs)


def read_csv_from_zip_cached(url: str, destination: Path, member_name: str, **kwargs) -> pd.DataFrame:
    local_path = download_with_cache(url, destination)
    with ZipFile(local_path) as archive:
        with archive.open(member_name) as handle:
            return pd.read_csv(handle, **kwargs)


def read_excel_cached(url: str, destination: Path, **kwargs) -> pd.DataFrame:
    local_path = download_with_cache(url, destination)
    return pd.read_excel(local_path, **kwargs)


def load_adult_income_dataset(data_dir: Path) -> DatasetBundle:
    adult_dir = data_dir / "adult_income"
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    train = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        adult_dir / "adult.data",
        names=columns,
        sep=r",\s*",
        engine="python",
        na_values=["?"],
    )
    test = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        adult_dir / "adult.test",
        names=columns,
        sep=r",\s*",
        engine="python",
        skiprows=1,
        na_values=["?"],
    )
    frame = pd.concat([train, test], ignore_index=True)
    frame["income"] = frame["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    categorical_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    continuous_cols = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    return DatasetBundle(
        name="adult_income",
        task="classification",
        frame=frame,
        target="income",
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="UCI Adult income classification benchmark with mixed categorical and continuous inputs.",
    )


def load_bank_marketing_dataset(data_dir: Path) -> DatasetBundle:
    frame = read_csv_from_zip_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
        data_dir / "bank_marketing" / "bank-additional.zip",
        "bank-additional/bank-additional-full.csv",
        sep=";",
    )
    categorical_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    continuous_cols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    return DatasetBundle(
        name="bank_marketing",
        task="classification",
        frame=frame,
        target="y",
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="UCI Bank Marketing benchmark with campaign metadata and socioeconomic signals.",
    )


def load_bike_sharing_dataset(data_dir: Path) -> DatasetBundle:
    frame = read_csv_from_zip_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        data_dir / "bike_sharing" / "bike-sharing.zip",
        "hour.csv",
    )
    frame = frame.drop(columns=["instant", "dteday", "casual", "registered"])
    categorical_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    continuous_cols = ["temp", "atemp", "hum", "windspeed"]
    frame = frame.astype({col: str for col in categorical_cols}, copy=False)
    return DatasetBundle(
        name="bike_sharing",
        task="regression",
        frame=frame,
        target="cnt",
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="UCI Bike Sharing hourly demand regression benchmark with mixed tabular covariates.",
    )


def load_abalone_dataset(data_dir: Path) -> DatasetBundle:
    columns = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]
    frame = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        data_dir / "abalone" / "abalone.data",
        names=columns,
    )
    categorical_cols = ["Sex"]
    continuous_cols = [col for col in columns if col not in categorical_cols + ["Rings"]]
    return DatasetBundle(
        name="abalone",
        task="regression",
        frame=frame,
        target="Rings",
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="UCI Abalone age estimation regression benchmark with one categorical and seven numeric features.",
    )


def load_wine_quality_red_dataset(data_dir: Path) -> DatasetBundle:
    frame = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        data_dir / "wine_quality_red" / "winequality-red.csv",
        sep=";",
    )


def load_airfoil_self_noise_dataset(data_dir: Path) -> DatasetBundle:
    columns = [
        "frequency",
        "angle_of_attack",
        "chord_length",
        "free_stream_velocity",
        "suction_side_displacement_thickness",
        "scaled_sound_pressure_level",
    ]
    frame = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        data_dir / "airfoil_self_noise" / "airfoil_self_noise.dat",
        sep=r"\s+",
        names=columns,
        engine="python",
    )
    target = "scaled_sound_pressure_level"
    continuous_cols = [col for col in columns if col != target]
    return DatasetBundle(
        name="airfoil_self_noise",
        task="regression",
        frame=frame,
        target=target,
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="UCI Airfoil Self-Noise regression benchmark with continuous aerodynamic features.",
    )


def load_energy_efficiency_dataset(data_dir: Path) -> DatasetBundle:
    dataset = fetch_openml_as_frame(1472, data_dir)
    frame = dataset.frame.copy()
    frame = frame.rename(
        columns={
            "V1": "relative_compactness",
            "V2": "surface_area",
            "V3": "wall_area",
            "V4": "roof_area",
            "V5": "overall_height",
            "V6": "orientation",
            "V7": "glazing_area",
            "V8": "glazing_area_distribution",
            "y1": "heating_load",
            "y2": "cooling_load",
        }
    )
    frame = frame.drop(columns=["cooling_load"])
    frame["heating_load"] = pd.to_numeric(frame["heating_load"])
    categorical_cols = ["orientation", "glazing_area_distribution"]
    for col in categorical_cols:
        frame[col] = frame[col].astype(str)
    target = "heating_load"
    continuous_cols = [col for col in frame.columns if col not in categorical_cols + [target]]
    return DatasetBundle(
        name="energy_efficiency",
        task="regression",
        frame=frame,
        target=target,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="UCI Energy Efficiency heating-load regression benchmark with compact building-design features.",
    )


def load_qsar_fish_toxicity_dataset(data_dir: Path) -> DatasetBundle:
    columns = ["CIC0", "SM1_DzZ", "GATS1i", "NdsCH", "NdssC", "MLOGP", "LC50"]
    frame = read_csv_cached(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv",
        data_dir / "qsar_fish_toxicity" / "qsar_fish_toxicity.csv",
        sep=";",
        names=columns,
    )
    target = "LC50"
    continuous_cols = [col for col in columns if col != target]
    return DatasetBundle(
        name="qsar_fish_toxicity",
        task="regression",
        frame=frame,
        target=target,
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="UCI QSAR Fish Toxicity regression benchmark with molecular descriptors.",
    )
    target = "quality"
    continuous_cols = [col for col in frame.columns if col != target]
    return DatasetBundle(
        name="wine_quality_red",
        task="regression",
        frame=frame,
        target=target,
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="UCI Red Wine Quality regression benchmark with physicochemical continuous features.",
    )


def load_breast_cancer_dataset() -> DatasetBundle:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame["target"] = dataset.target.astype(int)
    continuous_cols = [c for c in frame.columns if c != "target"]
    return DatasetBundle(
        name="breast_cancer",
        task="classification",
        frame=frame,
        target="target",
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="Scikit-learn/UCI breast cancer diagnostic classification benchmark with continuous features.",
    )


def load_diabetes_dataset() -> DatasetBundle:
    dataset = load_diabetes(as_frame=True)
    frame = dataset.frame.copy()
    frame["target"] = dataset.target.astype(float)
    continuous_cols = [c for c in frame.columns if c != "target"]
    return DatasetBundle(
        name="diabetes",
        task="regression",
        frame=frame,
        target="target",
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="Scikit-learn diabetes disease-progression regression benchmark with continuous features.",
    )


def load_wine_classification_dataset() -> DatasetBundle:
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.copy()
    frame["target"] = dataset.target.astype(int)
    continuous_cols = [c for c in frame.columns if c != "target"]
    return DatasetBundle(
        name="wine_classification",
        task="classification",
        frame=frame,
        target="target",
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="Scikit-learn/UCI wine cultivar multiclass classification benchmark with continuous features.",
    )


def fetch_openml_as_frame(data_id: int, data_dir: Path):
    openml_dir = data_dir / "openml"
    try:
        return fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=str(openml_dir))
    except TypeError:
        return fetch_openml(data_id=data_id, as_frame=True, data_home=str(openml_dir))


def load_electricity_dataset(data_dir: Path) -> DatasetBundle:
    dataset = fetch_openml_as_frame(151, data_dir)
    frame = dataset.frame.copy()
    target = "class"
    categorical_cols = ["day"]
    frame[target] = frame[target].astype(str)
    for col in categorical_cols:
        frame[col] = frame[col].astype(str)
    continuous_cols = [c for c in frame.columns if c not in categorical_cols + [target]]
    return DatasetBundle(
        name="electricity",
        task="classification",
        frame=frame,
        target=target,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description="OpenML electricity binary classification benchmark with mixed temporal and demand features.",
    )


def load_phoneme_dataset(data_dir: Path) -> DatasetBundle:
    dataset = fetch_openml_as_frame(1489, data_dir)
    frame = dataset.frame.copy()
    target = "Class"
    frame[target] = frame[target].astype(str)
    continuous_cols = [c for c in frame.columns if c != target]
    return DatasetBundle(
        name="phoneme",
        task="classification",
        frame=frame,
        target=target,
        categorical_cols=[],
        continuous_cols=continuous_cols,
        description="OpenML phoneme binary classification benchmark with continuous acoustic features.",
    )


def load_openml_classification_dataset(data_dir: Path, data_id: int, name: str, description: str) -> DatasetBundle:
    dataset = fetch_openml_as_frame(data_id, data_dir)
    frame = dataset.frame.copy()
    target = getattr(dataset.target, "name", None) or dataset.details.get("default_target_attribute") or frame.columns[-1]
    target = str(target)
    frame[target] = frame[target].astype(str)

    categorical_cols = [
        c
        for c in frame.columns
        if c != target and (str(frame[c].dtype) == "category" or frame[c].dtype == object)
    ]
    for col in categorical_cols:
        frame[col] = frame[col].astype(str)
    continuous_cols = [c for c in frame.columns if c not in categorical_cols + [target]]
    return DatasetBundle(
        name=name,
        task="classification",
        frame=frame,
        target=target,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        description=description,
    )


def load_dataset(dataset_name: str, data_dir: Path, sample_size: Optional[int], split_seed: int) -> DatasetBundle:
    if dataset_name == "covertype":
        frame, categorical_cols, continuous_cols, target = load_covertype_dataset(download_dir=str(data_dir))
        bundle = DatasetBundle(
            name=dataset_name,
            task="classification",
            frame=frame.copy(),
            target=target,
            categorical_cols=list(categorical_cols),
            continuous_cols=list(continuous_cols),
            description="UCI Covertype with Wilderness_Area and Soil_Type converted to categorical columns.",
        )
    elif dataset_name == "adult_income":
        bundle = load_adult_income_dataset(data_dir)
    elif dataset_name == "bank_marketing":
        bundle = load_bank_marketing_dataset(data_dir)
    elif dataset_name == "california_housing":
        dataset = fetch_california_housing(data_home=str(data_dir), as_frame=True)
        frame = dataset.frame.copy()
        frame["HouseAgeBin"] = pd.qcut(frame["HouseAge"], q=4, duplicates="drop")
        frame["HouseAgeBin"] = "age_" + frame["HouseAgeBin"].cat.codes.astype(str)
        frame["AveRoomsBin"] = pd.qcut(frame["AveRooms"], q=4, duplicates="drop")
        frame["AveRoomsBin"] = "rooms_" + frame["AveRoomsBin"].cat.codes.astype(str)
        target = dataset.target_names[0]
        categorical_cols = ["HouseAgeBin", "AveRoomsBin"]
        continuous_cols = [c for c in frame.columns if c not in categorical_cols + [target]]
        bundle = DatasetBundle(
            name=dataset_name,
            task="regression",
            frame=frame,
            target=target,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            description="California Housing with two quantile-binned categorical features added.",
        )
    elif dataset_name == "bike_sharing":
        bundle = load_bike_sharing_dataset(data_dir)
    elif dataset_name == "abalone":
        bundle = load_abalone_dataset(data_dir)
    elif dataset_name == "wine_quality_red":
        bundle = load_wine_quality_red_dataset(data_dir)
    elif dataset_name == "airfoil_self_noise":
        bundle = load_airfoil_self_noise_dataset(data_dir)
    elif dataset_name == "energy_efficiency":
        bundle = load_energy_efficiency_dataset(data_dir)
    elif dataset_name == "qsar_fish_toxicity":
        bundle = load_qsar_fish_toxicity_dataset(data_dir)
    elif dataset_name == "breast_cancer":
        bundle = load_breast_cancer_dataset()
    elif dataset_name == "diabetes":
        bundle = load_diabetes_dataset()
    elif dataset_name == "electricity":
        bundle = load_electricity_dataset(data_dir)
    elif dataset_name == "helena":
        bundle = load_openml_classification_dataset(
            data_dir,
            data_id=41169,
            name="helena",
            description="OpenML Helena multiclass classification benchmark used in RTDL-style tabular evaluations.",
        )
    elif dataset_name == "higgs":
        bundle = load_openml_classification_dataset(
            data_dir,
            data_id=4532,
            name="higgs",
            description="OpenML Higgs binary classification benchmark used in large tabular evaluations.",
        )
    elif dataset_name == "jannis":
        bundle = load_openml_classification_dataset(
            data_dir,
            data_id=41168,
            name="jannis",
            description="OpenML Jannis multiclass classification benchmark used in RTDL-style tabular evaluations.",
        )
    elif dataset_name == "phoneme":
        bundle = load_phoneme_dataset(data_dir)
    elif dataset_name == "wine_classification":
        bundle = load_wine_classification_dataset()
    elif dataset_name == "synthetic_classification":
        frame, categorical_cols, continuous_cols = make_mixed_dataset(
            task="classification",
            n_samples=max(sample_size or 5000, 1000),
            n_features=8,
            n_categories=3,
            n_informative=5,
            random_state=split_seed,
        )
        bundle = DatasetBundle(
            name=dataset_name,
            task="classification",
            frame=frame,
            target="target",
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            description="Synthetic mixed-type classification dataset for debugging.",
        )
    elif dataset_name == "synthetic_regression":
        frame, categorical_cols, continuous_cols = make_mixed_dataset(
            task="regression",
            n_samples=max(sample_size or 5000, 1000),
            n_features=8,
            n_categories=3,
            n_informative=5,
            random_state=split_seed,
        )
        bundle = DatasetBundle(
            name=dataset_name,
            task="regression",
            frame=frame,
            target="target",
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            description="Synthetic mixed-type regression dataset for debugging.",
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if sample_size is not None and sample_size < len(bundle.frame):
        sampled = bundle.frame.sample(sample_size, random_state=split_seed).reset_index(drop=True)
        bundle.frame = sampled
    else:
        bundle.frame = bundle.frame.reset_index(drop=True)
    return bundle


def subsample_train_split(
    train: pd.DataFrame,
    target: str,
    task: str,
    train_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 < train_fraction <= 1:
        raise ValueError(f"--train-fraction must be in (0, 1], got {train_fraction}.")
    if train_fraction >= 1:
        return train.reset_index(drop=True)

    if task == "classification":
        sampled, _discarded = train_test_split(
            train,
            train_size=train_fraction,
            random_state=seed,
            stratify=train[target],
        )
    else:
        sampled = train.sample(frac=train_fraction, random_state=seed)
    return sampled.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame,
    target: str,
    task: str,
    test_size: float,
    val_size: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = frame[target] if task == "classification" else None
    train_valid, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=split_seed,
        stratify=stratify,
    )
    valid_relative = val_size / (1.0 - test_size)
    stratify_train = train_valid[target] if task == "classification" else None
    train, valid = train_test_split(
        train_valid,
        test_size=valid_relative,
        random_state=split_seed,
        stratify=stratify_train,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dataset_profile(bundle: DatasetBundle, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    profile = {
        "dataset": bundle.name,
        "task": bundle.task,
        "description": bundle.description,
        "rows_total": int(len(bundle.frame)),
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "rows_test": int(len(test)),
        "target": bundle.target,
        "categorical_cols": bundle.categorical_cols,
        "continuous_cols": bundle.continuous_cols,
        "n_categorical": len(bundle.categorical_cols),
        "n_continuous": len(bundle.continuous_cols),
    }
    if bundle.task == "classification":
        profile["class_counts"] = train[bundle.target].value_counts().to_dict()
        profile["n_classes"] = int(train[bundle.target].nunique())
    return profile


def build_tabular_model(
    model_name: str,
    task: str,
    seed: int,
    categorical_cols: List[str],
    continuous_cols: List[str],
    target: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> TabularModel:
    use_early_stopping = bool(getattr(args, "deep_early_stopping", False))
    data_config = DataConfig(
        target=[target],
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
    )
    trainer_config = TrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        accelerator=args.accelerator,
        devices=args.devices,
        progress_bar="none",
        checkpoints="valid_loss" if use_early_stopping else None,
        checkpoints_path=str(run_dir / "checkpoints"),
        checkpoints_name=f"{model_name}_seed{seed}",
        checkpoints_mode="min",
        early_stopping="valid_loss" if use_early_stopping else None,
        early_stopping_mode="min",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        load_best=use_early_stopping,
        seed=seed,
        trainer_kwargs={"enable_model_summary": False, "num_sanity_val_steps": 0, "log_every_n_steps": 1},
    )
    optimizer_config = OptimizerConfig()
    model_config = DL_MODEL_FACTORIES[model_name](task, seed)
    return TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        suppress_lightning_logger=True,
        verbose=False,
    )


def build_target_transform(task: str, args: argparse.Namespace):
    transform_name = getattr(args, "target_transform", "none")
    if task != "regression" or transform_name == "none":
        return None
    if transform_name == "standard":
        return StandardScaler()
    if transform_name == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unsupported target transform: {transform_name}")


def extract_predictions(pred_df: pd.DataFrame, target: str, task: str) -> np.ndarray:
    pred_col = f"{target}_prediction"
    if pred_col not in pred_df.columns:
        raise KeyError(f"Expected prediction column `{pred_col}` not found. Columns: {pred_df.columns.tolist()}")
    pred = pred_df[pred_col].to_numpy()
    if task == "regression":
        pred = pred.astype(float)
    return pred


def score_predictions(task: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_deep_model(
    model_name: str,
    bundle: DatasetBundle,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict:
    model = build_tabular_model(
        model_name=model_name,
        task=bundle.task,
        seed=seed,
        categorical_cols=bundle.categorical_cols,
        continuous_cols=bundle.continuous_cols,
        target=bundle.target,
        run_dir=run_dir,
        args=args,
    )
    start = time.perf_counter()
    # TabularModel.fit defaults to seed=42 unless we pass it explicitly, so forward the loop seed here.
    target_transform = build_target_transform(bundle.task, args)
    model.fit(train=train, validation=valid, seed=seed, target_transform=target_transform)
    pred_df = model.predict(test, include_input_features=False)
    elapsed = time.perf_counter() - start
    y_pred = extract_predictions(pred_df, bundle.target, bundle.task)
    external_metrics = score_predictions(bundle.task, test[bundle.target], y_pred)
    eval_metrics = model.evaluate(test=test, verbose=False)[0]
    payload = {
        "model_family": "pytorch_tabular",
        "train_seconds": round(elapsed, 4),
        "n_params": int(model.num_params or 0),
        "target_transform": getattr(args, "target_transform", "none"),
    }
    payload.update({f"library_{k}": v for k, v in eval_metrics.items()})
    payload.update(external_metrics)
    return payload


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def import_optional_dependency(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Optional baseline `{module_name}` is not installed in the current environment. "
            f"Install it first or remove it from `--models`."
        ) from exc


def encode_for_sklearn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: List[str],
    continuous_cols: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train_num = train[continuous_cols].copy() if continuous_cols else pd.DataFrame(index=train.index)
    x_test_num = test[continuous_cols].copy() if continuous_cols else pd.DataFrame(index=test.index)
    if continuous_cols:
        fill_values = x_train_num.median(numeric_only=True)
        x_train_num = x_train_num.fillna(fill_values)
        x_test_num = x_test_num.fillna(fill_values)
        scaler = StandardScaler()
        train_num = scaler.fit_transform(x_train_num)
        test_num = scaler.transform(x_test_num)
    else:
        train_num = np.empty((len(train), 0), dtype=np.float32)
        test_num = np.empty((len(test), 0), dtype=np.float32)

    if categorical_cols:
        x_train_cat = train[categorical_cols].fillna("__missing__").astype(str)
        x_test_cat = test[categorical_cols].fillna("__missing__").astype(str)
        encoder = make_one_hot_encoder()
        train_cat = encoder.fit_transform(x_train_cat)
        test_cat = encoder.transform(x_test_cat)
    else:
        train_cat = np.empty((len(train), 0), dtype=np.float32)
        test_cat = np.empty((len(test), 0), dtype=np.float32)

    return (
        np.hstack([train_num, train_cat]).astype(np.float32),
        np.hstack([test_num, test_cat]).astype(np.float32),
    )


def build_sklearn_estimator(model_name: str, task: str, seed: int, catboost_iterations: int = 400):
    if task == "classification":
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, n_jobs=None)
        if model_name == "random_forest":
            return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)
        if model_name == "extra_trees":
            return ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=1)
        if model_name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(random_state=seed)
        if model_name == "xgboost":
            xgboost = import_optional_dependency("xgboost")
            return xgboost.XGBClassifier(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            )
        if model_name == "lightgbm":
            lightgbm = import_optional_dependency("lightgbm")
            return lightgbm.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
        if model_name == "catboost":
            catboost = import_optional_dependency("catboost")
            return catboost.CatBoostClassifier(
                iterations=catboost_iterations,
                learning_rate=0.05,
                depth=8,
                random_seed=seed,
                verbose=False,
                thread_count=1,
            )
    else:
        if model_name == "ridge":
            return Ridge()
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
        if model_name == "extra_trees":
            return ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=1)
        if model_name == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(random_state=seed)
        if model_name == "xgboost":
            xgboost = import_optional_dependency("xgboost")
            return xgboost.XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=1,
            )
        if model_name == "lightgbm":
            lightgbm = import_optional_dependency("lightgbm")
            return lightgbm.LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
        if model_name == "catboost":
            catboost = import_optional_dependency("catboost")
            return catboost.CatBoostRegressor(
                iterations=catboost_iterations,
                learning_rate=0.05,
                depth=8,
                random_seed=seed,
                verbose=False,
                thread_count=1,
            )
    raise ValueError(f"Unsupported sklearn model `{model_name}` for task `{task}`.")


def run_sklearn_model(
    model_name: str,
    bundle: DatasetBundle,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    catboost_iterations: int = 400,
) -> dict:
    x_train, x_test = encode_for_sklearn(train, test, bundle.categorical_cols, bundle.continuous_cols)
    y_train = train[bundle.target]
    y_test = test[bundle.target]
    if bundle.task == "classification" and model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
    estimator = build_sklearn_estimator(
        model_name,
        bundle.task,
        seed,
        catboost_iterations=catboost_iterations,
    )
    start = time.perf_counter()
    # Keep sklearn baselines single-threaded so Windows benchmark runs do not
    # fail when joblib/threadpool backends cannot create worker pipes.
    with threadpool_limits(limits=1):
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
    elapsed = time.perf_counter() - start
    payload = {
        "model_family": "sklearn",
        "train_seconds": round(elapsed, 4),
        "n_params": None,
    }
    payload.update(score_predictions(bundle.task, y_test, y_pred))
    return payload


def normalise_models(task: str, requested_models: Optional[List[str]]) -> List[str]:
    models = requested_models or DEFAULT_MODELS[task]
    valid_models = (
        set(DEFAULT_MODELS["classification"])
        | set(DEFAULT_MODELS["regression"])
        | set(OPTIONAL_BASELINE_MODELS["classification"])
        | set(OPTIONAL_BASELINE_MODELS["regression"])
        | set(DL_MODEL_FACTORIES)
    )
    unknown = [model for model in models if model not in valid_models]
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}")
    task_models = []
    for model in models:
        if task == "classification" and model == "ridge":
            raise ValueError("`ridge` is regression-only.")
        if task == "regression" and model == "logistic_regression":
            raise ValueError("`logistic_regression` is classification-only.")
        task_models.append(model)
    return task_models


def summarise_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "status" not in results_df.columns:
        return pd.DataFrame()
    ok = results_df[results_df["status"] == "ok"].copy()
    if ok.empty:
        return ok
    group_cols = ["dataset", "task", "model", "model_family"]
    metric_candidates = ["accuracy", "f1_macro", "rmse", "mae", "r2", "train_seconds"]
    metrics = [col for col in metric_candidates if col in ok.columns]
    summary = ok.groupby(group_cols, as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns.to_flat_index()
    ]
    counts = ok.groupby(group_cols, as_index=False).size().rename(columns={"size": "runs_completed"})
    return counts.merge(summary, on=group_cols, how="left")


def build_leaderboard(summary_df: pd.DataFrame, task: str, expected_runs_per_model: int) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    leaderboard = summary_df.copy()
    leaderboard["runs_expected"] = int(expected_runs_per_model)
    leaderboard["completion_ratio"] = leaderboard["runs_completed"] / max(expected_runs_per_model, 1)
    if task == "classification":
        primary_metric = "accuracy"
        leaderboard["primary_metric_name"] = primary_metric
        leaderboard["primary_metric_direction"] = "higher_is_better"
        leaderboard["primary_metric"] = leaderboard["accuracy_mean"]
        sort_cols = ["accuracy_mean"]
        ascending = [False]
        if "f1_macro_mean" in leaderboard.columns:
            sort_cols.append("f1_macro_mean")
            ascending.append(False)
    else:
        primary_metric = "rmse"
        leaderboard["primary_metric_name"] = primary_metric
        leaderboard["primary_metric_direction"] = "lower_is_better"
        leaderboard["primary_metric"] = leaderboard["rmse_mean"]
        sort_cols = ["rmse_mean"]
        ascending = [True]
        if "mae_mean" in leaderboard.columns:
            sort_cols.append("mae_mean")
            ascending.append(True)
        if "r2_mean" in leaderboard.columns:
            sort_cols.append("r2_mean")
            ascending.append(False)
    if "train_seconds_mean" in leaderboard.columns:
        sort_cols.append("train_seconds_mean")
        ascending.append(True)
    leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    preferred_cols = [
        "rank",
        "dataset",
        "task",
        "model",
        "model_family",
        "runs_completed",
        "runs_expected",
        "completion_ratio",
        "primary_metric_name",
        "primary_metric_direction",
        "primary_metric",
        "accuracy_mean",
        "accuracy_std",
        "f1_macro_mean",
        "f1_macro_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "train_seconds_mean",
        "train_seconds_std",
    ]
    ordered_cols = [col for col in preferred_cols if col in leaderboard.columns]
    remaining_cols = [col for col in leaderboard.columns if col not in ordered_cols]
    return leaderboard[ordered_cols + remaining_cols]


def extract_best_so_far(leaderboard_df: pd.DataFrame) -> Optional[dict]:
    if leaderboard_df.empty:
        return None
    best = leaderboard_df.iloc[0]
    payload = {
        "rank": int(best["rank"]),
        "model": str(best["model"]),
        "model_family": str(best["model_family"]),
        "runs_completed": int(best["runs_completed"]),
        "runs_expected": int(best["runs_expected"]),
        "primary_metric_name": str(best["primary_metric_name"]),
        "primary_metric_direction": str(best["primary_metric_direction"]),
        "primary_metric": round(float(best["primary_metric"]), 6),
    }
    return payload


def write_analysis_outputs(
    results: List[dict],
    raw_results_path: Path,
    summary_path: Path,
    leaderboard_path: Path,
    task: str,
    expected_runs_per_model: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(columns=["dataset", "task", "model", "seed", "status"])
    results_df.to_csv(raw_results_path, index=False)
    summary_df = summarise_results(results_df)
    summary_df.to_csv(summary_path, index=False)
    leaderboard_df = build_leaderboard(summary_df, task, expected_runs_per_model)
    leaderboard_df.to_csv(leaderboard_path, index=False)
    return summary_df, leaderboard_df


def safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return max(parsed, 0.0)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    rounded = int(round(max(seconds, 0.0)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "#" * width
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = min(width, int(round(ratio * width)))
    return "#" * filled + "-" * (width - filled)


def collect_duration_history(results: List[dict]) -> tuple[Dict[str, List[float]], List[float]]:
    per_model: Dict[str, List[float]] = {}
    overall: List[float] = []
    for row in results:
        if str(row.get("status", "")) != "ok":
            continue
        seconds = safe_float(row.get("train_seconds"))
        if seconds is None:
            continue
        model_name = str(row["model"])
        per_model.setdefault(model_name, []).append(seconds)
        overall.append(seconds)
    return per_model, overall


def estimate_job_seconds(model_name: str, per_model: Dict[str, List[float]], overall: List[float]) -> Optional[float]:
    if model_name in per_model and per_model[model_name]:
        return float(np.median(per_model[model_name]))
    if overall:
        return float(np.median(overall))
    return None


def estimate_remaining_seconds(all_jobs: List[tuple[str, int]], results: List[dict]) -> Optional[float]:
    completed_ok = {
        result_key(row)
        for row in results
        if str(row.get("status", "")) == "ok"
    }
    pending_jobs = [job for job in all_jobs if job not in completed_ok]
    if not pending_jobs:
        return 0.0
    per_model, overall = collect_duration_history(results)
    estimated = 0.0
    for model_name, _seed in pending_jobs:
        job_seconds = estimate_job_seconds(model_name, per_model, overall)
        if job_seconds is None:
            return None
        estimated += job_seconds
    return estimated


def build_progress_snapshot(
    *,
    dataset: str,
    task: str,
    all_jobs: List[tuple[str, int]],
    results: List[dict],
    session_start: float,
    best_so_far: Optional[dict] = None,
    current_job: Optional[dict] = None,
    last_event: Optional[str] = None,
    last_run_seconds: Optional[float] = None,
) -> dict:
    all_job_keys = set(all_jobs)
    ok_keys = {
        result_key(row)
        for row in results
        if str(row.get("status", "")) == "ok" and result_key(row) in all_job_keys
    }
    failed_keys = {
        result_key(row)
        for row in results
        if str(row.get("status", "")) == "failed" and result_key(row) in all_job_keys
    }
    total_runs = len(all_jobs)
    completed_ok = len(ok_keys)
    failed_runs = len(failed_keys - ok_keys)
    remaining_runs = max(total_runs - completed_ok, 0)
    percent_complete = (completed_ok / total_runs) if total_runs else 1.0
    elapsed_seconds = time.perf_counter() - session_start
    eta_seconds = estimate_remaining_seconds(all_jobs, results)
    snapshot = {
        "dataset": dataset,
        "task": task,
        "total_runs": total_runs,
        "completed_ok": completed_ok,
        "failed_runs": failed_runs,
        "remaining_runs": remaining_runs,
        "percent_complete": round(percent_complete, 4),
        "progress_bar": render_progress_bar(completed_ok, total_runs),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "elapsed_human": format_duration(elapsed_seconds),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 4),
        "eta_human": format_duration(eta_seconds),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if current_job is not None:
        snapshot["current_job"] = current_job
    if best_so_far is not None:
        snapshot["best_so_far"] = best_so_far
    if last_event is not None:
        snapshot["last_event"] = last_event
    if last_run_seconds is not None:
        snapshot["last_run_seconds"] = round(last_run_seconds, 4)
        snapshot["last_run_human"] = format_duration(last_run_seconds)
    return snapshot


def format_progress_prefix(snapshot: dict) -> str:
    failed_suffix = f"|failed {snapshot['failed_runs']}" if snapshot["failed_runs"] else ""
    return (
        f"[{snapshot['completed_ok']}/{snapshot['total_runs']}"
        f"|{snapshot['progress_bar']}"
        f"|{snapshot['percent_complete'] * 100:5.1f}%"
        f"|elapsed {snapshot['elapsed_human']}"
        f"|eta {snapshot['eta_human']}{failed_suffix}]"
    )


def load_existing_results(raw_results_path: Path, resume: bool) -> List[dict]:
    if not resume or not raw_results_path.exists():
        return []
    existing = pd.read_csv(raw_results_path)
    if existing.empty:
        return []
    return existing.to_dict(orient="records")


def result_key(row: dict) -> tuple[str, int]:
    return (str(row["model"]), int(row["seed"]))


def upsert_result(results: List[dict], row: dict) -> None:
    key = result_key(row)
    for idx, existing in enumerate(results):
        if result_key(existing) == key:
            results[idx] = row
            return
    results.append(row)


def runtime_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_output_dir(args)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dump_json(run_dir / "run_config.json", vars(args))
    dump_json(run_dir / "runtime_info.json", runtime_info())

    bundle = load_dataset(args.dataset, data_dir, args.sample_size, args.split_seed)
    models = normalise_models(bundle.task, args.models)
    train, valid, test = split_dataset(
        frame=bundle.frame,
        target=bundle.target,
        task=bundle.task,
        test_size=args.test_size,
        val_size=args.val_size,
        split_seed=args.split_seed,
    )
    train = subsample_train_split(
        train=train,
        target=bundle.target,
        task=bundle.task,
        train_fraction=args.train_fraction,
        seed=args.train_subsample_seed,
    )
    dump_json(run_dir / "dataset_profile.json", dataset_profile(bundle, train, valid, test))

    raw_results_path = run_dir / "raw_results.csv"
    summary_path = run_dir / "summary.csv"
    leaderboard_path = run_dir / "leaderboard.csv"
    results: List[dict] = load_existing_results(raw_results_path, args.resume)
    error_dir = run_dir / "errors"
    error_dir.mkdir(exist_ok=True)
    progress_path = run_dir / "progress.json"
    session_start = time.perf_counter()
    completed_runs = {
        result_key(row)
        for row in results
        if str(row.get("status", "")) == "ok"
    }
    all_jobs = [(model_name, seed) for seed in args.seeds for model_name in models]
    completed_runs = completed_runs & set(all_jobs)
    _summary_df, leaderboard_df = write_analysis_outputs(
        results=results,
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        leaderboard_path=leaderboard_path,
        task=bundle.task,
        expected_runs_per_model=len(args.seeds),
    )
    best_so_far = extract_best_so_far(leaderboard_df)
    initial_snapshot = build_progress_snapshot(
        dataset=bundle.name,
        task=bundle.task,
        all_jobs=all_jobs,
        results=results,
        session_start=session_start,
        best_so_far=best_so_far,
        last_event="run initialized",
    )
    dump_json(progress_path, initial_snapshot)
    print(
        f"[PLAN] dataset={bundle.name} task={bundle.task} total_runs={len(all_jobs)} "
        f"completed_ok={initial_snapshot['completed_ok']} remaining={initial_snapshot['remaining_runs']} "
        f"accelerator={args.accelerator} devices={args.devices}"
    )
    print(f"[PLAN] progress file: {progress_path}")

    for seed in args.seeds:
        for model_name in models:
            key = (model_name, seed)
            if key in completed_runs:
                skip_snapshot = build_progress_snapshot(
                    dataset=bundle.name,
                    task=bundle.task,
                    all_jobs=all_jobs,
                    results=results,
                    session_start=session_start,
                    best_so_far=best_so_far,
                    last_event=f"skipped {model_name} seed={seed}",
                )
                dump_json(progress_path, skip_snapshot)
                print(
                    f"{format_progress_prefix(skip_snapshot)} [SKIP] "
                    f"dataset={bundle.name} model={model_name} seed={seed} already completed"
                )
                continue
            error_path = error_dir / f"{model_name}_seed{seed}.txt"
            if error_path.exists():
                error_path.unlink()
            row = {
                "dataset": bundle.name,
                "task": bundle.task,
                "model": model_name,
                "seed": seed,
                "status": "ok",
                "train_fraction": args.train_fraction,
            }
            run_snapshot = build_progress_snapshot(
                dataset=bundle.name,
                task=bundle.task,
                all_jobs=all_jobs,
                results=results,
                session_start=session_start,
                best_so_far=best_so_far,
                current_job={"dataset": bundle.name, "model": model_name, "seed": seed},
                last_event=f"running {model_name} seed={seed}",
            )
            dump_json(progress_path, run_snapshot)
            print(
                f"{format_progress_prefix(run_snapshot)} [RUN] "
                f"dataset={bundle.name} model={model_name} seed={seed}"
            )
            try:
                if model_name in DL_MODEL_FACTORIES:
                    payload = run_deep_model(model_name, bundle, train, valid, test, seed, run_dir, args)
                else:
                    payload = run_sklearn_model(
                        model_name,
                        bundle,
                        train,
                        test,
                        seed,
                        catboost_iterations=args.catboost_iterations,
                    )
                row.update(payload)
            except Exception as exc:  # pragma: no cover - defensive runtime handling
                row["status"] = "failed"
                row["error"] = repr(exc)
                row["traceback_path"] = f"errors/{model_name}_seed{seed}.txt"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
            upsert_result(results, row)
            if row["status"] == "ok":
                completed_runs.add(key)
            _summary_df, leaderboard_df = write_analysis_outputs(
                results=results,
                raw_results_path=raw_results_path,
                summary_path=summary_path,
                leaderboard_path=leaderboard_path,
                task=bundle.task,
                expected_runs_per_model=len(args.seeds),
            )
            best_so_far = extract_best_so_far(leaderboard_df)
            finished_snapshot = build_progress_snapshot(
                dataset=bundle.name,
                task=bundle.task,
                all_jobs=all_jobs,
                results=results,
                session_start=session_start,
                best_so_far=best_so_far,
                last_event=f"{row['status']} {model_name} seed={seed}",
                last_run_seconds=safe_float(row.get("train_seconds")),
            )
            dump_json(progress_path, finished_snapshot)
            if row["status"] == "ok":
                print(
                    f"{format_progress_prefix(finished_snapshot)} [OK] "
                    f"dataset={bundle.name} model={model_name} seed={seed} "
                    f"took={format_duration(safe_float(row.get('train_seconds')))}"
                )
            else:
                print(
                    f"{format_progress_prefix(finished_snapshot)} [FAIL] "
                    f"dataset={bundle.name} model={model_name} seed={seed}: {row['error']}"
                )

    _summary_df, leaderboard_df = write_analysis_outputs(
        results=results,
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        leaderboard_path=leaderboard_path,
        task=bundle.task,
        expected_runs_per_model=len(args.seeds),
    )
    best_so_far = extract_best_so_far(leaderboard_df)
    final_snapshot = build_progress_snapshot(
        dataset=bundle.name,
        task=bundle.task,
        all_jobs=all_jobs,
        results=results,
        session_start=session_start,
        best_so_far=best_so_far,
        last_event="run finished",
    )
    dump_json(progress_path, final_snapshot)
    print(f"[DONE] Results saved to {run_dir}")


if __name__ == "__main__":
    main()
