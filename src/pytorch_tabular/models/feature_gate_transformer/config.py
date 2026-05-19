"""Feature Gate Transformer Config."""

from dataclasses import dataclass, field
from typing import Optional

from pytorch_tabular.config import ModelConfig


@dataclass
class FeatureGateTransformerConfig(ModelConfig):
    """Hybrid tabular model with feature gating, transformer interactions, and an MLP fusion branch.

    This config is intended as a strong research starting point when pure transformer models underperform
    on tabular data but we still want explicit feature interaction modeling.
    """

    input_embed_dim: int = field(
        default=32,
        metadata={"help": "Embedding dimension used for categorical and continuous feature tokens."},
    )
    embedding_initialization: Optional[str] = field(
        default="kaiming_uniform",
        metadata={
            "help": "Initialization scheme for embedding layers.",
            "choices": ["kaiming_uniform", "kaiming_normal"],
        },
    )
    embedding_bias: bool = field(
        default=True,
        metadata={"help": "Whether to learn a bias term for feature token embeddings."},
    )
    share_embedding: bool = field(
        default=False,
        metadata={"help": "Whether to inject shared feature embeddings for categorical columns."},
    )
    share_embedding_strategy: Optional[str] = field(
        default="fraction",
        metadata={
            "help": "Strategy for shared categorical embeddings.",
            "choices": ["add", "fraction"],
        },
    )
    shared_embedding_fraction: float = field(
        default=0.25,
        metadata={"help": "Fraction of embedding dimensions reserved for shared categorical embeddings."},
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of heads in each transformer block."},
    )
    num_attn_blocks: int = field(
        default=3,
        metadata={"help": "Number of stacked transformer blocks."},
    )
    transformer_head_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Optional head dimension override inside multi-head attention."},
    )
    attn_dropout: float = field(
        default=0.1,
        metadata={"help": "Attention dropout."},
    )
    add_norm_dropout: float = field(
        default=0.1,
        metadata={"help": "Residual/add-norm dropout."},
    )
    ff_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout inside transformer feed-forward blocks."},
    )
    ff_hidden_multiplier: int = field(
        default=4,
        metadata={"help": "Expansion multiplier for transformer feed-forward layers."},
    )
    transformer_activation: str = field(
        default="GEGLU",
        metadata={"help": "Activation used in transformer feed-forward blocks."},
    )
    feature_gate_hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden width of the feature gate network."},
    )
    feature_gate_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout used inside the gate network and on gated tokens."},
    )
    residual_gate_scale: float = field(
        default=0.5,
        metadata={"help": "Residual scaling strength for adaptive feature gating."},
    )
    gate_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature applied before the sigmoid gate activation."},
    )
    gating_activation: str = field(
        default="GELU",
        metadata={"help": "Activation function used inside the gate network."},
    )
    branch_gate_hidden_dim: int = field(
        default=64,
        metadata={"help": "Hidden width of the branch fusion gate network."},
    )
    use_feature_gating: bool = field(
        default=True,
        metadata={"help": "Whether to apply sample-wise feature gating before the local/global branches."},
    )
    auto_disable_gating_on_wide_inputs: bool = field(
        default=False,
        metadata={"help": "Whether to automatically disable feature gating when the number of feature tokens exceeds a threshold."},
    )
    gating_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Optional token-count threshold for automatic gating disablement on wide inputs."},
    )
    auto_disable_gating_on_multiclass_inputs: bool = field(
        default=False,
        metadata={"help": "Whether to automatically disable feature gating for multiclass classification tasks."},
    )
    use_local_branch: bool = field(
        default=True,
        metadata={"help": "Whether to keep the flattened local MLP branch active."},
    )
    use_global_branch: bool = field(
        default=True,
        metadata={"help": "Whether to keep the transformer interaction branch active."},
    )
    use_branch_fusion_gate: bool = field(
        default=True,
        metadata={"help": "Whether to use a learned gate to rebalance the local and global branches."},
    )
    use_raw_shortcut: bool = field(
        default=True,
        metadata={"help": "Whether to include the raw token summary shortcut in the final fusion stage."},
    )
    fusion_ensemble_size: int = field(
        default=1,
        metadata={"help": "Number of parameter-efficient fusion ensemble members. 1 disables ensembling."},
    )
    fusion_ensemble_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout applied to each fusion ensemble member input before the shared fusion MLP."},
    )
    auto_enable_categorical_regression_ensemble: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable a small fusion ensemble automatically for categorical-rich regression inputs.",
        },
    )
    categorical_regression_ensemble_min_tokens: int = field(
        default=4,
        metadata={"help": "Minimum number of categorical feature tokens required for automatic regression ensembling."},
    )
    categorical_regression_ensemble_size: int = field(
        default=2,
        metadata={"help": "Fusion ensemble size used when categorical-rich regression auto-ensembling is active."},
    )
    categorical_regression_ensemble_dropout: float = field(
        default=0.03,
        metadata={"help": "Fusion ensemble dropout used by categorical-rich regression auto-ensembling."},
    )
    auto_simplify_categorical_regression: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable gating, branch gating, and fusion ensembling on categorical-rich regression inputs.",
        },
    )
    categorical_regression_simplify_min_tokens: int = field(
        default=4,
        metadata={"help": "Minimum number of categorical tokens required for categorical-rich regression simplification."},
    )
    mlp_layers: str = field(
        default="128-64",
        metadata={"help": "Hyphen-separated hidden sizes for the local MLP branch."},
    )
    fusion_layers: str = field(
        default="128",
        metadata={"help": "Hyphen-separated hidden sizes for the final fusion branch."},
    )
    activation: str = field(
        default="ReLU",
        metadata={"help": "Activation used in the local MLP and fusion branches."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Whether to use batch norm in the local MLP and fusion branches."},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for local MLP and fusion layers.",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout used in the local MLP and fusion branches."},
    )

    _module_src: str = field(default="models.feature_gate_transformer")
    _model_name: str = field(default="FeatureGateTransformerModel")
    _backbone_name: str = field(default="FeatureGateTransformerBackbone")
    _config_name: str = field(default="FeatureGateTransformerConfig")
