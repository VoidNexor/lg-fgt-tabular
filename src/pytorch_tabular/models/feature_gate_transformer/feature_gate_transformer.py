"""Feature Gate Transformer model."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from pytorch_tabular.models.common.layers import AppendCLSToken, Embedding2dLayer, TransformerEncoderBlock
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel


class FeatureGateTransformerBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_mlp(self, input_dim: int, layers: str) -> tuple[nn.Module, int]:
        if not layers:
            return nn.Identity(), input_dim
        modules = []
        current_dim = input_dim
        for units in layers.split("-"):
            modules.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    current_dim,
                    int(units),
                    self.hparams.dropout,
                )
            )
            current_dim = int(units)
        seq = nn.Sequential(*modules)
        _initialize_layers(self.hparams.activation, self.hparams.initialization, seq)
        return seq, current_dim

    def _build_gate_network(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hparams.input_embed_dim, self.hparams.feature_gate_hidden_dim),
            getattr(nn, self.hparams.gating_activation)(),
            nn.Dropout(self.hparams.feature_gate_dropout),
            nn.Linear(self.hparams.feature_gate_hidden_dim, output_dim),
        )

    def _initialize_linear_modules(self, module: nn.Module) -> None:
        for child in module.modules():
            if isinstance(child, nn.Linear):
                _initialize_layers(self.hparams.activation, self.hparams.initialization, child)

    def _initialize_ensemble_parameters(self, tensor: nn.Parameter, mean: float) -> None:
        with torch.no_grad():
            tensor.normal_(mean=mean, std=0.02)

    def _forward_fusion_branch(self, fused_features: torch.Tensor) -> torch.Tensor:
        if self._fusion_ensemble_size <= 1:
            return self.fusion_branch(fused_features)

        ensemble_outputs = []
        for member_idx in range(self._fusion_ensemble_size):
            member_input = fused_features * self.fusion_input_scales[member_idx]
            member_input = F.dropout(
                member_input,
                p=self._fusion_ensemble_dropout,
                training=self.training,
            )
            member_output = self.fusion_branch(member_input)
            member_output = (
                member_output * self.fusion_output_scales[member_idx] + self.fusion_output_bias[member_idx]
            )
            ensemble_outputs.append(member_output)
        return torch.stack(ensemble_outputs, dim=0).mean(dim=0)

    def _build_network(self):
        num_feature_tokens = self.hparams.continuous_dim + self.hparams.categorical_dim
        self._use_feature_gating = bool(self.hparams.use_feature_gating)
        self._use_branch_fusion_gate = bool(self.hparams.use_branch_fusion_gate)
        self._force_single_fusion = False
        self.feature_gating_active_ = self._use_feature_gating
        self.feature_importance_ = None
        self.token_pool_feature_importance_ = None
        self.local_feature_importance = None
        self.local_token_pool_importance = None
        self.branch_importance_ = None
        output_dim = int(getattr(self.hparams, "output_dim", 0) or 0)
        if (
            self._use_feature_gating
            and self.hparams.auto_disable_gating_on_multiclass_inputs
            and self.hparams.task == "classification"
            and output_dim > 2
        ):
            self._use_feature_gating = False
            self.feature_gating_active_ = False
        if (
            self._use_feature_gating
            and self.hparams.auto_disable_gating_on_wide_inputs
            and self.hparams.gating_max_tokens is not None
            and num_feature_tokens > self.hparams.gating_max_tokens
        ):
            self._use_feature_gating = False
            self.feature_gating_active_ = False
        if (
            self.hparams.task == "regression"
            and self.hparams.auto_simplify_categorical_regression
            and self.hparams.categorical_dim >= self.hparams.categorical_regression_simplify_min_tokens
        ):
            self._use_feature_gating = False
            self._use_branch_fusion_gate = False
            self._force_single_fusion = True
            self.feature_gating_active_ = False
        if not any(
            [
                self.hparams.use_local_branch,
                self.hparams.use_global_branch,
                self.hparams.use_raw_shortcut,
            ]
        ):
            raise ValueError("At least one of local branch, global branch, or raw shortcut must be enabled.")
        self.input_norm = nn.LayerNorm(self.hparams.input_embed_dim)
        if self._use_feature_gating:
            self.token_gate = self._build_gate_network(1)
            self.context_gate = self._build_gate_network(1)
        else:
            self.token_gate = None
            self.context_gate = None
        self.token_dropout = nn.Dropout(self.hparams.feature_gate_dropout)

        if self.hparams.use_global_branch:
            self.add_cls = AppendCLSToken(
                d_token=self.hparams.input_embed_dim,
                initialization=self.hparams.embedding_initialization,
            )
            self.transformer_blocks = OrderedDict()
            for i in range(self.hparams.num_attn_blocks):
                self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                    input_embed_dim=self.hparams.input_embed_dim,
                    num_heads=self.hparams.num_heads,
                    ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                    ff_activation=self.hparams.transformer_activation,
                    attn_dropout=self.hparams.attn_dropout,
                    ff_dropout=self.hparams.ff_dropout,
                    add_norm_dropout=self.hparams.add_norm_dropout,
                    keep_attn=False,
                    transformer_head_dim=self.hparams.transformer_head_dim,
                )
            self.transformer_blocks = nn.Sequential(self.transformer_blocks)
            self.token_pool = nn.Linear(self.hparams.input_embed_dim, 1)
            global_output_dim = 2 * self.hparams.input_embed_dim
            self.global_norm = nn.LayerNorm(global_output_dim)
        else:
            self.add_cls = None
            self.transformer_blocks = None
            self.token_pool = None
            self.global_norm = None
            global_output_dim = 0

        local_input_dim = max(num_feature_tokens, 1) * self.hparams.input_embed_dim
        if self.hparams.use_local_branch:
            self.local_branch, local_output_dim = self._build_mlp(local_input_dim, self.hparams.mlp_layers)
            self.local_norm = nn.LayerNorm(local_output_dim)
        else:
            self.local_branch = None
            self.local_norm = None
            local_output_dim = 0

        if self.hparams.use_raw_shortcut:
            self.raw_token_projection = nn.Sequential(
                nn.Linear(self.hparams.input_embed_dim, self.hparams.input_embed_dim),
                nn.LayerNorm(self.hparams.input_embed_dim),
                getattr(nn, self.hparams.gating_activation)(),
                nn.Dropout(self.hparams.feature_gate_dropout),
            )
            self._initialize_linear_modules(self.raw_token_projection)
            raw_output_dim = self.hparams.input_embed_dim
        else:
            self.raw_token_projection = None
            raw_output_dim = 0

        if self.hparams.use_local_branch and self.hparams.use_global_branch and self._use_branch_fusion_gate:
            branch_gate_input_dim = global_output_dim + local_output_dim
            self.branch_gate = nn.Sequential(
                nn.Linear(branch_gate_input_dim, self.hparams.branch_gate_hidden_dim),
                getattr(nn, self.hparams.gating_activation)(),
                nn.Dropout(self.hparams.feature_gate_dropout),
                nn.Linear(self.hparams.branch_gate_hidden_dim, 2),
            )
            self._initialize_linear_modules(self.branch_gate)
        else:
            self.branch_gate = None

        fusion_input_dim = global_output_dim + local_output_dim + raw_output_dim
        self.fusion_branch, fusion_output_dim = self._build_mlp(fusion_input_dim, self.hparams.fusion_layers)
        self._fusion_ensemble_size = max(1, int(self.hparams.fusion_ensemble_size))
        self._fusion_ensemble_dropout = float(self.hparams.fusion_ensemble_dropout)
        if self._force_single_fusion:
            self._fusion_ensemble_size = 1
            self._fusion_ensemble_dropout = 0.0
        elif (
            self.hparams.task == "regression"
            and self.hparams.auto_enable_categorical_regression_ensemble
            and self.hparams.categorical_dim >= self.hparams.categorical_regression_ensemble_min_tokens
            and self._fusion_ensemble_size <= 1
        ):
            self._fusion_ensemble_size = max(1, int(self.hparams.categorical_regression_ensemble_size))
            self._fusion_ensemble_dropout = float(self.hparams.categorical_regression_ensemble_dropout)
        if self._fusion_ensemble_size > 1:
            self.fusion_input_scales = nn.Parameter(
                torch.empty(self._fusion_ensemble_size, fusion_input_dim),
            )
            self.fusion_output_scales = nn.Parameter(
                torch.empty(self._fusion_ensemble_size, fusion_output_dim),
            )
            self.fusion_output_bias = nn.Parameter(
                torch.zeros(self._fusion_ensemble_size, fusion_output_dim),
            )
            self._initialize_ensemble_parameters(self.fusion_input_scales, mean=1.0)
            self._initialize_ensemble_parameters(self.fusion_output_scales, mean=1.0)
        else:
            self.fusion_input_scales = None
            self.fusion_output_scales = None
            self.fusion_output_bias = None
        self.output_dim = fusion_output_dim

    def _build_embedding_layer(self):
        shared_embedding_strategy = self.hparams.share_embedding_strategy if self.hparams.share_embedding else None
        return Embedding2dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.input_embed_dim,
            shared_embedding_strategy=shared_embedding_strategy,
            frac_shared_embed=self.hparams.shared_embedding_fraction,
            embedding_bias=self.hparams.embedding_bias,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            embedding_dropout=self.hparams.embedding_dropout,
            initialization=self.hparams.embedding_initialization,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )

    def _reorder_to_input_features(self, scores: np.ndarray) -> np.ndarray:
        if self.hparams.continuous_dim > 0 and self.hparams.categorical_dim > 0:
            return np.concatenate([scores[self.hparams.continuous_dim :], scores[: self.hparams.continuous_dim]])
        return scores

    def _update_feature_importance(self, gates: torch.Tensor):
        gate_scores = gates.squeeze(-1).detach()
        mean_scores = gate_scores.mean(dim=0).cpu().numpy()
        self.local_feature_importance = gate_scores
        self.feature_importance_ = self._reorder_to_input_features(mean_scores)

    def _update_token_pool_importance(self, pool_weights: torch.Tensor):
        pool_scores = pool_weights.detach()
        mean_scores = pool_scores.mean(dim=0).cpu().numpy()
        self.local_token_pool_importance = pool_scores
        self.token_pool_feature_importance_ = self._reorder_to_input_features(mean_scores)

    def _update_branch_importance(self, branch_weights: torch.Tensor):
        self.branch_importance_ = branch_weights.detach().mean(dim=0).cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_norm(x)
        if self._use_feature_gating:
            context = tokens.mean(dim=1, keepdim=True)
            gate_temperature = max(float(self.hparams.gate_temperature), 1e-6)
            gate_logits = self.token_gate(tokens) + self.context_gate(context)
            gates = torch.sigmoid(gate_logits / gate_temperature)
            gate_residual = 1.0 + self.hparams.residual_gate_scale * ((2.0 * gates) - 1.0)
            gated_tokens = self.token_dropout(tokens * gate_residual)
        else:
            gates = torch.ones(
                tokens.size(0),
                tokens.size(1),
                1,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            gated_tokens = tokens
        self._update_feature_importance(gates)

        local_features = None
        if self.hparams.use_local_branch:
            local_features = self.local_norm(self.local_branch(gated_tokens.flatten(start_dim=1)))

        global_features = None
        if self.hparams.use_global_branch:
            transformer_tokens = self.add_cls(gated_tokens)
            for block in self.transformer_blocks:
                transformer_tokens = block(transformer_tokens)
            cls_token = transformer_tokens[:, -1]
            feature_tokens = transformer_tokens[:, :-1]
            pool_weights = torch.softmax(self.token_pool(feature_tokens).squeeze(-1), dim=1)
            self._update_token_pool_importance(pool_weights)
            pooled_features = torch.sum(feature_tokens * pool_weights.unsqueeze(-1), dim=1)
            global_features = self.global_norm(torch.cat([cls_token, pooled_features], dim=1))

        if self.branch_gate is not None:
            branch_weights = torch.softmax(self.branch_gate(torch.cat([global_features, local_features], dim=1)), dim=1)
            self._update_branch_importance(branch_weights)
            global_features = global_features * branch_weights[:, 0:1]
            local_features = local_features * branch_weights[:, 1:2]

        fusion_parts = []
        if global_features is not None:
            fusion_parts.append(global_features)
        if local_features is not None:
            fusion_parts.append(local_features)
        if self.hparams.use_raw_shortcut:
            fusion_parts.append(self.raw_token_projection(tokens.mean(dim=1)))

        fused_features = torch.cat(fusion_parts, dim=1)
        return self._forward_fusion_branch(fused_features)


class FeatureGateTransformerModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        self._backbone = FeatureGateTransformerBackbone(self.hparams)
        self._embedding_layer = self._backbone._build_embedding_layer()
        self._head = self._get_head_from_config()
