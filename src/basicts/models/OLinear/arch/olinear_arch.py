"""OLinear adapted to the BasicTS forecasting interface.

The implementation follows the official NeurIPS 2025 architecture: RevIN,
learnable dimension extension, a training-set OrthoTrans basis, NormLin
cross-series mixing, an intra-series linear learner, and transformed-domain
decoding. Official code: https://github.com/jackyue1994/OLinear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from basicts.modules.norm import RevIN
from basicts.modules.dpr import dpr_orthogonal_loss

from ..config.olinear_config import OLinearConfig


def _temporal_correlation_basis(data: np.ndarray, length: int) -> np.ndarray:
    """Compute the paper's average temporal Pearson-correlation eigenbasis."""
    if data.ndim != 2:
        raise ValueError(f"Expected [time, variables], got {data.shape}")
    if length < 1 or length > data.shape[0]:
        raise ValueError(f"Invalid basis length {length} for training length {data.shape[0]}")

    corr = np.zeros((length, length), dtype=np.float64)
    used = 0
    for channel in range(data.shape[1]):
        series = np.asarray(data[:, channel], dtype=np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(series, length)
        channel_corr = np.corrcoef(windows, rowvar=False)
        if np.isfinite(channel_corr).all():
            corr += channel_corr
            used += 1
    if used == 0:
        return np.eye(length, dtype=np.float32)
    corr /= used
    corr = np.nan_to_num((corr + corr.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    # The official files store eigenvectors row-wise (Q = V^T). The forward
    # transform uses x @ Q^T and the inverse transform uses z @ Q.
    return eigenvectors[:, order].T.astype(np.float32)


def prepare_olinear_bases(
    train_data: np.ndarray,
    input_len: int,
    output_len: int,
    save_path: str | Path,
) -> str:
    """Create Q matrices from training data only and save them atomically."""
    save_path = Path(save_path)
    if save_path.exists():
        return str(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    q_in = _temporal_correlation_basis(train_data, input_len)
    q_out = _temporal_correlation_basis(train_data, output_len)
    temporary = save_path.with_suffix(save_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, q_in=q_in, q_out=q_out)
    temporary.replace(save_path)
    return str(save_path)


class NormLinBlock(nn.Module):
    """Official positive row-normalized token mixing plus point-wise FFN."""

    def __init__(self, num_features: int, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        initial = torch.eye(num_features) + torch.randn(num_features, num_features)
        self.weight = nn.Parameter(initial.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn1 = nn.Linear(hidden_size, intermediate_size)
        self.ffn2 = nn.Linear(intermediate_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.value_projection(x)
        weight = F.normalize(F.softplus(self.weight), p=1, dim=-1)
        weight = self.dropout(weight)
        mixed = weight @ values
        x = self.norm1(x + self.dropout(self.output_projection(mixed)))
        y = self.ffn2(self.dropout(F.relu(self.ffn1(x))))
        return self.norm2(x + self.dropout(y))


class OLinearForForecasting(nn.Module):
    def __init__(self, config: OLinearConfig):
        super().__init__()
        if not config.q_mat_path:
            raise ValueError("OLinearConfig.q_mat_path must point to training-only OrthoTrans matrices")
        matrices = np.load(config.q_mat_path)
        q_in = torch.as_tensor(matrices["q_in"], dtype=torch.float32)
        q_out = torch.as_tensor(matrices["q_out"], dtype=torch.float32)
        if q_in.shape != (config.input_len, config.input_len):
            raise ValueError(f"Unexpected q_in shape {tuple(q_in.shape)}")
        if q_out.shape != (config.output_len, config.output_len):
            raise ValueError(f"Unexpected q_out shape {tuple(q_out.shape)}")
        self.register_buffer("q_in", q_in, persistent=True)
        self.register_buffer("q_out", q_out, persistent=True)

        self.num_features = config.num_features
        self.output_len = config.output_len
        self.embed_size = config.embed_size
        self.use_revin = config.use_revin
        self.revin = RevIN(config.num_features) if self.use_revin else None
        self.dimension_extension = nn.Parameter(torch.randn(1, config.embed_size))
        self.input_projection = nn.Linear(config.input_len * config.embed_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            NormLinBlock(
                config.num_features,
                config.hidden_size,
                config.intermediate_size,
                config.dropout,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.transformed_decoder = nn.Linear(
            config.hidden_size, config.output_len * config.embed_size
        )
        self.output_head = nn.Sequential(
            nn.Linear(config.output_len * config.embed_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.output_len),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.dpr_cfg = config.dpr
        self.dpr = config.dpr.build_module(config.num_features * config.embed_size)
        self.delta_in = nn.Parameter(
            torch.zeros(1, config.num_features, 1, config.input_len)
        )
        self.delta_out = nn.Parameter(
            torch.zeros(1, config.num_features, 1, config.output_len)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.revin(inputs, "norm") if self.revin is not None else inputs
        # [B,T,N] -> [B,N,T,E] -> [B,N,E,T].
        x = x.transpose(1, 2).unsqueeze(-1) * self.dimension_extension
        x = x.transpose(-1, -2)
        # Q is shared by all samples and variables and is computed from train only.
        x = torch.einsum("bnet,tv->bnev", x, self.q_in.T) + self.delta_in
        batch, channels, _, _ = x.shape
        x = self.input_projection(x.flatten(-2))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.transformed_decoder(x).reshape(
            batch, channels, self.embed_size, -1
        )
        x = torch.einsum("bnet,tv->bnev", x, self.q_out) + self.delta_out
        if self.dpr is not None:
            # Reinterpret the decoded transformed representation as a temporal
            # hidden sequence [B, output_len, variables * embed_size].
            x = x.permute(0, 3, 1, 2).reshape(batch, x.shape[-1], -1)
            x = self.dpr(x)
            x = x.reshape(batch, self.output_len, self.num_features, self.embed_size)
            x = x.permute(0, 2, 3, 1)
        x = x.transpose(-1, -2).flatten(-2)
        prediction = self.dropout(self.output_head(x)).transpose(1, 2)
        prediction = self.revin(prediction, "denorm") if self.revin is not None else prediction
        if self.dpr is not None and self.dpr_cfg.orth_lambda > 0:
            return {
                "prediction": prediction,
                "dpr_orth": self.dpr_cfg.orth_lambda * dpr_orthogonal_loss(self.dpr.mode_table),
            }
        return prediction
