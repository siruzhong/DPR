"""TimeBase, adapted from the authors' official ICML 2025 implementation.

Paper: TimeBase: The Power of Minimalism in Efficient Long-term Time Series
Forecasting. Official code: https://github.com/hqh0728/TimeBase (MIT license).
"""

from __future__ import annotations

import torch
from torch import nn
from basicts.modules.dpr import dpr_orthogonal_loss

from ..config.timebase_config import TimeBaseConfig


def temporal_basis_orthogonal_loss(matrix: torch.Tensor) -> torch.Tensor:
    gram = matrix.transpose(-2, -1) @ matrix
    diagonal = torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    return torch.linalg.matrix_norm(gram - diagonal, ord="fro", dim=(-2, -1)).mean()


class TimeBaseForForecasting(nn.Module):
    def __init__(self, config: TimeBaseConfig):
        super().__init__()
        if config.period_len <= 0:
            raise ValueError("period_len must be positive")
        if config.basis_num <= 0:
            raise ValueError("basis_num must be positive")

        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.period_len = config.period_len
        self.basis_num = config.basis_num
        self.use_period_norm = config.use_period_norm
        self.use_orthogonal = config.use_orthogonal
        self.orthogonal_weight = config.orthogonal_weight
        self.individual = config.individual
        self.dpr_cfg = config.dpr
        self.dpr = config.dpr.build_module(config.basis_num)

        self.input_segments = (self.input_len + self.period_len - 1) // self.period_len
        self.output_segments = (self.output_len + self.period_len - 1) // self.period_len
        self.input_padding = self.input_segments * self.period_len - self.input_len

        if self.individual:
            self.ts_to_basis = nn.ModuleList(
                nn.Linear(self.input_segments, self.basis_num)
                for _ in range(self.num_features)
            )
            self.basis_to_ts = nn.ModuleList(
                nn.Linear(self.basis_num, self.output_segments)
                for _ in range(self.num_features)
            )
        else:
            self.ts_to_basis = nn.Linear(self.input_segments, self.basis_num)
            self.basis_to_ts = nn.Linear(self.basis_num, self.output_segments)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_padding == 0:
            return x
        # Match the official implementation: repeat the most recent complete
        # period instead of introducing zeros at the sequence boundary.
        complete_start = max(0, (self.input_segments - 1) * self.period_len - self.input_padding)
        return torch.cat([x, x[..., complete_start:complete_start + self.input_padding]], dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        batch, _, channels = inputs.shape
        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} variables, got {channels}")

        x = self._pad(inputs.transpose(1, 2))
        x = x.reshape(batch, channels, self.input_segments, self.period_len)
        x = x.permute(0, 1, 3, 2)  # [B, C, P, N]

        if self.use_period_norm:
            center = x.mean(dim=-1, keepdim=True)
            x = x - center
        else:
            series_center = x.reshape(batch, channels, -1).mean(dim=-1, keepdim=True)
            x = x - series_center.unsqueeze(-1)

        if self.individual:
            basis = torch.stack(
                [self.ts_to_basis[i](x[:, i]) for i in range(channels)], dim=1
            )
            prediction = torch.stack(
                [self.basis_to_ts[i](basis[:, i]) for i in range(channels)], dim=1
            )
        else:
            basis = self.ts_to_basis(x)
            prediction = self.basis_to_ts(basis)

        raw_basis = basis
        if self.dpr is not None:
            b, c, p, k = basis.shape
            basis = self.dpr(basis.reshape(b * c, p, k)).reshape(b, c, p, k)
            if self.individual:
                prediction = torch.stack(
                    [self.basis_to_ts[i](basis[:, i]) for i in range(channels)], dim=1
                )
            else:
                prediction = self.basis_to_ts(basis)

        if self.use_period_norm:
            prediction = prediction + center
        else:
            prediction = prediction + series_center.unsqueeze(-1)

        prediction = prediction.permute(0, 1, 3, 2).reshape(batch, channels, -1)
        prediction = prediction[..., : self.output_len].transpose(1, 2)

        extras = {}
        if self.use_orthogonal and self.orthogonal_weight > 0:
            extras["timebase_orth"] = self.orthogonal_weight * temporal_basis_orthogonal_loss(
                raw_basis.reshape(-1, self.period_len, self.basis_num)
            )
        if self.dpr is not None and self.dpr_cfg.orth_lambda > 0:
            extras["dpr_orth"] = self.dpr_cfg.orth_lambda * dpr_orthogonal_loss(self.dpr.mode_table)
        if extras:
            return {"prediction": prediction, **extras}
        return prediction
