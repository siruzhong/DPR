"""Dynamic Pattern Routing (DPR): adaptive per-position modulation via local dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


def dpr_orthogonal_loss(mode_table: torch.Tensor) -> torch.Tensor:
    """
    Encourage rows of mode_table (K x d) to be orthonormal in expectation.

    Args:
        mode_table: Tensor of shape [K, d].

    Returns:
        Scalar: sum of squared Gram errors vs identity, divided by K (not K^2),
        so gradients to ``mode_table`` are not diluted by the full matrix size.
    """
    m = F.normalize(mode_table, dim=-1, eps=1e-6)
    gram = m @ m.T
    k = m.size(0)
    eye = torch.eye(k, device=m.device, dtype=m.dtype)
    return torch.sum((gram - eye) ** 2) / k


class TemporalContextualGating(nn.Module):
    """
    Residual adapter on hidden states [B, L, d].

    Multi-scale context uses depthwise Conv1d (``groups=d_model``): O(d*k) params per
    kernel, per-dimension local dynamics along L; mixing happens in ``context_mlp``.
    Then softmax routing over K learnable patterns and Hadamard modulation x * (1 + gamma * m).
    """

    def __init__(
        self,
        d_model: int,
        num_patterns: int = 8,
        use_multiscale: bool = True,
        conv_kernels: Optional[Sequence[int]] = None,
        identity_init: bool = True,
        discrete_topk: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.use_multiscale = use_multiscale
        self.discrete_topk = discrete_topk
        self.d_context = max(16, d_model // 4)

        if conv_kernels is None:
            kernels = (3, 7) if use_multiscale else (1,)
        else:
            kernels = tuple(int(k) for k in conv_kernels)
            if len(kernels) == 0:
                raise ValueError("conv_kernels must not be empty")
            if any(k <= 0 for k in kernels):
                raise ValueError(f"conv_kernels must be positive, got {kernels}")
        self.conv_kernels = kernels

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model, d_model, k, padding="same", groups=d_model, bias=True
                )
                for k in self.conv_kernels
            ]
        )
        ctx_in = len(self.conv_layers) * d_model

        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_in, self.d_context),
            nn.GELU(),
        )
        self.route_centroids = nn.Parameter(torch.randn(num_patterns, self.d_context))
        nn.init.normal_(self.route_centroids, std=0.02)
        self.routing_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.mode_table = nn.Parameter(torch.empty(num_patterns, d_model))
        if identity_init:
            self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.gamma = nn.Parameter(torch.randn(1) * 0.01)

        nn.init.normal_(self.mode_table, std=0.02)
        for conv in self.conv_layers:
            if conv is not None:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="linear")
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        nn.init.normal_(self.context_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.context_mlp[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """
        Args:
            x: [B, L, d_model]
            return_aux: If True, also return dict with routing_probs and optional orth term.

        Returns:
            Modulated x of shape [B, L, d_model], optionally with aux dict.
        """
        b, l, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d}")

        xt = x.transpose(1, 2)  # [B, d, L]
        conv_outs = [conv(xt) for conv in self.conv_layers]
        if len(conv_outs) == 1:
            z = conv_outs[0]
        else:
            z = torch.cat(conv_outs, dim=1)
        z = z.transpose(1, 2)  # [B, L, 2d] or [B, L, d]
        c = self.context_mlp(z)  # [B, L, d_c]
        c_norm = F.normalize(c, dim=-1) # [B, L, d_c]
        cent_norm = F.normalize(self.route_centroids, dim=-1) # [K, d_c]
        logits = torch.einsum("bld,kd->blk", c_norm, cent_norm) * self.routing_scale
        if self.discrete_topk > 1:
            k = self.discrete_topk
            topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
            p = torch.zeros_like(logits).scatter_(-1, topk_idx, F.softmax(topk_vals, dim=-1))
        else:
            p = F.softmax(logits, dim=-1)  # [B, L, P]
        m = torch.einsum("blk,kd->bld", p, self.mode_table)
        out = x * (1.0 + self.gamma * m)

        if not return_aux:
            return out

        aux: Dict[str, torch.Tensor] = {"routing_probs": p.detach()}
        return out, aux


def _resolve_kernels(use_multiscale: bool, conv_kernels: Optional[Sequence[int]]) -> tuple[int, ...]:
    if conv_kernels is None:
        return (3, 7) if use_multiscale else (1,)
    kernels = tuple(int(k) for k in conv_kernels)
    if not kernels or any(k <= 0 for k in kernels):
        raise ValueError(f"Invalid conv_kernels: {kernels}")
    return kernels


class _LocalPerception(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(d_model, d_model, k, padding="same", groups=d_model)
            for k in kernels
        )
        self.output_size = len(self.layers) * d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        return torch.cat([layer(xt) for layer in self.layers], dim=1).transpose(1, 2)


def _matched_bottleneck(input_size: int, output_size: int, fixed_params: int, target_params: int) -> int:
    best = min(
        range(1, max(2, target_params // max(1, input_size) + 2)),
        key=lambda width: abs(
            fixed_params + input_size * width + width + width * output_size + output_size - target_params
        ),
    )
    return max(1, best)


class GlobalSEAdapter(nn.Module):
    def __init__(self, d_model: int, target_params: int):
        super().__init__()
        width = _matched_bottleneck(d_model, d_model, 1, target_params)
        self.mlp = nn.Sequential(nn.Linear(d_model, width), nn.GELU(), nn.Linear(width, d_model))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = 2.0 * torch.sigmoid(self.mlp(x.mean(dim=1, keepdim=True))) - 1.0
        return x * (1.0 + self.gamma * gate)


class LocalSEAdapter(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int], target_params: int):
        super().__init__()
        self.perception = _LocalPerception(d_model, kernels)
        fixed = sum(p.numel() for p in self.perception.parameters()) + 1
        width = _matched_bottleneck(self.perception.output_size, d_model, fixed, target_params)
        self.mlp = nn.Sequential(
            nn.Linear(self.perception.output_size, width), nn.GELU(), nn.Linear(width, d_model)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = 2.0 * torch.sigmoid(self.mlp(self.perception(x))) - 1.0
        return x * (1.0 + self.gamma * gate)


class LocalFiLMAdapter(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int], target_params: int):
        super().__init__()
        self.d_model = d_model
        self.perception = _LocalPerception(d_model, kernels)
        fixed = sum(p.numel() for p in self.perception.parameters()) + 1
        width = _matched_bottleneck(self.perception.output_size, 2 * d_model, fixed, target_params)
        self.mlp = nn.Sequential(
            nn.Linear(self.perception.output_size, width), nn.GELU(), nn.Linear(width, 2 * d_model)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mlp(self.perception(x)).chunk(2, dim=-1)
        return x * (1.0 + self.gamma * torch.tanh(scale)) + self.gamma * shift


class GatedResidualAdapter(nn.Module):
    def __init__(self, d_model: int, kernels: Sequence[int], target_params: int):
        super().__init__()
        self.perception = _LocalPerception(d_model, kernels)
        fixed = sum(p.numel() for p in self.perception.parameters()) + 1
        width = _matched_bottleneck(self.perception.output_size, d_model, fixed, target_params)
        self.mlp = nn.Sequential(
            nn.Linear(self.perception.output_size, width), nn.GELU(), nn.Linear(width, d_model)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.mlp(self.perception(x)))
        return x + self.gamma * residual


def build_response_adapter(
    adapter_type: str,
    d_model: int,
    num_patterns: int = 8,
    use_multiscale: bool = True,
    conv_kernels: Optional[Sequence[int]] = None,
    identity_init: bool = True,
    discrete_topk: int = 1,
) -> nn.Module:
    """Build DPR or a parameter-matched conditional-modulation control."""
    kernels = _resolve_kernels(use_multiscale, conv_kernels)
    dpr = TemporalContextualGating(
        d_model=d_model,
        num_patterns=num_patterns,
        use_multiscale=use_multiscale,
        conv_kernels=kernels,
        identity_init=identity_init,
        discrete_topk=discrete_topk,
    )
    if adapter_type == "dpr":
        return dpr
    target_params = sum(p.numel() for p in dpr.parameters())
    if adapter_type == "global_se":
        return GlobalSEAdapter(d_model, target_params)
    if adapter_type == "local_se":
        return LocalSEAdapter(d_model, kernels, target_params)
    if adapter_type == "local_film":
        return LocalFiLMAdapter(d_model, kernels, target_params)
    if adapter_type == "gated_residual":
        return GatedResidualAdapter(d_model, kernels, target_params)
    raise ValueError(f"Unknown response adapter type: {adapter_type}")


def response_adapter_orthogonal_loss(adapter: nn.Module, weight: float) -> Optional[torch.Tensor]:
    mode_table = getattr(adapter, "mode_table", None)
    if mode_table is None or weight <= 0:
        return None
    return weight * dpr_orthogonal_loss(mode_table)
