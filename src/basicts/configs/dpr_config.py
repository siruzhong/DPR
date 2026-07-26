"""Hyperparameters for Dynamic Pattern Routing (DPR)."""

from dataclasses import dataclass
from typing import Literal, Optional

from torch import nn

from basicts.modules.dpr import build_response_adapter


@dataclass
class DPRConfig:
    """When ``enabled`` is False, models skip building DPR and related losses.

    Ablation flags (default True/1 preserves original behavior):
        use_multiscale: Use k1=3, k2=7 depthwise convs (False -> k=1 point-wise only)
        conv_kernels: Optional explicit conv kernel sizes (e.g., (3,), (3, 7)).
            When provided, it overrides ``use_multiscale`` behavior.
        identity_init: Initialize gamma=0 (False -> gamma ~ N(0, 0.01))
        discrete_topk: Soft routing (1) vs discrete Top-K routing (>1, e.g. 2 for Top-2)
    """

    enabled: bool = False
    adapter_type: Literal["dpr", "global_se", "local_se", "local_film", "gated_residual"] = "dpr"
    num_patterns: int = 8
    orth_lambda: float = 0.01
    use_multiscale: bool = True
    conv_kernels: Optional[tuple[int, ...]] = None
    identity_init: bool = True
    discrete_topk: int = 1

    def build_module(self, d_model: int) -> Optional[nn.Module]:
        if not self.enabled:
            return None
        return build_response_adapter(
            adapter_type=self.adapter_type,
            d_model=d_model,
            num_patterns=self.num_patterns,
            use_multiscale=self.use_multiscale,
            conv_kernels=self.conv_kernels,
            identity_init=self.identity_init,
            discrete_topk=self.discrete_topk,
        )
