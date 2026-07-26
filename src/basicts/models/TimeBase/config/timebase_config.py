from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig, DPRConfig


@dataclass
class TimeBaseConfig(BasicTSModelConfig):
    """Configuration for the ICML 2025 TimeBase forecaster."""

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Forecast horizon."})
    num_features: int = field(default=1, metadata={"help": "Number of variables."})
    period_len: int = field(default=24, metadata={"help": "Length of each temporal segment."})
    basis_num: int = field(default=6, metadata={"help": "Number of temporal basis components."})
    use_period_norm: bool = field(default=True, metadata={"help": "Normalize each within-period position."})
    use_orthogonal: bool = field(default=True, metadata={"help": "Apply the temporal-basis orthogonal loss."})
    orthogonal_weight: float = field(default=0.08, metadata={"help": "Weight of the orthogonal loss."})
    individual: bool = field(default=False, metadata={"help": "Use separate mappings for each variable."})
    dpr: DPRConfig = field(
        default_factory=DPRConfig,
        metadata={"help": "Optional DPR on temporal basis positions."},
    )
