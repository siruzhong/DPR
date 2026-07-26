from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig, DPRConfig


@dataclass
class OLinearConfig(BasicTSModelConfig):
    """Configuration for the NeurIPS 2025 OLinear forecaster."""

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Forecast horizon."})
    num_features: int = field(default=1, metadata={"help": "Number of variables."})
    embed_size: int = field(default=16, metadata={"help": "Official scalar dimension-extension size."})
    hidden_size: int = field(default=512, metadata={"help": "Transformed-domain model width."})
    intermediate_size: int = field(default=512, metadata={"help": "Feed-forward width."})
    num_layers: int = field(default=2, metadata={"help": "Number of NormLin encoder blocks."})
    q_mat_path: str = field(default=None, metadata={"help": "NPZ containing training-only Q_in and Q_out matrices."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout in the transformed-domain learner."})
    use_revin: bool = field(default=True, metadata={"help": "Use Reversible Instance Normalization."})
    dpr: DPRConfig = field(
        default_factory=DPRConfig,
        metadata={"help": "Optional DPR on the transformed forecast representation."},
    )
