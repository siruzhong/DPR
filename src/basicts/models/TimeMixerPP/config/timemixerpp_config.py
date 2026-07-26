from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig, DPRConfig


@dataclass
class TimeMixerPPConfig(BasicTSModelConfig):
    """Configuration for TimeMixer++ forecasting."""

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Forecast horizon."})
    num_features: int = field(default=1, metadata={"help": "Number of variables."})
    num_layers: int = field(default=2, metadata={"help": "Number of mixer blocks."})
    hidden_size: int = field(default=32, metadata={"help": "Embedding width."})
    intermediate_size: int = field(default=32, metadata={"help": "Mixer intermediate width."})
    num_heads: int = field(default=1, metadata={"help": "Channel-mixing attention heads."})
    top_k: int = field(default=5, metadata={"help": "Dominant periods used by temporal imaging."})
    num_kernels: int = field(default=3, metadata={"help": "Inception kernels per block."})
    down_sampling_window: int = field(default=2, metadata={"help": "Multi-resolution downsampling factor."})
    down_sampling_layers: int = field(default=1, metadata={"help": "Number of downsampled resolutions."})
    channel_mixing: bool = field(default=True, metadata={"help": "Enable coarse-scale channel mixing."})
    channel_independence: bool = field(default=True, metadata={"help": "Use channel-independent temporal encoding."})
    use_revin: bool = field(default=True, metadata={"help": "Use RevIN in the model."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability."})
    term: str = field(default="long", metadata={"help": "Forecasting term: long or short."})
    dpr: DPRConfig = field(
        default_factory=DPRConfig,
        metadata={"help": "Optional DPR on encoder temporal hidden states."},
    )
