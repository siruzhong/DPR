"""BasicTS wrapper for TimeMixer++.

The backbone is provided by PyPOTS' BSD-3-Clause implementation, which follows
the authors' released TimeMixer++ architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from basicts.modules.dpr import dpr_orthogonal_loss

from ..config.timemixerpp_config import TimeMixerPPConfig


class TimeMixerPPForForecasting(nn.Module):
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        try:
            from pypots.nn.modules.timemixerpp import BackboneTimeMixerPP
        except ImportError as exc:
            raise ImportError(
                "TimeMixer++ requires pypots==1.5; install the repository requirements in the BasicTS environment"
            ) from exc

        self.backbone = BackboneTimeMixerPP(
            task_name=f"{config.term}_term_forecast",
            n_steps=config.input_len,
            n_features=config.num_features,
            n_pred_steps=config.output_len,
            n_pred_features=config.num_features,
            n_layers=config.num_layers,
            d_model=config.hidden_size,
            d_ffn=config.intermediate_size,
            n_heads=config.num_heads,
            dropout=config.dropout,
            top_k=config.top_k,
            n_kernels=config.num_kernels,
            channel_mixing=config.channel_mixing,
            channel_independence=config.channel_independence,
            downsampling_layers=config.down_sampling_layers,
            downsampling_window=config.down_sampling_window,
            downsampling_method="avg",
            use_future_temporal_feature=False,
            use_norm=config.use_revin,
        )
        self.dpr_cfg = config.dpr
        self.dpr = config.dpr.build_module(config.hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Mirror the PyPOTS forecast path, exposing its encoder states to DPR."""
        backbone = self.backbone
        b, _, n_features = inputs.size()
        x_enc, x_mark_enc = backbone._BackboneTimeMixerPP__multi_scale_process_inputs(inputs, None)

        x_list = []
        for x in x_enc:
            _, t, n = x.size()
            x = backbone.revin_layers[len(x_list)](x, mode="norm") if backbone.use_norm else x
            if backbone.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(b * n, t, 1)
            x_list.append(x)

        if backbone.channel_mixing and backbone.channel_independence == 1:
            _, t, d = x_list[-1].size()
            coarse = x_list[-1].reshape(b, n_features, t * d)
            coarse, _ = backbone.channel_mixing_attention(coarse, coarse, coarse, None)
            x_list[-1] = coarse.reshape(b * n_features, t, d) + x_list[-1]

        enc_out_list = [backbone.enc_embedding(x, None) for x in x_list]
        for i in range(backbone.n_layers):
            enc_out_list = backbone.encoder_model[i](enc_out_list)
        if self.dpr is not None:
            enc_out_list = [self.dpr(enc_out) for enc_out in enc_out_list]

        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            dec_out = backbone.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = backbone.projection_layer(dec_out)
            dec_out = dec_out.reshape(b, n_features, -1).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)
        # Keep the released PyPOTS behavior: the final-scale prediction is
        # passed to RevIN (the backbone does not sum the scale outputs here).
        dec_out = dec_out_list[-1]
        dec_out = backbone.revin_layers[0](dec_out, mode="denorm") if backbone.use_norm else dec_out
        if self.dpr is not None and self.dpr_cfg.orth_lambda > 0:
            return {
                "prediction": dec_out,
                "dpr_orth": self.dpr_cfg.orth_lambda * dpr_orthogonal_loss(self.dpr.mode_table),
            }
        return dec_out
