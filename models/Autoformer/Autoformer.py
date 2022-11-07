import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .Autoformer_EncDec import (Decoder, DecoderLayer, Encoder, EncoderLayer,
                                my_Layernorm, series_decomp)
from .Embed import DataEmbedding, DataEmbedding_wo_pos


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, cfg):
        super(Autoformer, self).__init__()
        self.label_len = cfg["model"]["label_len"]
        self.pred_len = cfg["model"]["pred_len"]
        self.output_attention = cfg["model"]["output_attention"]
        self.use_amp = cfg["model"]["use_amp"]
        self.device = cfg["exp"]["device"]

        # Decomp
        kernel_size = cfg["model"]["moving_avg"]
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            cfg["model"]["enc_in"],
            cfg["model"]["d_model"],
            cfg["model"]["embed"],
            cfg["model"]["freq"],
            cfg["model"]["dropout"],
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            cfg["model"]["dec_in"],
            cfg["model"]["d_model"],
            cfg["model"]["embed"],
            cfg["model"]["freq"],
            cfg["model"]["dropout"],
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            cfg["model"]["factor"],
                            attention_dropout=cfg["model"]["dropout"],
                            output_attention=self.output_attention,
                        ),
                        cfg["model"]["d_model"],
                        cfg["model"]["n_heads"],
                    ),
                    cfg["model"]["d_model"],
                    cfg["model"]["d_ff"],
                    moving_avg=cfg["model"]["moving_avg"],
                    dropout=cfg["model"]["dropout"],
                    activation=cfg["model"]["activation"],
                )
                for l in range(cfg["model"]["e_layers"])
            ],
            norm_layer=my_Layernorm(cfg["model"]["d_model"]),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            cfg["model"]["factor"],
                            attention_dropout=cfg["model"]["dropout"],
                            output_attention=False,
                        ),
                        cfg["model"]["d_model"],
                        cfg["model"]["n_heads"],
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            cfg["model"]["factor"],
                            attention_dropout=cfg["model"]["dropout"],
                            output_attention=False,
                        ),
                        cfg["model"]["d_model"],
                        cfg["model"]["n_heads"],
                    ),
                    cfg["model"]["d_model"],
                    cfg["model"]["c_out"],
                    cfg["model"]["d_ff"],
                    moving_avg=cfg["model"]["moving_avg"],
                    dropout=cfg["model"]["dropout"],
                    activation=cfg["model"]["activation"],
                )
                for l in range(cfg["model"]["d_layers"])
            ],
            norm_layer=my_Layernorm(cfg["model"]["d_model"]),
            projection=nn.Linear(
                cfg["model"]["d_model"], cfg["model"]["c_out"], bias=True
            ),
        )

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # (self, x_enc, x_mark_enc, x_dec, x_mark_dec,
        #             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        # decoder input
        enc_self_mask = None
        dec_self_mask = None
        dec_enc_mask = None
        x_enc = batch_x
        x_mark_enc = batch_x_mark
        x_mark_dec = batch_y_mark
        dec_inp = torch.zeros_like(batch_x[:, -self.pred_len:, :]).float()
        dec_inp = (
            torch.cat([batch_x[:, -self.label_len:, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )
        x_dec = dec_inp

        # decomp init
        mean = torch.mean(
            x_enc, dim=1).unsqueeze(1).repeat(
            1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )
        # final
        dec_out = trend_part + seasonal_part

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out[:, -self.pred_len:, :]
