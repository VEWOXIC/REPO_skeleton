import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from .Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.label_len = cfg['model']["label_len"]
        self.pred_len = cfg['model']["pred_len"]
        self.output_attention = cfg['model']["output_attention"]
        self.use_amp = cfg['model']['use_amp']
        self.device = cfg['exp']['device']

        # Embedding
        self.enc_embedding = DataEmbedding(cfg['model']["enc_in"], cfg['model']["d_model"], cfg['model']["embed"],
                                                cfg['model']["freq"], cfg['model']["dropout"])
        self.dec_embedding = DataEmbedding(cfg['model']["dec_in"], cfg['model']["d_model"], cfg['model']["embed"],
                                                cfg['model']["freq"], cfg['model']["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, cfg['model']["factor"], attention_dropout=cfg['model']["dropout"],
                                      output_attention=self.output_attention),
                        cfg['model']["d_model"], cfg['model']["n_heads"]),
                    cfg['model']["d_model"],
                    cfg['model']["d_ff"],
                    dropout=cfg['model']["dropout"],
                    activation=cfg['model']["activation"]
                ) for l in range(cfg['model']['e_layers'])
            ],
            [
                ConvLayer(
                    cfg['model']["d_model"]
                ) for l in range(cfg['model']['e_layers'] - 1)
            ] if cfg['model']['distil'] else None,
            norm_layer=torch.nn.LayerNorm(cfg['model']["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, cfg['model']["factor"], attention_dropout=cfg['model']["dropout"], output_attention=False),
                        cfg['model']["d_model"], cfg['model']["n_heads"]),
                    AttentionLayer(
                        ProbAttention(False, cfg['model']["factor"], attention_dropout=cfg['model']["dropout"], output_attention=False),
                        cfg['model']["d_model"], cfg['model']["n_heads"]),
                    cfg['model']["d_model"],
                    cfg['model']["d_ff"],
                    dropout=cfg['model']["dropout"],
                    activation=cfg['model']["activation"],
                )
                for l in range(cfg['model']["d_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(cfg['model']["d_model"]),
            projection=nn.Linear(cfg['model']["d_model"], cfg['model']["c_out"], bias=True)
        )

    def forward(self, batch_x, batch_x_mark, batch_y_mark):
    # (self, x_enc, x_mark_enc, x_dec, x_mark_dec,
    #             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        # decoder input
        enc_self_mask=None
        dec_self_mask=None
        dec_enc_mask=None
        x_enc = batch_x # label_len
        x_mark_enc = batch_x_mark
        x_mark_dec = batch_y_mark
        dec_inp = torch.zeros_like(batch_x[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp], dim=1).float().to(self.device)
        x_dec = dec_inp # label_len + pred_len

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out[:, -self.pred_len:, :]
