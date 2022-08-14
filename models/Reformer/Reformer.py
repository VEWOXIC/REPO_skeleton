import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .SelfAttention_Family import ReformerLayer
from .Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
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
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, cfg['model']["d_model"], cfg['model']["n_heads"], bucket_size=cfg['model']["bucket_size"],
                                  n_hashes=cfg['model']["n_hashes"]),
                    cfg['model']["d_model"],
                    cfg['model']["d_ff"],
                    dropout=cfg['model']["dropout"],
                    activation=cfg['model']["activation"]
                ) for l in range(cfg['model']['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(cfg['model']["d_model"])
        )
        self.projection = nn.Linear(cfg['model']["d_model"], cfg['model']["c_out"], bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        # if self.output_attention:
        #     return enc_out[:, -self.pred_len:, :], attns
        # else:
        #     return enc_out[:, -self.pred_len:, :]  # [B, L, D]
        return enc_out[:, -self.pred_len:, :]
