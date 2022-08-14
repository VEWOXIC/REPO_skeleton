import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding
import numpy as np

class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.pred_len = cfg['model']['pred_len']
        self.output_attention = self.pred_len = cfg['model']['output_attention']

        # Embedding
        self.enc_embedding = DataEmbedding(cfg['model']['enc_in'], cfg['model']['d_model'], cfg['model']['embed'], 
                                           cfg['model']['freq'], cfg['model']['dropout'])
        self.dec_embedding = DataEmbedding(cfg['model']['dec_in'], cfg['model']['d_model'], cfg['model']['embed'], 
                                           cfg['model']['freq'], cfg['model']['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, cfg['model']['factor'], attention_dropout=cfg['model']['dropout'],
                                      output_attention=cfg['model']['output_attention']), cfg['model']['d_model'], cfg['model']['n_heads']),
                    cfg['model']['d_model'],
                    cfg['model']['d_ff'],
                    dropout=cfg['model']['dropout'],
                    activation=cfg['model']['activation']
                ) for l in range(cfg['model']['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(cfg['model']['d_model'])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, cfg['model']['factor'], attention_dropout=cfg['model']['dropout'], output_attention=False),
                        cfg['model']['d_model'], cfg['model']['n_heads']),
                    AttentionLayer(
                        FullAttention(False, cfg['model']['factor'], attention_dropout=cfg['model']['dropout'], output_attention=False),
                        cfg['model']['d_model'], cfg['model']['n_heads']),
                    cfg['model']['d_model'],
                    cfg['model']['d_ff'],
                    dropout=cfg['model']['dropout'],
                    activation=cfg['model']['activation'],
                )
                for l in range(cfg['model']['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(cfg['model']['d_model']),
            projection=nn.Linear(cfg['model']['d_model'], cfg['model']['c_out'], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]