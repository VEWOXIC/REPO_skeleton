import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embed import DataEmbedding, DataEmbedding_wo_pos
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .FourierCorrelation import FourierBlock, FourierCrossAttention
from .MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .SelfAttention_Family import FullAttention, ProbAttention
from .Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np

class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.version = cfg['model']["version"]
        self.mode_select = cfg['model']["mode_select"]
        self.modes = cfg['model']["modes"]
        
        self.seq_len = cfg['data']["lookback"]

        self.label_len = cfg['model']["label_len"]
        self.pred_len = cfg['model']["pred_len"]
        self.output_attention = cfg['model']["output_attention"]
        self.use_amp = cfg['model']['use_amp']
        self.device = cfg['exp']['device']

        # Decomp
        kernel_size = cfg['model']["moving_avg"]
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(cfg['model']["enc_in"], cfg['model']["d_model"], cfg['model']["embed"],
                                                cfg['model']["freq"], cfg['model']["dropout"])
        self.dec_embedding = DataEmbedding_wo_pos(cfg['model']["dec_in"], cfg['model']["d_model"], cfg['model']["embed"],
                                                cfg['model']["freq"], cfg['model']["dropout"])

        if cfg['model']["version"] == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=cfg['model']["d_model"], L=cfg['model']["L"], base=cfg['model']["base"])
            decoder_self_att = MultiWaveletTransform(ich=cfg['model']["d_model"], L=cfg['model']["L"], base=cfg['model']["base"])
            decoder_cross_att = MultiWaveletCross(in_channels=cfg['model']["d_model"],
                                                  out_channels=cfg['model']["d_model"],
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=cfg['model']["modes"],
                                                  ich=cfg['model']["d_model"],
                                                  base=cfg['model']["base"],
                                                  activation=cfg['model']["cross_activation"])
        else:
            encoder_self_att = FourierBlock(in_channels=cfg['model']["d_model"],
                                            out_channels=cfg['model']["d_model"],
                                            seq_len=self.seq_len,
                                            modes=cfg['model']["modes"],
                                            mode_select_method=cfg['model']["mode_select"])
            decoder_self_att = FourierBlock(in_channels=cfg['model']["d_model"],
                                            out_channels=cfg['model']["d_model"],
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=cfg['model']["modes"],
                                            mode_select_method=cfg['model']["mode_select"])
            decoder_cross_att = FourierCrossAttention(in_channels=cfg['model']["d_model"],
                                                      out_channels=cfg['model']["d_model"],
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=cfg['model']["modes"],
                                                      mode_select_method=cfg['model']["mode_select"])
        # Encoder
        enc_modes = int(min(cfg['model']["modes"], cfg['model']["seq_len"]//2))
        dec_modes = int(min(cfg['model']["modes"], (cfg['model']["seq_len"]//2+cfg['model']["pred_len"])//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        cfg['model']["d_model"], cfg['model']["n_heads"]),

                    cfg['model']["d_model"],
                    cfg['model']["d_ff"],
                    moving_avg=cfg['model']["moving_avg"],
                    dropout=cfg['model']["dropout"],
                    activation=cfg['model']["activation"]
                ) for l in range(cfg['model']["e_layers"])
            ],
            norm_layer=my_Layernorm(cfg['model']["d_model"])
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        cfg['model']["d_model"], cfg['model']["n_heads"]),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        cfg['model']["d_model"], cfg['model']["n_heads"]),
                    cfg['model']["d_model"],
                    cfg['model']["c_out"],
                    cfg['model']["d_ff"],
                    moving_avg=cfg['model']["moving_avg"],
                    dropout=cfg['model']["dropout"],
                    activation=cfg['model']["activation"],
                )
                for l in range(cfg['model']["d_layers"])
            ],
            norm_layer=my_Layernorm(cfg['model']["d_model"]),
            projection=nn.Linear(cfg['model']["d_model"], cfg['model']["c_out"], bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return dec_out[:, -self.pred_len:, :]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)
