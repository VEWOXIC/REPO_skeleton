
from .layers import *
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

# Cell
class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel,self).__init__()
    # def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """

        self.c_in = cfg['model']['c_in']
        self.c_out = cfg['model']['c_out']
        self.d_model = cfg['model']['d_model']
        self.n_head = cfg['model']['n_head']
        self.d_ffn = cfg['model']['d_ffn']
        self.dropout = cfg['model']['dropout']
        self.activation = cfg['model']['activation']
        self.n_layers = cfg['model']['n_layers']

        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(self.c_in, self.d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(self.d_model, self.n_head, dim_feedforward=self.d_ffn, dropout=self.dropout, activation=self.activation)
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.n_layers, norm=encoder_norm)
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(self.d_model, self.c_out)

    def forward(self,x):
        x = self.permute(x)  # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = self.transpose(x) # seq_len x bs x d_model -> bs x seq_len x d_model
        # x = self.max(x);print(x.size())
        x = self.relu(x)
        x = self.outlinear(x)
        x = x.transpose(1,2)
        return x
