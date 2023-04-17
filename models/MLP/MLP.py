# File modified from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py

from fastai.layers import *
from .imports import *
from .layers import *


class MLP(Module):
    def __init__(self, cfg, act=nn.ReLU(inplace=True), y_range=None):
        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        seq_len = cfg["model"]["seq_len"]
        layers = cfg["model"]["layers"]
        ps = cfg["model"]["ps"]
        use_bn = cfg["model"]["use_bn"]
        bn_final = cfg["model"]["bn_final"]
        lin_first = cfg["model"]["lin_first"]
        fc_dropout = cfg["model"]["fc_dropout"]
        self.c_out = c_out
        self.seq_out = seq_len
        self.bs = cfg["model"]["batchsize"]

        layers, ps = L(layers), L(ps)
        if len(ps) <= 1:
            ps = ps * len(layers)
        assert len(layers) == len(ps), "#layers and #ps must match"
        self.flatten = Reshape(-1)
        nf = [c_in * seq_len] + layers
        self.mlp = nn.ModuleList()
        for i in range(len(layers)):
            self.mlp.append(
                LinBnDrop(
                    nf[i],
                    nf[i + 1],
                    bn=use_bn,
                    p=ps[i],
                    act=get_act_fn(act),
                    lin_first=lin_first,
                )
            )
        _head = [LinBnDrop(nf[-1], c_out * seq_len, bn=bn_final, p=fc_dropout)]
        if y_range is not None:
            _head.append(SigmoidRange(*y_range))
        self.head = nn.Sequential(*_head)

    def forward(self, x):
        x = self.flatten(x)
        for mlp in self.mlp:
            x = mlp(x)
        x = self.head(x)
        x = x.reshape(self.bs, self.c_out, self.seq_out)
        return x
