# File modified from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py

from .imports import *
from .layers import *


# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:

# Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., ... & Petitjean, F. (2019).
# InceptionTime: Finding AlexNet for Time Series Classification. arXiv preprint arXiv:1909.04939.
# Official InceptionTime tensorflow implementation: https://github.com/hfawaz/InceptionTime


class InceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList(
            [Conv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks]
        )
        self.maxconvpool = nn.Sequential(
            *[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)]
        )
        self.concat = Concat()
        self.bn = BN1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(self.bn(x))


@delegates(InceptionModule.__init__)
class InceptionBlock(Module):
    def __init__(self, ni, nf=32, residual=True, depth=6, **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(
                InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs)
            )
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(
                    BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None)
                )
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
                res = x = self.act(self.add(x, self.shortcut[d // 3](res)))
        return x


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(self, cfg, nf=8, nb_filters=None, **kwargs):
        # def __init__(self, c_in, c_out, seq_len=None, nf=32, nb_filters=None, **kwargs):
        self.num_layer = cfg["model"]["num_layers"]
        self.seq_len = cfg["data"]["lookback"]
        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        nf = ifnone(nf, nb_filters)  # for compatibility
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * 4, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)  # ;print(x.size())
        # x = self.gap(x);print(x.size())
        x = x.transpose(1, 2)  # ;print(x.size())
        x = self.fc(x)  # ;print(x.size())
        x = x.transpose(1, 2)  # ;print(x.size())
        return x
