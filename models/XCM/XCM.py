# File modified from
# https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py

from .explainability import *
from .imports import *
from .layers import *

# This is an unofficial PyTorch implementation of XVM created by Ignacio
# Oguiza - timeseriesAU@gmail.com based on:

# Fauvel, K., Lin, T., Masson, V., Fromont, É., & Termier, A. (2020). XCM: An Explainable Convolutional Neural Network
# https://hal.inria.fr/hal-03469487/document
# Official tensorflow implementation available at: https://github.com/XAIseries/XCM
# No official XCM PyTorch implementation available as of Dec 11, 2021


class XCM(Module):
    def __init__(
        self,
        cfg,
        flatten: bool = False,
        custom_head: callable = None,
        concat_pool: bool = False,
        bn: bool = False,
        y_range: tuple = None,
        **kwargs
    ):
        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        seq_len = cfg["model"]["seq_len"]
        nf = cfg["model"]["nf"]
        window_perc = cfg["model"]["window_perc"]
        fc_dropout = cfg["model"]["fc_dropoutn"]
        self.c_out = c_out
        self.seq_out = seq_len
        self.bs = cfg["model"]["batchsize"]

        window_size = int(round(seq_len * window_perc, 0))
        self.conv2dblock = nn.Sequential(
            *[
                Unsqueeze(1),
                Conv2d(1, nf, kernel_size=(1, window_size), padding="same"),
                BatchNorm(nf),
                nn.ReLU(),
            ]
        )
        self.conv2d1x1block = nn.Sequential(
            *[nn.Conv2d(nf, 1, kernel_size=1), nn.ReLU(), Squeeze(1)]
        )
        self.conv1dblock = nn.Sequential(
            *[
                Conv1d(c_in, nf, kernel_size=window_size, padding="same"),
                BatchNorm(nf, ndim=1),
                nn.ReLU(),
            ]
        )
        self.conv1d1x1block = nn.Sequential(
            *[nn.Conv1d(nf, 1, kernel_size=1), nn.ReLU()]
        )
        self.concat = Concat()
        self.conv1d = nn.Sequential(
            *[
                Conv1d(c_in + 1, nf, kernel_size=window_size, padding="same"),
                BatchNorm(nf, ndim=1),
                nn.ReLU(),
            ]
        )

        self.head_nf = nf
        self.c_out = c_out
        self.seq_len = seq_len
        if custom_head:
            self.head = custom_head(self.head_nf, c_out, seq_len, **kwargs)
        else:
            self.head = self.create_head(
                self.head_nf,
                c_out,
                seq_len,
                flatten=flatten,
                concat_pool=concat_pool,
                fc_dropout=fc_dropout,
                bn=bn,
                y_range=y_range,
            )

    def forward(self, x):
        x1 = self.conv2dblock(x)
        x1 = self.conv2d1x1block(x1)
        x2 = self.conv1dblock(x)
        x2 = self.conv1d1x1block(x2)
        out = self.concat((x2, x1))
        out = self.conv1d(out)
        out = self.head(out)
        out = out.reshape(self.bs, self.c_out, self.seq_out)
        return out

    def create_head(
        self,
        nf,
        c_out,
        seq_len=None,
        flatten=False,
        concat_pool=False,
        fc_dropout=0.0,
        bn=False,
        y_range=None,
    ):
        if flatten:
            nf *= seq_len
            layers = [Flatten()]
        else:
            if concat_pool:
                nf *= 2
            layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        layers += [LinBnDrop(nf, c_out * seq_len, bn=bn, p=fc_dropout)]
        if y_range:
            layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def show_gradcam(
        self,
        x,
        y=None,
        detach=True,
        cpu=True,
        apply_relu=True,
        cmap="inferno",
        figsize=None,
        **kwargs
    ):
        att_maps = get_attribution_map(
            self,
            [self.conv2dblock, self.conv1dblock],
            x,
            y=y,
            detach=detach,
            cpu=cpu,
            apply_relu=apply_relu,
        )
        att_maps[0] = (att_maps[0] - att_maps[0].min()) / (
            att_maps[0].max() - att_maps[0].min()
        )
        att_maps[1] = (att_maps[1] - att_maps[1].min()) / (
            att_maps[1].max() - att_maps[1].min()
        )

        figsize = ifnone(figsize, (10, 10))
        fig = plt.figure(figsize=figsize, **kwargs)
        ax = plt.axes()
        plt.title("Observed variables")
        im = ax.imshow(att_maps[0], cmap=cmap)
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )
        plt.colorbar(im, cax=cax)
        plt.show()

        fig = plt.figure(figsize=figsize, **kwargs)
        ax = plt.axes()
        plt.title("Time")
        im = ax.imshow(att_maps[1], cmap=cmap)
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )
        plt.colorbar(im, cax=cax)
        plt.show()
