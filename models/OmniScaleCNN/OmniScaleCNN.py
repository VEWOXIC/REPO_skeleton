# File modified from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py

from .imports import *
from .layers import *

# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@gmail.com based on:
# Rußwurm, M., & Körner, M. (2019). Self-attention for raw optical satellite time series classification. arXiv preprint arXiv:1910.10536.
# Official implementation: https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/OmniScaleCNN.py


class SampaddingConv1D_BN(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.padding = nn.ConstantPad1d(
            (int((kernel_size - 1) / 2), int(kernel_size / 2)), 0
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv1d(x)
        x = self.bn(x)
        return x


class build_layer_with_layer_parameter(Module):
    """
    formerly build_layer_with_layer_parameter
    """

    def __init__(self, layer_parameters):
        """
        layer_parameters format
            [in_channels, out_channels, kernel_size,
            in_channels, out_channels, kernel_size,
            ..., nlayers
            ]
        """
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, x):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(x)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class OmniScaleCNN(Module):
    def __init__(self, cfg):

        c_in = cfg["model"]["c_in"]
        c_out = cfg["model"]["c_out"]
        seq_len = cfg["model"]["seq_len"]
        few_shot = cfg["model"]["few_shot"]
        layers = cfg["model"]["layers"]

        receptive_field_shape = seq_len // 4
        layer_parameter_list = generate_layer_parameter_list(
            1, receptive_field_shape, layers, in_channel=c_in
        )
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)
        self.gap = GAP1d(1)
        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number = out_put_channel_number + final_layer_parameters[1]
        self.hidden = nn.Linear(out_put_channel_number, c_out)

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 1)
        # x = self.gap(x)
        if not self.few_shot:
            x = self.hidden(x)
        x = x.permute(0, 2, 1)
        return x


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = max(1, int(paramenter_layer / (in_channel * sum(prime_list))))
    return out_channel_expect


def generate_layer_parameter_list(start, end, layers, in_channel=1):
    prime_list = get_Prime_number_in_a_range(start, end)

    layer_parameter_list = []
    for paramenter_number_of_layer in layers:
        out_channel = get_out_channel_number(
            paramenter_number_of_layer, in_channel, prime_list
        )

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(
        layers[0], 1, prime_list
    )
    tuples_in_layer_last.append((in_channel, first_out_channel, 1))
    tuples_in_layer_last.append((in_channel, first_out_channel, 2))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list
