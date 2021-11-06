# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch as th
from torch import Tensor, nn

from .utils import capture_init, center_trim
from torch.nn.modules.conv import Conv1d, ConvTranspose1d
from typing import Union


class BLSTM(nn.Module):
    def __init__(self, dim: int, layers: int=1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.lstm.flatten_parameters()
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv: Union[Conv1d, ConvTranspose1d], reference: float) -> None:
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module: th.nn.Module, reference: float) -> None:
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def upsample(x, stride: int):
    """
    Linear upsampling, the output will be `stride` times longer.
    """
    batch, channels, time = x.size()
    weight = th.arange(stride, device=x.device, dtype=th.float) / stride
    x = x.view(batch, channels, time, 1)
    out = x[..., :-1, :] * (1 - weight) + x[..., 1:, :] * weight
    return out.reshape(batch, channels, -1)


def downsample(x, stride: int):
    """
    Downsample x by decimation.
    """
    return x[:, :, ::stride]


class Demucs(th.jit.ScriptModule):
    __constants__ = ["stride"]

    @capture_init
    def __init__(self,
                 sources: int=4,
                 audio_channels: int=2,
                 channels: int=64,
                 depth: int=6,
                 rewrite: bool=True,
                 glu: bool=True,
                 upsample: bool=False,
                 rescale: float=0.1,
                 kernel_size: int=8,
                 stride: int=4,
                 growth: float=2.,
                 lstm_layers: int=2,
                 context: int=3) -> None:
        """
        Args:
            sources (int): number of sources to separate
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            upsample (bool): use linear upsampling with convolutions
                Wave-U-Net style, instead of transposed convolutions
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.upsample = upsample
        self.channels = channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.final = None
        if upsample:
            self.final = nn.Conv1d(channels + audio_channels, sources * audio_channels, 1)
            stride = 1

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        in_channels = audio_channels
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                if upsample:
                    out_channels = channels
                else:
                    out_channels = sources * audio_channels
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            if upsample:
                decode += [
                    nn.Conv1d(channels, out_channels, kernel_size, stride=1),
                ]
            else:
                decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        for _ in range(self.depth):
            if self.upsample:
                length = math.ceil(length / self.stride) + self.kernel_size - 1
            else:
                length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            if self.upsample:
                length = length * self.stride + self.kernel_size - 1
            else:
                length = (length - 1) * self.stride + self.kernel_size

        return int(length)

    def forward(self, mix: Tensor) -> Tensor:
        x = mix
        saved = [x]
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            if self.upsample:
                x = downsample(x, self.stride)
        if self.lstm is not None:
            x = self.lstm(x)
        for decode in self.decoder:
            if self.upsample:
                x = upsample(x, stride=self.stride)
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)
        if self.final is not None:
            skip = center_trim(saved.pop(-1), x)
            x = th.cat([x, skip], dim=1)
            x = self.final(x)

        x = x.view(x.size(0), self.sources, self.audio_channels, x.size(-1))
        return x
