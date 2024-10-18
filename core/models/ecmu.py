from .conformer import ConformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter

class eCMU(nn.Module):
    # __constants__ = ["max_bins"]

    # max_bins: int

    def __init__(
        self, n_fft=4096, hidden_channels=512, max_bins=1487, nb_channels=2, nb_layers=3, nb_heads=8
    ):
        super().__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bins:
            self.max_bins = max_bins
        else:
            self.max_bins = self.nb_output_bins
        self.hidden_channels = hidden_channels
        self.n_fft = n_fft
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.nb_sources = 1
        self.nb_heads = nb_heads

        self.input_means = nn.Parameter(torch.zeros(self.nb_sources * self.max_bins))
        self.input_scale = nn.Parameter(torch.ones(self.nb_sources * self.max_bins))

        self.output_means = nn.Parameter(torch.zeros(self.nb_sources * self.nb_output_bins))
        self.output_scale = nn.Parameter(torch.ones(self.nb_sources * self.nb_output_bins))

        self.affine1 = nn.Sequential(
            nn.Conv1d(
                nb_channels * self.max_bins * self.nb_sources,
                hidden_channels * self.nb_sources,
                1,
                bias=False,
                groups=self.nb_sources,
            ),
            nn.BatchNorm1d(hidden_channels * self.nb_sources),
            nn.Tanh(),
        )
        # self.lstm = nn.LSTM(
        #     input_size=self.hidden_channels,
        #     hidden_size=self.hidden_channels // 2,
        #     num_layers=nb_layers,
        #     dropout=0.4,
        #     bidirectional=True,
        # )
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_channels, nhead=8)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        conformer_layer = ConformerBlock(
            dim=hidden_channels,
            dim_head=hidden_channels // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.conformer = nn.Sequential(conformer_layer, conformer_layer)
       
        self.affine2 = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels * self.nb_sources, 1, bias=False),
            nn.BatchNorm1d(hidden_channels * self.nb_sources),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels * self.nb_sources,
                nb_channels * self.nb_output_bins * self.nb_sources,
                1,
                bias=False,
                groups=self.nb_sources,
            ),
            nn.BatchNorm1d(nb_channels * self.nb_output_bins * self.nb_sources),
        )
        

    def forward(self, spec: torch.Tensor, inference: bool=False):
        mix = spec.detach().clone()
        batch, channels, bins, frames = spec.shape
        spec = spec[..., :self.max_bins, :]

        x = (
            spec.unsqueeze(1) + self.input_means.view(self.nb_sources, 1, -1, 1)
        ) * self.input_scale.view(self.nb_sources, 1, -1, 1)

        x = x.reshape(batch, -1, frames)
        x = self.affine1(x).view(batch, self.nb_sources, -1, frames).mean(1)

        # x = x.permute(2, 0, 1) # [t, b, h]
        # attn_out = self.transformer(x)
        # x = torch.cat([x, attn_out], 2).permute(1, 2, 0) # [b, h, t]
        # print(x.shape, attn_out.shape)

        x = x.transpose(2, 1) # [b, t, h]
        attn_out = self.conformer(x)
        x = torch.cat([x, attn_out], 2).transpose(2,1) # [b, h, t]
        

        mask = self.affine2(x).view(
            batch, self.nb_sources, channels, bins, frames
        ) * self.output_scale.view(self.nb_sources, 1, -1, 1) + self.output_means.view(self.nb_sources, 1, -1, 1)
        if inference:
            return mask.relu()*mix.unsqueeze(1)
        else:
            return mask.relu()


if __name__ == "__main__":
    x = torch.rand(4,2,2049,255)
    net = eCMU(hidden_channels=256, max_bins=1487)
    print(sum(p.numel() for p in net.parameters()
              if p.requires_grad))
    y = net(x)
    print(y.shape)