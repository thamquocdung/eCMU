from .conformer import ConformerBlock
import torch
import torch.nn as nn


class eCMU(nn.Module):
    def __init__(
        self, 
        nb_sources=1,
        n_fft=4096, 
        hidden_channels=512, 
        max_bins=1487, 
        nb_channels=2, 
        nb_layers=3, 
    ):
        """Initialize eCMU module.
        Args:
            nb_sources (int): Number of target sources
            n_fft (int): Windown size
            hidden_channels (int): Hidden feature size
            max_bins (int): Max number of frequency bands
            nb_channels (int): Number channels of audio
            nb_layers (int): Number of conformer blocks
        """
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
        self.nb_sources = nb_sources

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
        

    def forward(self, mix_spec: torch.Tensor, is_training: bool=True):
        """Forward
        Args:
            mix_spec (Tensor): Magnitude Mixture Spectrogram (B, 2, F, T)
            is_training (bool): training or inference?
        Returns:
            mask (Tensor): A Estimated mask for the target source (B, 1, 2, F, T) if is training
            Y_hat (Tensor): Estimated spectrogram of the target source (B, 1, 2, F, T) if is inference
        """
        X = mix_spec.detach().clone()
        batch, channels, bins, frames = mix_spec.shape
        mix_spec = mix_spec[..., :self.max_bins, :]

        x = (
            mix_spec.unsqueeze(1) + self.input_means.view(self.nb_sources, 1, -1, 1)
        ) * self.input_scale.view(self.nb_sources, 1, -1, 1)

        x = x.reshape(batch, -1, frames)
        x = self.affine1(x).view(batch, self.nb_sources, -1, frames).mean(1)


        x = x.transpose(2, 1) # [b, t, h]
        attn_out = self.conformer(x)
        x = torch.cat([x, attn_out], 2).transpose(2,1) # [b, h, t]
        

        mask = self.affine2(x).view(
            batch, self.nb_sources, channels, bins, frames
        ) * self.output_scale.view(self.nb_sources, 1, -1, 1) + self.output_means.view(self.nb_sources, 1, -1, 1)

        if is_training:
            return mask.relu()
        else:
            return mask.relu()*X.unsqueeze(1)
            


if __name__ == "__main__":
    x = torch.rand(4,2,2049,255)
    net = eCMU(hidden_channels=256, max_bins=1487)
    print(sum(p.numel() for p in net.parameters()
              if p.requires_grad))
    y = net(x)
    print(y.shape)