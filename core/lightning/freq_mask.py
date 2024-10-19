import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import numpy as np

from core.data.augment_cuda import CudaBase
from ..loss.time import SDR
from ..loss.freq import FLoss
from ..utils import MWF, padding, SOURCES


class MaskPredictor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: FLoss,
        transforms: List[CudaBase] = None,
        targets: Dict[str, None] = {},
        n_fft: int = 4096,
        hop_length: int = 1024,
        dim_f: int = 2048,
        **mwf_kwargs,
    ):
        """Initialize MaskPredictor training pipeline.
        Args:
            model: Our main model (eCMU)
            criterion: Loss function
            transforms: List of audio agumentation functions
            targets: List of target stems ([vocals])
            n_fft: Window length
            hop_length: Hop length
            dim_f: Feature size of frequency-axis
            **mwf_kwargs: Parameter dict for Multi-channel Wiener Filter (MWF)
        """
        super().__init__()

        self.model = model

        self.criterion = criterion
        self.targets = targets
        self.nb_tagets = len(targets)
        self.sdr = SDR()
        self.mwf = MWF(**mwf_kwargs)
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        if transforms is None:
            transforms = []
 
        self.transforms = nn.Sequential(*transforms)
        self.sources = SOURCES

        self.register_buffer(
            "targets_idx",
            torch.tensor(sorted([self.sources.index(target) for target in targets])),
        )
    

    def training_step(self, batch, batch_idx):
        """Training Step
        Args:
            batch (Tuple[Tensor, Tensor]): batch input
                - x: mixture waveform (B, 2, L)
                - y: target waveforms (B, 1, 2, L), num_targets = 1
        """
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx] 
        
        X = self.spec(x)
        Y = self.spec(y)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)

        Y_hat = self.mwf(pred_mask, X)[:,0,...].unsqueeze(1)
        y_hat = self.inv_spec(Y_hat)
        loss, values = self.criterion(pred_wave=y_hat,
                                      target_wave=y,
                                      pred_spec=Y_hat, 
                                      target_spec=Y, 
                                      mixture_wave=x)

        # values["loss"] = loss
        # self.log_dict(values, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx].cpu()

        y_hat = self.separate(x)
        

        loss, values = 0, {}

        batch = y_hat.shape[0]
        sdrs = (
            self.sdr(y_hat.view(-1, *y_hat.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(batch, -1)
            .mean(0)
        )
        values["avg_sdr"] = sdrs.mean().item()
        

        return loss, values

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_values = {}

        for k in outputs[0][1].keys():
            if 'sdr' in k:
                avg_values[k] = np.mean([x[1][k] for x in outputs])
            else:
                avg_values[k] =  sum(x[1][k] for x in outputs) / len(outputs)
        
        # self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log_dict(avg_values, prog_bar=False, sync_dist=True)

    def separate(self, x, overlap=0.25, segment=18, sampling_rate=44100):
        """Apply separate function for the long input (i.e: a whole song sample)
           using overlap-add algorithm
        Args:
            x (Tensor): Mixture waveform (B, 2, L) 
            overlap (Float[0,1]): Overlap ratio
            segment: Chunk size
        Returns:
            y_hat (Tensor): Estimated target waveform (B, nb_sources, 2, L)
        """
        segment *= sampling_rate 
        stride = int((1-overlap) * segment)
        
        nb_sources = self.nb_tagets
        batch, channels, length = x.shape
        offsets = range(0, length, stride)

        y_hat = torch.zeros(batch, nb_sources, channels, length)
        sum_weight = torch.zeros(length)
        weight = torch.cat([torch.arange(1, segment // 2 + 1),
                        torch.arange(segment - segment // 2, 0, -1)])
        weight = (weight / weight.max())
        assert len(weight) == segment

        for offset in offsets:
            chunk = x[..., offset:offset+segment]
            left_pad, right_pad, chunk_pad = padding(chunk, length=segment)
            chunk_out =  self.forward(chunk_pad)

            if left_pad > 0:
                chunk_out = chunk_out[...,left_pad:]
            if right_pad > 0:
                chunk_out = chunk_out[...,:-right_pad]

            chunk_out = chunk_out.cpu().detach()
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            y_hat[..., offset:offset + segment] += (w * chunk_out)
            sum_weight[offset:offset + segment] += w #.to(mix.device)
            offset += segment

        assert sum_weight.min() > 0
        y_hat /= sum_weight

        return y_hat
    
    def forward(self, x):
        """Estimate target waveforms from chunk-level mixture
        Args:
            x (Tensor): 2-channels mixture waveform (B, 2, L) 
        """
        X = self.spec(x)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        Y = self.mwf(pred_mask, X)[:,0,...].unsqueeze(1)
        pred_wave = self.inv_spec(Y, length=x.shape[-1])

        return pred_wave
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx] #.squeeze(1)

        y_hat = self.seperate(x)
        y = y.cpu()

        values = {}
        batch = y_hat.shape[0]
        sdrs = (
            self.sdr(y_hat.view(-1, *y_hat.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(batch, -1)
            .mean(0)
        )
        values["test_sdr"] = sdrs.mean().item()

        return 0, values
    
    def test_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_values = {}

        for k in outputs[0][1].keys():
            if 'sdr' in k:
                avg_values[k] = np.mean([x[1][k] for x in outputs])
        
        print(avg_values)
    
