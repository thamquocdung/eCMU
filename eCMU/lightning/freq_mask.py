import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict
from torchaudio.transforms import Spectrogram, InverseSpectrogram
import numpy as np

from ..loss.time import SDR
from ..loss.freq import FLoss
from ..augment import CudaBase

from ..utils import MWF, MDX_SOURCES, SDX_SOURCES, SE_SOURCES
import museval
from filtering import wiener
from utils import on_load_checkpoint

def compute_score(estimates, references):
    win = 44100
    hop = 44100
    references = references.squeeze(0).transpose(1, 2).double()
    estimates = estimates.squeeze(0).transpose(1, 2).double()
    references = references.cpu().numpy()
    estimates = estimates.cpu().numpy()

    sdr,_,_,_ = museval.metrics.bss_eval(
        references, estimates,
        compute_permutation=False,
        window=win,
        hop=hop,
        framewise_filters=False,
        bsseval_sources_version=False)[:-1]
    print(sdr.shape, sdr)
    return sdr[0]
    
def padding(x, length):
    offsets = length - x.shape[-1]
    left_pad = offsets // 2
    right_pad = offsets - left_pad

    return left_pad, right_pad, torch.nn.functional.pad(x, (left_pad, right_pad))

def on_load_checkpoint1(model, checkpoint):
    state_dict = checkpoint.copy()
    model_state_dict = model.state_dict()

    is_changed = False
    for k in checkpoint.keys():
        # print(k.split("model.")[0])
        o = k.split("model.")[-1]
        if o in model_state_dict:

            if state_dict[k].shape != model_state_dict[o].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[o].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[o]
                is_changed = True
        else:
            print(f"Dropping parameter {k}")
            state_dict.pop(k)
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)

    return state_dict

class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)        
        self.dim_f = dim_f
    
    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)
        x = x.reshape([*batch_dims,c,x.shape[-2],-1])

        return x[...,:self.dim_f,:]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c,f,t = x.shape[-3:]
        n = self.n_fft//2+1
        f_pad = torch.zeros([*batch_dims,c,n-f,t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims,-1,2,n,t]).reshape([-1,n,t]) # [B,F,T]
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims,2,-1])

        return x
    

class MaskPredictor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: FLoss,
        transforms: List[CudaBase] = None,
        target_track: str = None,
        targets: Dict[str, None] = {},
        n_fft: int = 4096,
        hop_length: int = 1024,
        dim_f: int = 2048,
        **mwf_kwargs,
    ):
        super().__init__()

        self.model = model

        
        cp = torch.load('checkpoints/vocals-512/epoch=435-avg_sdr=7.254.ckptpre')
        # cp = torch.load('/home/ec2-user/.cache/torch/hub/checkpoints/vocals-b62c91ce.pth')
        self.model.load_state_dict(on_load_checkpoint(self.model, cp), strict=False)
        # self.model.load_state_dict(cp, strict=False)


        self.criterion = criterion
        self.targets = targets
        self.sdr = SDR()
        self.mwf = MWF(**mwf_kwargs)
        # self.stft = STFT(n_fft=n_fft, hop_length=hop_length, dim_f=dim_f)
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        if transforms is None:
            transforms = []
        target_track = "mdx"
        self.transforms = nn.Sequential(*transforms)
        if target_track == "sdx":
            self.sources = SDX_SOURCES
        elif target_track == "mdx":
            self.sources = MDX_SOURCES
        elif target_track == "se":
            self.sources = SE_SOURCES
        else:
            raise ValueError(f"Invalid target track: {target_track}")
        self.register_buffer(
            "targets_idx",
            torch.tensor(sorted([self.sources.index(target) for target in targets])),
        )
        # self.validation_result = []

    def forward(self, x, length=None):
        X = self.spec(x)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        Y = self.mwf(pred_mask, X)[:,0,...].unsqueeze(1)
        pred = self.inv_spec(Y, length=length)
        return pred
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        if len(self.transforms) > 0:
            y = self.transforms(y)
            x = y.sum(1)
        y = y[:, self.targets_idx] # .squeeze(1)
        
        X = self.spec(x)
        Y = self.spec(y)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)

        # Y_hat = pred_mask * X_mag.unsqueeze(1)
        # loss = F.mse_loss(Y.abs(), Y_hat)

        Y_hat = self.mwf(pred_mask, X)[:,0,...].unsqueeze(1)
        pred = self.inv_spec(Y_hat)
        loss, values = self.criterion(pred, Y_hat, Y, X, y, x)

        # values["loss"] = loss
        # self.log_dict(values, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step1(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx]#.squeeze(1)

        X = self.spec(x)
        Y = self.spec(y)
        X_mag = X.abs()
        pred_mask = self.model(X_mag)
        
        
        Y_hat = self.mwf(pred_mask, X)[:,0,...].unsqueeze(1)
        pred = self.inv_spec(Y_hat)


        loss, values = self.criterion(pred, Y_hat, Y, X, y, x)
        # pred = self.inv_spec(self.mwf(pred_mask, X))[:,0,...].unsqueeze(1)

        batch = pred.shape[0]
        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(batch, -1)
            .mean(0)
        )
        values["avg_sdr"] = sdrs.mean().item()

        # for i, t in enumerate(self.targets_idx):
        #     values[f"{self.sources[t]}_sdr"] = sdrs[i].item()
        # values["avg_sdr"] = sdrs.mean().item()
        return loss, values

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, self.targets_idx].cpu() #.squeeze(1)

        pred = self.seperate(x)
        

        loss, values = 0, {}
        # loss, values = self.criterion(pred, Y_hat, Y, X, y, x)
        # pred = self.inv_spec(self.mwf(pred_mask, X))[:,0,...].unsqueeze(1)

        batch = pred.shape[0]
        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
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
        
        # print(avg_values)
        # self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log_dict(avg_values, prog_bar=False, sync_dist=True)

    def seperate(self, x):
        overlap = 0.25
        segment = 18*44100 # 127*2048 
        stride = int((1-overlap) * segment)
        samplerate = 441000
        nb_sources = 1 #len(self.sources)
        batch, channels, length = x.shape
        offsets = range(0, length, stride)

        num_batch = len(offsets) // 4 + 1

        out = torch.zeros(batch, nb_sources, channels, length)
        sum_weight = torch.zeros(length)
        weight = torch.cat([torch.arange(1, segment // 2 + 1),
                        torch.arange(segment - segment // 2, 0, -1)])
        weight = (weight / weight.max())
        # weight = torch.hann_window(segment)
        assert len(weight) == segment

        for offset in offsets:
            chunk = x[..., offset:offset+segment]
            left_pad, right_pad, chunk_pad = padding(chunk, length=segment)
            chunk_out =  self.forward(chunk_pad, length=chunk_pad.shape[-1])
            # chunk_out = self.seperate1(chunk_pad)[:, 0, ...]

            if left_pad > 0:
                chunk_out = chunk_out[...,left_pad:]
            if right_pad > 0:
                chunk_out = chunk_out[...,:-right_pad]

            chunk_out = chunk_out.cpu().detach()
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            out[..., offset:offset + segment] += (w * chunk_out)
            sum_weight[offset:offset + segment] += w #.to(mix.device)
            offset += segment

        assert sum_weight.min() > 0
        out /= sum_weight

        return out

    def seperate1(self, audio):
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """

        nb_sources = 1
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.spec(audio)
        X = mix_stft.abs()
        mix_stft = torch.view_as_real(mix_stft)

        # initializing spectrograms variable
        spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)

      
        target_spectrogram = self.model(X.detach().clone(), inference=True)
        spectrograms[..., 0] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

       
        nb_sources += 1
        niter = 1

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            wiener_win_len = 300

            while pos < nb_frames:
                cur_frame = torch.arange(pos, min(nb_frames, pos + wiener_win_len))
                pos = int(cur_frame[-1]) + 1

                targets_stft[sample, cur_frame] = wiener(
                    spectrograms[sample, cur_frame],
                    mix_stft[sample, cur_frame],
                    niter,
                    softmask=False,
                    residual=True,
                )


        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()
        targets_stft = torch.view_as_complex(targets_stft)

        # inverse STFT
        estimates = self.inv_spec(targets_stft, length=audio.shape[2])

        return estimates #.cpu().detach()


    def test_step(self, batch, batch_idx):
        x, y = batch

       
        y = y[:, self.targets_idx]#.squeeze(1)

        
        pred = self.seperate(x)
        y = y.cpu()

        values = {}
        batch = pred.shape[0]
        sdrs = (
            self.sdr(pred.view(-1, *pred.shape[-2:]), y.view(-1, *y.shape[-2:]))
            .view(batch, -1)
            .mean(0)
        )
        values["test_sdr"] = sdrs.mean().item()
        print(values["test_sdr"])
        # for i, t in enumerate(self.targets_idx):
        #     values[f"{self.sources[t]}_sdr"] = sdrs[i].item()
        # values["avg_sdr"] = sdrs.mean().item()
        return 0, values
    
    def test_epoch_end(self, outputs) -> None:
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_values = {}

        for k in outputs[0][1].keys():
            if 'sdr' in k:
                avg_values[k] = np.mean([x[1][k] for x in outputs])
        
        print(avg_values)
    
