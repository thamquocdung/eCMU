import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ml_collections import ConfigDict
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from .ecmu import eCMU
from ..utils import MWF
import os
from pathlib import Path
import librosa

def padding(x, length):
    offsets = length - x.shape[-1]
    left_pad = offsets // 2
    right_pad = offsets - left_pad

    return left_pad, right_pad, torch.nn.functional.pad(x, (left_pad, right_pad))

class MusicSeparationModel:
    def __init__(self, ckpts, sources):     
        self.models = {}
        self.sources = sources
        self.target_sr = 44100
        self.residual = False
        self.mwf = MWF()
        self.spec = Spectrogram(n_fft=4096, hop_length=1024, power=None).cuda()
        self.inv_spec = InverseSpectrogram(n_fft=4096, hop_length=1024).cuda()

        for i, src in enumerate(sources):
            print(src)
            self.models[src] = Net(hidden_channels=256).cuda()
            checkpoint = torch.load(ckpts[i], map_location='cuda')

            try:
                print(f"{src} load from ema")
                self.models[src].load_state_dict(checkpoint['callbacks']['EMACallback']['state_dict_ema'], strict=False)
            except:
                if "state_dict" in checkpoint:
                    print("aaaa")
                    self.models[src].load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    print("bbbb")
                    self.models[src].load_state_dict(checkpoint, strict=False)

            self.models[src].eval()

    def seperate(self, audio_path, output_path="./outputs"):
        filename = os.path.basename(audio_path)[:-4]
        mix, sr = sf.read(
          audio_path,
          dtype='float32', fill_value=0.)
        mix = mix.T
        print(mix.shape, sr, self.target_sr)
        mix = librosa.resample(mix, orig_sr=sr, target_sr=self.target_sr)
        mix = torch.tensor(mix).unsqueeze(0).cuda()
        print(mix.shape)
        estimated_waveforms = self.inference(mix)[0]
        print(estimated_waveforms.shape)
        
        for i, source in enumerate(self.sources):
            Path(f'{output_path}/{filename}').mkdir(parents=True, exist_ok=True)
            sf.write(f'{output_path}/{filename}/{source}.mp3', estimated_waveforms[i].T.numpy(), 44100)


    def inference(self, mix):
        overlap = 0.25
        segment = 18*44100 #127*2048 
        stride = int((1-overlap) * segment)
        samplerate = self.target_sr
        nb_sources = len(self.sources)
        batch, channels, length = mix.shape
        length = mix.shape[-1]
        offsets = range(0, length, stride)
        num_batch = len(offsets) // 4 + 1
        if self.residual:
            # we add an additional target
            nb_sources += 1
        out = torch.zeros(batch, nb_sources, channels, length)
        sum_weight = torch.zeros(length)
        weight = torch.cat([torch.arange(1, segment // 2 + 1),
                        torch.arange(segment - segment // 2, 0, -1)])
        weight = (weight / weight.max())
        # weight = torch.hann_window(segment)
        assert len(weight) == segment

        for offset in offsets:
            chunk = mix[..., offset:offset+segment]
            left_pad, right_pad, chunk_pad = padding(chunk, length=segment)
            chunk_out = self._forward(chunk_pad)
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
    
    def _forward(self, x):
        X = self.spec(x)
        X_mag = X.abs()
        
        pred_masks = []
        for _, model in self.models.items():
            out = model(X_mag).squeeze()
            pred_masks.append(out)
        
        pred_masks = torch.stack(pred_masks, axis=0).unsqueeze(0)
        Y_hat = self.mwf(pred_masks, X)
        y_hat = self.inv_spec(Y_hat, length=x.shape[-1])

        return y_hat

if __name__ == "__main__":
    import soundfile as sf
    separator = MusicSeparationModel(sources=["vocals", "drums", "bass", "other"],
                                     ckpts=["eMUC_cp/vocals-ema/best.ckpt", "eMUC_cp/drums-ema/epoch=20-avg_sdr=7.195.ckptpre", "eMUC_cp/bass-ema/best_epoch=14-avg_sdr=5.483.ckptpre", "eMUC_cp/other-ema/best.ckptpre"])

    # separator.seperate("/home/ec2-user/musdb/test/Al James - Schoolboy Facination/mixture.wav")
    
    for filename in os.listdir("test_inputs/"):
        # print(filename)
        separator.seperate(f"test_inputs/{filename}")