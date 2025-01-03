import argparse
import os
from pathlib import Path
import yaml
import torch
import librosa
import soundfile as sf
from torchaudio.transforms import Spectrogram, InverseSpectrogram


from .ecmu import eCMU
from core.utils import MWF, padding

from typing import List
import time

class MusicSeparationModel:
    SOURCES = ["vocals", "drums", "bass", "other"]
    def __init__(self, 
                model_root: str,
                source_names: List[str],
                n_fft: int = 4096,
                hop_length: int = 1024,
                use_gpu: bool = True
    ):
        """Initilize 
        Args:
            model_root (str): Path to model checkpoints
            source_names List[str]: A list of target source names (stems)
            n_fft (int): window size
            hop_length (int): hop length
            use_gpu (bool): Use GPU or CPU
        Outputs:
            Output audio file for each stem
        """
        self.models = {}
        self.source_names = source_names
        self.target_sr = 44100
        
        for src_name in source_names:
            assert src_name in MusicSeparationModel.SOURCES, f"{src_name} is no available!"
        
        use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")

        if not use_gpu:
            print("WARNING - Running on cpu takes much more time!")

        self.mwf = MWF()
        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None).to(self.device)
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(self.device)

        for src_name in source_names:
            ckpt_path = os.path.join(model_root, "weights", f"{src_name}.ckpt")
            cfg_path = os.path.join(model_root, "configs", f"{src_name}.yaml")

            with open(cfg_path) as f:
                args = yaml.safe_load(f)
            
            ecmu_args = args["model"]["init_args"]["model"]["init_args"]
            self.models[src_name] = eCMU(**ecmu_args).to(self.device)

            if self.device == "cpu":
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            else:
                checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)

            try:
                self.models[src_name].load_state_dict(checkpoint['callbacks']['EMACallback']['state_dict_ema'], strict=False)
            except:
                if "state_dict" in checkpoint:
                    self.models[src_name].load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.models[src_name].load_state_dict(checkpoint, strict=False)


            self.models[src_name].eval()

    def __call__(self, audio_path, output_path="./outputs"):
        """Call for separating a mixture audio file into stems
        Args:
            audio_path (str): Path to input audio
            output_path (str): Path to save output audio
        """
        print("#########################################")
        print("#                                       #")
        print("# 🎤 🥁 Music Source Separation 🎸 🪗    #")
        print("#                                       #")
        print("#########################################")


        filename = os.path.basename(audio_path)[:-4]
        mix, sr = sf.read(audio_path, dtype='float32', fill_value=0.)
        mix = mix.T
        print(f"---> Mixture duration: {round(mix.shape[-1]/sr, 2)}s - Sampling rate: {sr}Hz")

        if sr != self.target_sr:
            print(f"---> Resample mixture audio: {sr}Hz -> {self.target_sr}Hz")
            mix = librosa.resample(mix, orig_sr=sr, target_sr=self.target_sr)
        
        
        mix = torch.tensor(mix).unsqueeze(0).to(self.device)

        print(f"---> Separating mixture audio into {len(self.source_names)} stems ({', '.join(self.source_names)})")
        start_time = time.time()
        estimated_waveforms = self.separate(mix)[0]
        elapsed_time = time.time() - start_time
        print(f"---> Separated in {round(elapsed_time, 2)}s")
        
        print(f"---> Saving result at: {output_path}")
        for src_id, source in enumerate(self.source_names):
            Path(f'{output_path}/{filename}').mkdir(parents=True, exist_ok=True)
            sf.write(f'{output_path}/{filename}/{source}.mp3', estimated_waveforms[src_id].T.numpy(), 44100)
        print("Done!")


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
        
        nb_sources = len(self.source_names)
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
            chunk_out = self.forward(chunk_pad)

            if left_pad > 0:
                chunk_out = chunk_out[...,left_pad:]
            if right_pad > 0:
                chunk_out = chunk_out[...,:-right_pad]

            chunk_out = chunk_out.cpu().detach()
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            y_hat[..., offset:offset + segment] += (w * chunk_out)
            sum_weight[offset:offset + segment] += w
            offset += segment

        assert sum_weight.min() > 0
        y_hat /= sum_weight

        return y_hat
    
    def forward1(self, x):
        """Estimate target waveforms from chunk-level mixture
        Args:
            x (Tensor): 2-channels mixture waveform (B, 2, L) 
        """
        X = self.spec(x)
        X_mag = X.abs()
        
        pred_masks = []
        for source_name in self.source_names:
            out = self.models[source_name](X_mag).squeeze()
            pred_masks.append(out)
        
        pred_masks = torch.stack(pred_masks, axis=0).unsqueeze(0)
        Y_hat = self.mwf(pred_masks, X)
        y_hat = self.inv_spec(Y_hat, length=x.shape[-1])
        return y_hat
    
    def forward(self, x):
        """Estimate target waveforms from chunk-level mixture
        Args:
            x (Tensor): 2-channels mixture waveform (B, 2, L) 
        Return:
            y_hat (Tensor): 2-channels estimated waveform of n_sources (B, n_sources, 2, L)
        """
        X = self.spec(x)
        X_mag = X.abs()
        
        y_hat = []
        for source_name in self.source_names:
            pred_mask = self.models[source_name](X_mag)
            Y = self.mwf(pred_mask, X)[:,0,...]
            pred_wave = self.inv_spec(Y, length=x.shape[-1])
            y_hat.append(pred_wave)
        
        y_hat = torch.stack(y_hat, axis=1)
        
        return y_hat
    
def main():
    parser = argparse.ArgumentParser("eCMU Music Demixer", add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument(
        "--model_ckpt", 
        type=str,
        help="Path to model checkpoints")
    
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        help="provide targets to be processed. If None, all available targets will be computed",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/outputs",
        help="Output path to save separated audio files"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Enable/Disable inference on GPU"
    )
   
    args = parser.parse_args()
    if args.targets is None:
        args.targets = MusicSeparationModel.SOURCES
    
    separator = MusicSeparationModel(model_root=args.model_ckpt, source_names=args.targets, use_gpu=(not args.no_gpu))
    separator(audio_path=args.input, output_path=args.out_dir)
    
if __name__ == "__main__":
    main()
    
