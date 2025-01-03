
"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import numpy as np
import json
import os
import time
from tqdm import tqdm
from pathlib import Path

import torch
import musdb
import museval
import soundfile as sf

from main import MyLightningCLI
from core.loss.time import SDR
from core.models.separator import MusicSeparationModel


sdr = SDR()
def eval_track(references, estimates, windown_size, hop_size, compute_chunk_sdr=False):
    """Call for compute the sdr score
    Args:
        references (Tensor): Groundtruth waveform(B, n_sources, 2, L)
        estimates (Tensor): Estimated waveform(B, n_sources, 2, L)
        windown_size (int): window size
        hop_size (int): hop size
        compute_chunk_sdr (bool): compute chunk-level SDR or only utterance-level SDR
    Return:
        scores (Tuple[np.ndarray, shape(n_sources, )]): A tuple of numpy array scores (sdr, isr, sir, sar) for each source. In this problem, we only use sdr
        new_scores (float): utterance-level SDR
    """
    new_scores = (
            sdr(estimates.view(-1, *estimates.shape[-2:]), references.view(-1, *references.shape[-2:]))   
    )
    references = references.squeeze(0).transpose(1, 2).double()
    estimates = estimates.squeeze(0).transpose(1, 2).double()

    if not compute_chunk_sdr:
        return None, new_scores
    else:
        references = references.cpu().numpy()
        estimates = estimates.cpu().numpy()

        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=windown_size,
            hop=hop_size,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]

        return scores, new_scores

def read_audio(track, sources, src_rate=44100):
    references = []

    # read stem audios
    for source in sources:
        y, _ = sf.read(
            str(track.sources[source]), frames=int(track.duration)*src_rate, start=0,
            dtype='float32', fill_value=0.
        )
        y = torch.as_tensor(y.T, dtype=torch.float32)
        references.append(y)
    
    references = torch.stack(references).unsqueeze(0)

    # read mixture audio
    mixture, _ = sf.read(
                str(track.path), frames=int(track.duration)*src_rate, start=0,
                dtype='float32', fill_value=0.
            )
    mixture = torch.as_tensor(mixture.T, dtype=torch.float32)
    return mixture, references

def compute_score(track_scores, sources):
    """Aggregate the dataset-level SDR from the score of each track
    Args:
        track_scores (Dict[str, Dict[str, Dict[str, float]]]): A score dictionary of each source for each track. ({"Song 1": {"vocals": {"sdr": 5.5}, "bass": {"sdr": 5.5}}})
        sources (List[str]): List of target sources
    Return:
        result (Dict[str, float]): Mean and median of SDR score for each source on the whole dataset
    """
    result = {}

    for value in track_scores.values():
        metric_names = value[sources[0]].keys()
        break

    for metric_name in metric_names:
        avg = 0
        avg_of_medians = 0
        for source in sources:
            medians = [
                np.nanmedian(track_scores[track][source][metric_name])
                for track in track_scores.keys()]
            mean = np.mean(medians)
            median = np.median(medians)
            result[metric_name.lower() + "_" + source] = mean
            result[metric_name.lower() + "_med" + "_" + source] = median
            avg += mean / len(sources)
            avg_of_medians += median / len(sources)
        result[metric_name.lower()] = avg
        result[metric_name.lower() + "_med"] = avg_of_medians

    return result
 

def main():
    parser = argparse.ArgumentParser(description='SS Trainer')
    parser.add_argument("--all", action="store_true", default=False, help="Evaluate for 1 or all sources")
    parser.add_argument("--config", type=str, default=None, help="Path to .yaml file config.")
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to model checkpoints.')
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        help="provide targets to be processed. If None, all available targets will be computed",
    )
    parser.add_argument('--data_root', type=str, help='Path to musdb dataset')

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    src_rate = 44100
    if args.all:
        assert args.model_ckpt is not None, f"Invalid args.model_ckpt: {args.model_ckpt}"
        if args.targets is None:
            args.targets = MusicSeparationModel.SOURCES
        model = MusicSeparationModel(model_root=args.model_ckpt, source_names=args.targets, use_gpu=use_gpu)
        sources = model.source_names
        model_ver = f'all.{args.model_ckpt.split("/")[-1]}'
    else:
        assert args.config is not None, f"Invalid args.config: {args.config}"
        cli = MyLightningCLI(run=False)
        model = cli.model.eval().to(device)
        
        ckpt = cli.config['ckpt_path']
        sources = list(model.targets.keys())
        model_ver = ".".join(ckpt.split("/")[-2:])
   

        print("Loading model from ckpt: ", ckpt)
        checkpoint = torch.load(ckpt, map_location=device, weights_only=True)

        if "EMACallback" in checkpoint['callbacks']:
            print("ema")
            model.model.load_state_dict(checkpoint['callbacks']['EMACallback']['state_dict_ema'], strict=False)
        else:
            print("state_dict")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        
        # print(ckpt+"pre")
        # torch.save(model.model.state_dict(), ckpt+"pre")

    test_set = musdb.DB(args.data_root, subsets=["test"], is_wav=True) 
    # test_set = musdb.DB(args.data_root, subsets=["train"], split="valid", is_wav=True)
    
    print(f"Make an evaluation for: {sources} on {device}")
    start_time = time.time()
    kwargs = {
        "windown_size": int(1. * src_rate),
        "hop_size": int(1. * src_rate),
        "compute_chunk_sdr": True
    }
    
    track_scores = {}
    for index in tqdm(range(len(test_set)), desc="Separate:"):
        track = test_set.tracks[index]
        track_name = track.name
        mixture, references = read_audio(track, sources, src_rate)
        estimates = model.separate(mixture[None,...].to(device))

        track_scores[track_name] = {}

        # Eval track for each single source is faster than for all sources
        for idx, target in enumerate(sources):
            _references = references[:,idx,...].unsqueeze(1)
            _estimates = estimates[:,idx,...].unsqueeze(1)
            scores, nsdrs = eval_track(_references, _estimates, **kwargs)
            track_scores[track_name][target] = {'nsdr': [float(nsdrs[0])]}

            (sdr, isr, sir, sar) = scores
            values = {
                "SDR": sdr[0].tolist(),
                # "SIR": sir[idx].tolist(),
                # "ISR": isr[idx].tolist(),
                # "SAR": sar[idx].tolist()
            }
            track_scores[track_name][target].update(values)

    results = compute_score(track_scores, sources)
    print(results)

    # Dump to .json file
    json_folder = f'results/test/{model_ver}'
    Path(json_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{json_folder}/result.json', 'w') as f:
        json.dump(results, f)

    elapsed_time = int(time.time() - start_time)
    print(f"Evaluation done in {elapsed_time}s")

if __name__ == "__main__":
    main()
