
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

def read_audio(track, sources, src_rate):
    references = []

    # read stem audios
    for target in sources:
        y, _ = sf.read(
            str(track.sources[target]), frames=int(track.duration)*src_rate, start=0,
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

def compute_score(tracks, sources):
    result = {}

    for value in tracks.values():
        metric_names = value[sources[0]].keys()
        break

    for metric_name in metric_names:
        avg = 0
        avg_of_medians = 0
        for source in sources:
            medians = [
                np.nanmedian(tracks[track][source][metric_name])
                for track in tracks.keys()]
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
    parser.add_argument('--model_root', type=str, default=None, help='Path to model checkpoints.')
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
    if args.all:
        assert args.model_root is not None, f"Invalid args.model_root: {args.model_root}"
        if args.targets is None:
            args.targets = MusicSeparationModel.SOURCES
        model = MusicSeparationModel(model_root=args.model_root, source_names=args.targets, use_gpu=use_gpu)
        sources = model.source_names
        model_ver = f'all.{args.model_root.split("/")[-1]}'
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
            model.model.load_state_dict(checkpoint['callbacks']['EMACallback']['state_dict_ema'], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"], strict=True)

    test_set = musdb.DB(args.data_root, subsets=["test"], is_wav=True) 
    # test_set = musdb.DB(args.data_root, subsets=["train"], split="valid", is_wav=True)
    src_rate = 44100
    print(f"Make an evaluation for: {sources} on {device}")
    start_time = time.time()

    pool = ThreadPoolExecutor(multiprocessing.cpu_count())
    kwargs = {
        "windown_size": int(1. * src_rate),
        "hop_size": int(1. * src_rate),
        "compute_chunk_sdr": True
    }
    tracks = {}
    futures = []
    for index in tqdm(range(len(test_set)), desc="Separate:"):
        track = test_set.tracks[index]
        track_name = track.name
        mixture, references = read_audio(track, sources, src_rate)
        estimates = model.separate(mixture[None,...].cuda())

        future = pool.submit(eval_track, references, estimates, **kwargs)
        futures.append((future, track_name))

    futures = tqdm(futures, desc="Compute score:")
    for future, track_name in futures:
        scores, nsdrs = future.result()
        tracks[track_name] = {}

        for idx, target in enumerate(sources):
            tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
    
        if scores is not None:
            (sdr, isr, sir, sar) = scores
            for idx, target in enumerate(sources):
                values = {
                    "SDR": sdr[idx].tolist(),
                    # "SIR": sir[idx].tolist(),
                    # "ISR": isr[idx].tolist(),
                    # "SAR": sar[idx].tolist()
                }
                tracks[track_name][target].update(values)

    results = compute_score(tracks, sources)
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
