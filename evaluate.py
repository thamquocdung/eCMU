# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import musdb
import museval

import torch
from torch.utils.data import DataLoader

from torchinfo import summary
import argparse
import json
from datetime import datetime
import os
import time


from utils import write_samples, tensorboard_add_sample, visualize
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from tqdm.contrib.concurrent import process_map
from core.models.separator import MusicSeparationModel
from main import MyLightningCLI

from core.models.ecmu import eCMU
from core.loss.time import SDR


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

def evaluate(model, windown_size=44100, hop_size=44100, compute_chunk_SDR=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """
    pass
 

# parser = argparse.ArgumentParser(description='SS Trainer')

# parser.add_argument('config', type=str, help='config file')
# parser.add_argument('--checkpoint', type=str, default=None,
#                     help='training checkpoint')

# args = parser.parse_args()

def initialize(cli):
    save_pretrain = False
    model = cli.model.eval().cuda()
    ckpt = cli.config['ckpt_path']
    print("Loading model from ckpt: ", ckpt)
    checkpoint = torch.load(cli.config['ckpt_path'], map_location='cuda', weights_only=True)

    if "EMACallback" in checkpoint['callbacks']:
        model.model.load_state_dict(checkpoint['callbacks']['EMACallback']['state_dict_ema'], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True)


    separator = model # MusicSeparationModel(**params)
    test_set = musdb.DB("/home/datasets/musdb", subsets=["test"], is_wav=True)
    # test_set = musdb.DB("../musdb", subsets=["train"], split="valid", is_wav=True)

    # output_dir = "results/samples"
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_ver = ".".join(ckpt.split("/")[-2:])
    json_folder = f'results/test/{model_ver}'
    Path(json_folder).mkdir(parents=True, exist_ok=True)

    if save_pretrain:
        print(ckpt+"pre")
        torch.save(model.model.state_dict(), ckpt+"pre")
    
    return separator, test_set, json_folder

def main():
    cli = MyLightningCLI(run=False)
    model, test_set, json_folder = initialize(cli)
    src_rate = 44100

    sources = list(model.targets.keys())
    print(sources)
    b = time.time()

    pool = ThreadPoolExecutor(16)
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
    with open(f'{json_folder}/result.json', 'w') as f:
        json.dump(results, f)
    print(time.time() - b)

if __name__ == "__main__":
   main()
