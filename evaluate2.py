# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import numpy as np
import musdb
import museval
import torch as th
import logging

import random
import torch
from torch.utils.data import DataLoader

from torchinfo import summary
import argparse
import json
from datetime import datetime
import os



# import dataset as module_data
# import loss as module_loss
from eCMU import models as module_arch
from utils import write_samples, tensorboard_add_sample, visualize
import copy
from tqdm import tqdm
from jsonschema import validate
from pathlib import Path
import soundfile as sf
from tqdm.contrib.concurrent import process_map
import yaml
from ml_collections import ConfigDict
from eCMU.models.separator import MusicSeparationModel


from eCMU.models.transformer import OpenUnmix

def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores


def eval_track(references, estimates, win, hop, compute_sdr=False):
    new_scores = new_sdr(references, estimates)[0]
    references = references.squeeze(0).transpose(1, 2).double()
    estimates = estimates.squeeze(0).transpose(1, 2).double()

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.cpu().numpy()
        estimates = estimates.cpu().numpy()

        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]

        return scores, new_scores

def help_eval(index):
    track = test_set.tracks[index]
    references = []
    for target in sources:
        y, _ = sf.read(
            str(track.sources[target]), frames=int(track.duration)*src_rate, start=0,
            dtype='float32', fill_value=0.
        )
        y = torch.as_tensor(y.T, dtype=torch.float32)
        references.append(y)
    
    references = torch.stack(references).unsqueeze(0)
    return references
    # for target in sources:
    #     y, _ = sf.read(
    #         f'{output_dir}/{track.name}/{target}.wav', frames=int(track.duration)*src_rate, start=0,
    #         dtype='float32', fill_value=0.
    #     )
    #     y = torch.as_tensor(y.T, dtype=torch.float32)
    #     estimates.append(y)
    # print('1')
    # estimates = torch.stack(estimates).unsqueeze(0)
    # return eval_track(references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)

def separate(model, index):
    track = test_set.tracks[index]
    # if index != 28:
    #     continue
    # else:
    #     print(track.name)
    # if track.name not in ['PR - Happy Daze', 'Enda Reilly - Cur An Long Ag Seol', 'Skelpolu - Resurrection', 'Lyndsey Ollard - Catching Up', 'PR - Oh No']:
    #     continue
        
    mix, _ = sf.read(
                str(track.path), frames=int(track.duration)*src_rate, start=0,
                dtype='float32', fill_value=0.
            )
    mix = torch.as_tensor(mix.T, dtype=torch.float32).cuda()

    estimates = model.seperate(mix[None,...])

    return estimates

    # for k, target in enumerate(sources):
    #     # Path(f'{output_dir}/{target}/{track.name}').mkdir(parents=True, exist_ok=True)
    #     # sf.write(f'{output_dir}/{target}/{track.name}/predict.wav', estimates[0][k].T.numpy(), src_rate)
    #     # sf.write(f'{output_dir}/{target}/{track.name}/gt.wav', references[0][k].T.numpy(), src_rate)
    #     Path(f'{output_dir}/{track.name}').mkdir(parents=True, exist_ok=True)
    #     sf.write(f'{output_dir}/{track.name}/{target}.wav', estimates[0][k].T.numpy(), src_rate)
  
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
    print(result)

    with open(f'{json_folder}/result.json', 'w') as f:
        json.dump(result, f)

    return result

def evaluate1(model, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """
    tracks = {}

    for index in tqdm(range(len(test_set))):
        track = test_set.tracks[index]
        track_name = track.name
        for track_name not in ["Lyndsey Ollard - Catching Up"]:
            continue
        estimates = separate(model, index)
        references = help_eval(index)

        scores, nsdrs = eval_track(references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)

        tracks[track_name] = {}
        print(track_name, nsdrs, nsdrs.mean())
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

    compute_score(tracks, sources)
 


def dump_report_materials(model, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """
    tracks = {}
    sources = model.targets
    examples = {"good": ['Al James - Schoolboy Facination', 'AM Contra - Heart Peripheral'], "bad": ['PR - Happy Daze', 'Skelpolu - Resurrection']}
    for index in tqdm(range(len(test_set))):
        track = test_set.tracks[index]
        track_name = track.name
        if track.name not in (examples["good"]+examples["bad"]):
            continue
        estimates = separate(model, index).to(device)
        references = help_eval(index).to(device)

        scores, nsdrs = eval_track(references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)

        tracks[track_name] = {}
        print(track_name, nsdrs)
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

        src_rate = 44100
        t1, t2 = 20*src_rate, 26*src_rate
        mix, _ = sf.read(
                    str(track.path), frames=6*src_rate, start=t1,
                    dtype='float32', fill_value=0.
                )
        print(mix.shape, estimates.shape)
        
        for k, target in enumerate(sources):
            Path(f'{output_dir}/{track.name}/{target}').mkdir(parents=True, exist_ok=True)
            sf.write(f'{output_dir}/{track.name}/{target}/pred.wav', estimates[0][k][..., t1:t2].cpu().T.numpy(), src_rate)
            sf.write(f'{output_dir}/{track.name}/{target}/gt.wav', references[0][k][..., t1:t2].cpu().T.numpy(), src_rate)
            sf.write(f'{output_dir}/{track.name}/mix.wav', mix, src_rate)
  
        visualize(sources, (mix.T, references.cpu().numpy()[0], estimates.cpu().numpy()[0]), f'{output_dir}/{track.name}',t1, t2)

    compute_score(tracks, sources)
 



save_pretrain = True



sources = ["vocals", "drums", "bass", "other"]
ckpts = ["checkpoints/vocals-ema/epoch=6-avg_sdr=6.849.ckpt", 
        "checkpoints/drums/epoch=189-avg_sdr=6.392.ckptpre",
        "checkpoints/bass-ema/epoch=0-avg_sdr=5.468.ckpt",
        "checkpoints/other-ema/epoch=5-avg_sdr=4.389.ckpt"]
separator = MusicSeparationModel(sources=sources, ckpts=ckpts)
test_set = musdb.DB("../musdb", subsets=["test"], is_wav=True)
# test_set = musdb.DB("../musdb", subsets=["train"], split="valid", is_wav=True)
output_dir = "results/samples"
Path(output_dir).mkdir(parents=True, exist_ok=True)

model_ver = "all"
json_folder = f'results/test/{model_ver}'
# if save_pretrain:
#     print(ckpt+"pre")
#     torch.save(model.model.state_dict(), ckpt+"pre")
Path(json_folder).mkdir(parents=True, exist_ok=True)
src_rate = 44100

win = int(1. * src_rate)
hop = int(1. * src_rate)
print(sources)

if __name__ == "__main__":
    # dump_report_materials(separator, compute_sdr=False)
    evaluate1(separator, compute_sdr=True)
