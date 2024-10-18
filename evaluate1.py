import argparse
import functools
import json
import multiprocessing
from typing import Optional, Union

import musdb
import museval
import torch
import tqdm

from pathlib import Path
import soundfile as sf
from main import MyLightningCLI



def separate_and_evaluate(
    track: musdb.MultiTrack,
    target,
    output_dir: str,
    eval_dir: str,
    mus
) -> str:


    mix = sf.read(
              str(track.path), frames=int(track.duration)*src_rate, start=0,
                dtype='float32', fill_value=0.
            )[0]
    mix = torch.as_tensor(mix.T, dtype=torch.float32).cuda()

    out = model.seperate(mix[None,...])[0].cpu().detach().numpy().T
    estimates = {target: out, "residual": out}


    scores = museval.eval_mus_track(track, estimates, output_dir=eval_dir)
    return scores


if __name__ == "__main__":

    
    cli = MyLightningCLI(run=False)
    model = cli.model.eval().cuda()
    print(cli.config['ckpt_path'])
    checkpoint = torch.load(cli.config['ckpt_path'], map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    separator = model # MusicSeparationModel(**params)
    # test_set = musdb.DB("../musdb", subsets=["test"], is_wav=True)
    test_set = musdb.DB("../musdb", subsets=["train"], split="valid", is_wav=True)
    output_dir = "results/samples"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_ver = ".".join(cli.config['ckpt_path'].split("/")[-2:])
    json_folder = f'results/new/{model_ver}'
    Path(json_folder).mkdir(parents=True, exist_ok=True)
    src_rate = 44100

    win = int(1. * src_rate)
    hop = int(1. * src_rate)
    print(model.targets)
    sources = ["drums"] # model.targets
    print(sources)

    
    results = museval.EvalStore()
    for track in tqdm.tqdm(test_set.tracks):
        scores = separate_and_evaluate(
            track,
            target="drums",
            mus=test_set,
            output_dir=output_dir,
            eval_dir=json_folder,
        )
        print(track, "\n", scores)
        results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.model)
    method.save(args.model + ".pandas")