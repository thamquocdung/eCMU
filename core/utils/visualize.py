from torch import nn
import librosa
import soundfile as sf
import os
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import specshow
# import pandas as pd


def _help(file_path):
    target_root='/app/dataset_resampling'
    target_sr=16000
    print(file_path)
    flags = str(file_path).split('/')
    output_path = os.path.join(target_root, *flags[-3:])
    basedir = os.path.join(target_root, *flags[-3:-1])
    Path(basedir).mkdir(exist_ok=True, parents=True)
    x, sr = sf.read(file_path, always_2d=True, dtype='float32', fill_value=0)
    x = librosa.resample(x.T, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, x.T, target_sr, 'PCM_24')

def resampling(source_root='/app/musdb18', target_root='/app/musdb16', target_sr=16000):
    filenames = Path(source_root).glob('**/*.wav')
    
    with Pool(os.cpu_count()) as p:
        p.map(_help, filenames)

def write_samples(tb_logger, gt, output, sources, epoch):
    rows = 2*4 # output, gt
    fig = plt.figure(figsize=(10, 2 * rows))
    axes = fig.subplots(4, 2, sharex=True)
    for i, stem in enumerate(sources):
        axes[i][0].plot(gt[i], label='wave')
        axes[i][0].set_title(f'gt_{stem}')
        axes[i][0].legend()

        axes[i][1].plot(output[i], label='wave')
        axes[i][1].set_title(f'output_{stem}')
        axes[i][1].legend()
    
    tb_logger.writer.add_figure(f'test/waveform', fig, epoch)

    # write spectrogram
    fig = plt.figure(figsize=(8, 1.5 * rows))
    axes = fig.subplots(4, 2, sharex=True)
    for i, stem in enumerate(sources):
        pred_spec = np.abs(librosa.stft(output[i], n_fft=512, hop_length=256))
        gt_spec = np.abs(librosa.stft(gt[i], n_fft=512, hop_length=256))
        specshow(librosa.amplitude_to_db(gt_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes[i][0])
        specshow(librosa.amplitude_to_db(pred_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes[i][1])
        axes[i][0].set_title(f'gt_{stem}')
        axes[i][1].set_title(f'output_{stem}')
    
    tb_logger.writer.add_figure(f'test/spectrogram', fig, epoch)


def tensorboard_add_sample(writer, sources, sample, epoch, id, save_fig=False):
    mix, gt, pred = sample
    t1, t2 = 20*44100, 30*44100
    if writer:
        writer.add_audio(f'mixture/{id}', mix[0, t1:t2], epoch)
    

    for i, source in enumerate(sources):
        fig = plt.figure(figsize=(10, 2 * 3))
        axes = fig.subplots(3, 1, sharex=True)
        axes[0].plot(mix[0,t1:t2])
        if writer:
            writer.add_audio(f'pred/{id}/{source}', pred[i,0, t1:t2], epoch)
            writer.add_audio(f'gt/{id}/{source}', gt[i,0, t1:t2], epoch)
        axes[1].plot(gt[i, 0,t1:t2])
        axes[2].plot(pred[i, 0, t1:t2])

        axes[0].set_title("mix")
        axes[1].set_title("gt")
        axes[2].set_title("pred")
        plt.tight_layout()
        if writer:
            writer.add_figure(f'waveform/{id}/{source}', fig, epoch)

    if save_fig:
        fig.savefig("a.png")


    for i, source in enumerate(sources):
        fig = plt.figure(figsize=(12, 2 * 3))
        axes = fig.subplots(3, 1, sharex=True)

        mix_spec = np.abs(librosa.stft(mix[0, t1:t2].numpy(), n_fft=4096, hop_length=1024))
        gt_spec = np.abs(librosa.stft(gt[i, 0, t1:t2].numpy(), n_fft=4096, hop_length=1024))
        pred_spec = np.abs(librosa.stft(pred[i, 0, t1:t2].numpy(), n_fft=4096, hop_length=1024))

        librosa.display.specshow(librosa.amplitude_to_db(mix_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes[0])
        librosa.display.specshow(librosa.amplitude_to_db(gt_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes[1])
        librosa.display.specshow(librosa.amplitude_to_db(pred_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes[2])
    
    
        axes[0].set_title("mix")
        axes[1].set_title("gt")
        axes[2].set_title("pred")
        plt.tight_layout()

        if writer:
            writer.add_figure(f'spec/{id}/{source}', fig, epoch)
    
    if save_fig:
        fig.savefig("b.png")

def visualize(sources, sample, o_path, t1, t2):
    mix, gt, pred = sample

    fig = plt.figure(figsize=(4.5, 3))
    axes = fig.subplots(1, 1, sharex=True)
    mix_spec = np.abs(librosa.stft(mix[0], n_fft=4096, hop_length=1024))
    librosa.display.specshow(librosa.amplitude_to_db(mix_spec, ref=np.max), y_axis='log', x_axis='time', ax=axes)
    # fig.suptitle("mix")
    plt.tight_layout()
    fig.savefig(f"{o_path}/spec_mix.png")

    for i, source in enumerate(sources):
        for name, wav in [("gt",gt), ("pred", pred)]:
            fig = plt.figure(figsize=(4.5, 3))
            axes = fig.subplots(1, 1, sharex=True)

            spec = np.abs(librosa.stft(wav[i, 0, t1:t2], n_fft=4096, hop_length=1024))
            librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time', ax=axes)
        
            plt.tight_layout()
            fig.savefig(f"{o_path}/{source}/spec_{name}.png")
            plt.close()