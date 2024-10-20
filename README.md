# eCMU
<b>This is an Official implementation of eCMU: An Efficient Phase-aware Framework for Music Source Separation with Conformer (IEEE RIVF23) </b>
<p align="center">
  <img src="./static/eCMU.png" width="100%"  alt="our pipeline"/>
</p>

Our implementation was developed based on the [`sdx23-aimless`](https://github.com/aim-qmul/sdx23-aimless.git) framework.
## Abstract
From our baseline [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) (UMX), we:
- Attempt to build an affordable model to solve the music source separation (MSS) task in the spectral domain with limited computing resources.
- Apply a differentiable Multi-channel Wiener Filter (MWF) into a mask-based prediction model to end-to-end estimated the complex spectrogram for each source.
- Optimize the model by using the Multi-domain loss function on the public [`MUSDB18-HQ`](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) dataset.
- Leverage the ability of Conformer blocks to capture both local and global feature dependencies on time and frequency axis.

## Demo Page:
🎼 You can remix songs and enjoy [here](https://thamquocdung.github.io/mss-demo/) 📻

## Installation
- python 3.8+
- pytorch-lightning
- pytorch

```bash
pip install -r requirments.txt
```

## Getting Started

## Training

## Evaluation

## Citations
If you find our eCMU useful, please consider citing as below:
```
@INPROCEEDINGS{dungtham2023eCMU,
  author={Tham, Quoc Dung and Nguyen, Duc Dung},
  booktitle={2023 RIVF International Conference on Computing and Communication Technologies (RIVF)}, 
  title={eCMU: An Efficient Phase-aware Framework for Music Source Separation with Conformer}, 
  year={2023},
  pages={447-451},
  doi={10.1109/RIVF60135.2023.10471783}
}
```
