import torch

def padding(x, length):
    offsets = length - x.shape[-1]
    left_pad = offsets // 2
    right_pad = offsets - left_pad

    return left_pad, right_pad, torch.nn.functional.pad(x, (left_pad, right_pad))
