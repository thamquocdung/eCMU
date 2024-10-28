import torch

def padding(x, length):
    offsets = length - x.shape[-1]
    left_pad = offsets // 2
    right_pad = offsets - left_pad

    return left_pad, right_pad, torch.nn.functional.pad(x, (left_pad, right_pad))

def on_load_checkpoint(model, checkpoint: dict) -> None:
    state_dict = checkpoint.copy()
    model_state_dict = model.state_dict()
    is_changed = False
    for k in checkpoint.keys():
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            print(f"Dropping parameter {k}")
            state_dict.pop(k)
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)

    return state_dict