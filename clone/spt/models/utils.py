
import torch
from typing import Dict
import time

def inputs_to_cuda(inputs: Dict[str, torch.Tensor]):
    """
    Move tensors in the inputs to cuda.

    Args:
        inputs (dict[str, torch.Tensor]): Inputs dict

    Returns:
        dict[str, torch.Tensor]: Moved inputs dict

    """
    if not torch.cuda.is_available():
        return inputs
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.cuda()
    return inputs


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)