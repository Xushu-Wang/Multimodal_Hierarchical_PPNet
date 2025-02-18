import datetime
import os
import torch
import numpy as np
from torch import Tensor
from typing_extensions import deprecated 

@deprecated("This is not called anywhere.")
def list_of_distances(X: Tensor, Y: Tensor) -> Tensor:
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def format_dictionary_nicely_for_printing(obj):
    """
    Format a dictionary nicely for printing.

    Format all decimals with 5 decimal places.
    """
    return "\n".join([f"{k}: {v:.5f}" for k, v in obj.items()])

def does_it_match(
    matcher,
    location
):
    for i in range(min(len(location), len(matcher))):
        if matcher[i] != location[i]:
            return False
    return True
