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

def handle_run_name_weirdness(cfg):
    """
    All of this prevents overwriting of existing runs.
    """
    if cfg.RUN_NAME == '':
        # Generate a run name from the current time
        cfg.RUN_NAME = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')

    mode_name = "genetic_only" if cfg.DATASET.MODE == 1 else ("image_only" if cfg.DATASET.MODE == 2 else "joint")
    # Check if RUN_NAME already exists in output, change it if it doesn't
    i = 0
    print(os.path.join("../output", cfg.RUN_NAME))
    root_run_name = cfg.RUN_NAME
    while os.path.exists(os.path.join("../output", mode_name, cfg.RUN_NAME)):
        i += 1
        cfg.RUN_NAME = f"{root_run_name}_{i:03d}"

    if cfg.OUTPUT.MODEL_DIR == '':
        cfg.OUTPUT.MODEL_DIR = os.path.join("../output", mode_name, cfg.RUN_NAME)
        # If the model directory doesn't exist, create it
        os.makedir(cfg.OUTPUT.MODEL_DIR)
    cfg.OUTPUT.IMG_DIR = os.path.join(cfg.OUTPUT.MODEL_DIR, "images")

def format_dictionary_nicely_for_printing(obj):
    """
    Format a dictionary nicely for printing.
    """
    return '\n'.join([f"{k}: {v}" for k, v in obj.items()])
