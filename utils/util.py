import datetime
import os
import torch
import numpy as np
from torch import Tensor
from typing_extensions import deprecated 

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

def does_it_match(
    matcher,
    location
):
    for i in range(min(len(location), len(matcher))):
        if matcher[i] != location[i]:
            return False
    return True

def find_node(model, location):
    for node in model.classifier_nodes:
        if len(node.idx) == len(location) and all([a == b for a, b in zip(node.idx, location)]):
            return node
    return None

def find_node_hierarchical(hierarchy, location):
    succ, node = hierarchy.traverse(location)
    return node if succ else None