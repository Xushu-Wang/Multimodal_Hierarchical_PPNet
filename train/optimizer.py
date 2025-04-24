import torch
from torch.nn import Module
from yacs.config import CfgNode
from torch.optim import Optimizer 
from typing import Optional 
from enum import Enum

class OptimMode(Enum): 
    """
    Enumeration object to specify which parameters to train. 
    """
    WARM = "warm"
    JOINT = "train"
    LAST = "last"
    TEST = "test"

class Optim: 
    """
    Wrapper class around torch Optimizers with extra label on mode
    """

    def __init__(self, mode: OptimMode, optimizer: Optional[Optimizer]): 
        self.optimizer = optimizer 
        self.mode = mode 

    def step(self): 
        self.optimizer.step()  # type: ignore

    def zero_grad(self): 
        self.optimizer.zero_grad() # type: ignore

def get_optimizers(model: Module, cfg: CfgNode): 
    warm = torch.optim.Adam([
        {
            'params': model.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.TRAIN.WARM.ADD_ON_LAYERS_LR, 
            'weight_decay': cfg.OPTIM.TRAIN.WARM.ADD_ON_LAYERS_WD
        },
        {
            'params': model.get_prototype_parameters(), 
            'lr': cfg.OPTIM.TRAIN.WARM.PROTOTYPE_LR, 
            'weight_decay': 0
        }
    ])

    joint = torch.optim.Adam([
        {
            'params': model.features.parameters(), 
            'lr': cfg.OPTIM.TRAIN.JOINT.FEATURES_LR,
            'weight_decay': cfg.OPTIM.TRAIN.JOINT.FEATURES_WD,
            "name": "features"
        },
        {
            'params': model.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.TRAIN.JOINT.ADD_ON_LAYERS_LR,
            'weight_decay': cfg.OPTIM.TRAIN.JOINT.ADD_ON_LAYERS_WD,
            "name": "add_on_layers"
        },
        {
            'params': model.get_last_layer_parameters(), 
            'lr': cfg.OPTIM.TRAIN.JOINT.LAST_LAYER_LR,
            'weight_decay': cfg.OPTIM.TRAIN.JOINT.LAST_LAYER_WD,
            "name": "last_layer"
        },
        {
            'params': model.get_prototype_parameters(), 
            'lr': cfg.OPTIM.TRAIN.JOINT.PROTOTYPE_LR,
            'weight_decay': 0,
            "name": "prototype"
        }
    ])

    last_layer = torch.optim.SGD(
        params = model.get_last_layer_parameters(), 
        lr = cfg.OPTIM.TRAIN.LAST_LAYER.LAST_LAYER_LR,
        momentum = cfg.OPTIM.TRAIN.LAST_LAYER.LAST_LAYER_MOM
    )

    warm_optim = Optim(OptimMode.WARM, warm)
    joint_optim = Optim(OptimMode.JOINT, joint)
    last_layer_optim = Optim(OptimMode.LAST, last_layer) 
    test_optim = Optim(OptimMode.TEST, None)

    return warm_optim, joint_optim, last_layer_optim, test_optim
