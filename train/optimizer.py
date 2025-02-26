import torch
from torch.nn import Module
from yacs.config import CfgNode

def get_optimizers(model: Module, cfg: CfgNode):
    warm = torch.optim.Adam([
        {
            'params': model.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.WARM.ADD_ON_LAYERS_LR, 
            'weight_decay': cfg.OPTIM.WARM.ADD_ON_LAYERS_WD
        },
        {
            'params': model.get_prototype_parameters(), 
            'lr': cfg.OPTIM.WARM.PROTOTYPE_LR, 
            'weight_decay': cfg.OPTIM.WARM.PROTOTYPE_WD
        }
    ])

    joint = torch.optim.Adam([
        {
            'params': model.features.parameters(), 
            'lr': cfg.OPTIM.JOINT.FEATURES_LR,
            'weight_decay': cfg.OPTIM.JOINT.FEATURES_WD
        },
        {
            'params': model.add_on_layers.parameters(), 
            'lr': cfg.OPTIM.JOINT.ADD_ON_LAYERS_LR,
            'weight_decay': cfg.OPTIM.JOINT.ADD_ON_LAYERS_WD
        },
        {
            'params': model.get_last_layer_parameters(), 
            'lr': cfg.OPTIM.JOINT.LAST_LAYER_LR,
            'weight_decay': cfg.OPTIM.JOINT.LAST_LAYER_WD
        },
        {
            'params': model.get_prototype_parameters(), 
            'lr': cfg.OPTIM.JOINT.PROTOTYPE_LR,
            'weight_decay': cfg.OPTIM.JOINT.PROTOTYPE_WD
        }
    ])

    last_layer = torch.optim.SGD(
        params = model.get_last_layer_parameters(), 
        lr = cfg.OPTIM.LAST_LAYER.LAST_LAYER_LR,
        momentum = cfg.OPTIM.LAST_LAYER.LAST_LAYER_MOM
    )

    return warm, joint, last_layer
