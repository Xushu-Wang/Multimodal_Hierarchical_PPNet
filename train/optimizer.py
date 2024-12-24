import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import Tuple
from typing_extensions import deprecated

def get_optimizers(ppnet: Module) -> Tuple[Optimizer, Optimizer, Optimizer, Optimizer]:
    # through_protos_optimizer
    through_protos_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': ppnet.get_prototype_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}
    ]
    through_protos_optimizer = torch.optim.Adam(through_protos_optimizer_specs)
    
    warm_optimizer_specs = [
        {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': ppnet.get_prototype_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    
    last_layer_optimizer = torch.optim.SGD(
        params = ppnet.get_last_layer_parameters(), 
        lr = 3e-3,
        momentum = .9
    )
    
    # joint optimizer
    joint_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': ppnet.get_last_layer_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': ppnet.get_prototype_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}
    ]
    
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

    return through_protos_optimizer, warm_optimizer, last_layer_optimizer, joint_optimizer

@deprecated("Not called anywhere.")
def adjust_learning_rate(optimizers) -> None:
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr_ = lr * (0.1 ** (epoch // decay))    
    for optimizer in  optimizers:
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] * .1
            param_group['lr'] = new_lr

