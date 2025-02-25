import torch
from torch.nn import Module
from typing_extensions import deprecated

def get_optimizers(model: Module):
    # through_protos_optimizer
    through_protos_optimizer_specs = [
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.get_prototype_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}
    ]
    through_protos_optimizer = torch.optim.Adam(through_protos_optimizer_specs)
    
    warm_optimizer_specs = [
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.get_prototype_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    
    last_layer_optimizer = torch.optim.SGD(
        params = model.get_last_layer_parameters(), 
        lr = 3e-3,
        momentum = .9
    )
    
    # joint optimizer
    joint_optimizer_specs = [
        {'params': model.features.parameters(), 'lr': 1e-6, 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.get_last_layer_parameters(), 'lr': 3e-4, 'weight_decay': 1e-3},
        {'params': model.get_prototype_parameters(), 'lr': 3e-4, 'weight_decay': 1e-3}
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

