import torch
from torch.nn import Module

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
        {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.get_last_layer_parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.get_prototype_parameters(), 'lr': 3e-4, 'weight_decay': 1e-3}
    ]
    
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

    return through_protos_optimizer, warm_optimizer, last_layer_optimizer, joint_optimizer
