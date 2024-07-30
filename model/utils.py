import numpy as np
import torch

from model.node import Node

def position_encodings(self, x):
    
    """
        Position Encoding Idea:
        We want the dot product of two nearby encodings to be close to 1,
        and the dot product of two faraway encodings to be close to 0.

        We can achieve this by encoding positions into vectors along
        the unit circle between theta=0 and theta = pi/2.

        We append these vectors to the end of the latent space channels.
    """
    
    # x = F.normalize(x, dim=1)
    th = torch.linspace(0, torch.pi /2, x.shape[3], device=x.device)
    pos_1 = torch.cos(th)
    pos_2 = torch.sin(th)

    pos_vec = torch.stack([pos_1, pos_2], dim=0).repeat(x.shape[0], 1, 1).unsqueeze(2)

    return torch.cat([x, pos_vec], dim=1)


def get_optimizers(cfg, ppnet, root): 
    
    # through_protos_optimizer
    
    through_protos_optimizer_specs = [{'params': ppnet.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
	 {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}]
    
    
    internal_nodes = root.nodes_with_children()
    
    for node in internal_nodes:
        through_protos_optimizer_specs.append({'params': getattr(ppnet, node.name + "_prototype_vectors"), 'lr': 3e-3})
    
    through_protos_optimizer = torch.optim.Adam(through_protos_optimizer_specs)
    
    # warm optimizer
    
    warm_optimizer_specs = [{'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}]
    
    for node in internal_nodes:
        warm_optimizer_specs.append({'params': getattr(ppnet, node.name + "_prototype_vectors"), 'lr': 3e-3})

    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    
    # last layer optimizer
    
    last_layers_specs = []
    
    for node in internal_nodes:
        last_layers_specs.append({'params': getattr(ppnet, node.name + "_layer").parameters(), 'lr': 3e-3})

    last_layer_optimizer = torch.optim.SGD(last_layers_specs,momentum=.9)
    
    # joint optimizer
    
    joint_optimizer_specs = [{'params': ppnet.features.parameters(), 'lr': 1e-5, 'weight_decay': 1e-3},
	 {'params': ppnet.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3}]
    
    
    for node in internal_nodes:
        joint_optimizer_specs.append({'params': getattr(ppnet, node.name + "_prototype_vectors"), 'lr': 3e-3})
        joint_optimizer_specs.append({'params': getattr(ppnet, node.name + "_layer").parameters(), 'lr': 3e-3})
    
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

    
    return through_protos_optimizer, warm_optimizer, last_layer_optimizer, joint_optimizer




def adjust_learning_rate(optimizers):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr_ = lr * (0.1 ** (epoch // decay))    
    for optimizer in  optimizers:
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] * .1
            param_group['lr'] = new_lr




nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}
def decode_onehot(onehot, three_dim=True):
    if three_dim:
        onehot = onehot[:, 0, :]
    # Add another row encoding whether the nucleotide is unknown
    onehot = np.vstack([np.zeros(onehot.shape[1]), onehot])
    # Make the unknown nucleotide 1 if all other nucleotides are 0
    onehot[0] = 1 - onehot[1:].sum(0)
    return "".join([list(nucleotides.keys())[list(nucleotides.values()).index(i)] for i in onehot.argmax(0)])



def construct_tree(json_data, parent):
    
    if not json_data or json_data == "not_classified":
        return 
        
    for key, value in json_data.items():
        if key != "not_classified" and key != 'levels':
            parent.add_children(key)
            # Find the child node we just added
            child_node = next(child for child in parent.children if child.name == key)
            if isinstance(value, dict):
                construct_tree(value, child_node)

    
    
def print_tree(node, level=0):
    print('  ' * level + f"{node.name} (Label: {node.label})")
    for child in node.children:
        print_tree(child, level + 1)
        
        
        
if __name__ == '__main__':
    
    json_data = {
        "Diptera": {
            "Cecidomyiidae": {
                "Asteromyia": {"not_classified": None},
                "Campylomyza": {
                    "Campylomyza flavipes": None,
                    "not_classified": None
                },
                "not_classified": {"not_classified": None}
            },
            "Chironomidae": {
                "Tanytarsus": {
                    "Tanytarsus hastatus": None,
                    "Tanytarsus pallidicornis": None,
                    "not_classified": None
                },
                "Metriocnemus": {
                    "Metriocnemus albolineatus": None,
                    "Metriocnemus eurynotus": None,
                    "not_classified": None
                },
                "not_classified": {"not_classified": None}
            },
            "not_classified": {"not_classified": None}
        },
        "Hymenoptera": {
            "Scelionidae": {
                "Telenomus": {
                    "Telenomus remus": None,
                    "Telenomus sechellensis": None,
                    "not_classified": None
                },
                "Trissolcus": {
                    "Trissolcus hyalinipennis": None,
                    "not_classified": None
                },
                "not_classified": {"not_classified": None}
            },
            "Braconidae": {
                "Dinotrema": {
                    "Dinotrema longworthi": None,
                    "Dinotrema Malaise6176": None,
                    "not_classified": None
                },
                "Pseudapanteles": {
                    "Pseudapanteles raulsolorzanoi": None,
                    "Pseudapanteles Malaise34": None,
                    "not_classified": None
                },
                "not_classified": {"not_classified": None}
            },
            "not_classified": {"not_classified": None}
        }
    }
    root = Node("Diptera")
    construct_tree(json_data, root)
    print_tree(root)
