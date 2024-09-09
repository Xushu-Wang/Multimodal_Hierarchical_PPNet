import torch

"""
This updates the prototype mask of a node to only include the best prototypes for each class as defined by output weights.
"""
def nodal_prune_prototypes_weights(
    node,
    log=print
):
    # Get the n best prototypes per class
    corresponding_weights = node.last_layer.weight.data[node.prototype_class_identity.T.bool()]
    corresponding_weights = corresponding_weights.view(node.num_prototypes_per_class, -1)

    # Get the indicies of the best prototypes per class
    best_prototypes = torch.argsort(corresponding_weights, dim=0, descending=True)[:node.max_num_prototypes_per_class]

    # Best prototypes
    node.prototype_mask.data.zero_()
    node.prototype_mask.data[best_prototypes] = 1

"""
This function unmasks all prototypes.
"""
def unprune_prototypes(
    node,
    log=print
):
    node.prototype_mask.data[:] = 1

"""
This function prunes prototypes according to the approach described in the supplement to This Looks Like That.

k: Number of nearest neighbors to consider
tau: Pruning threshold
"""
def prune_prototypes_quality(
    prototype_network_parallel,
    dataloader,
    k,
    tau,
    log=print
):
    for node in prototype_network_parallel.module.nodes_with_children:
        nodal_init_pruning_datastructures(node, k)

    for i, (inputs, labels) in enumerate(dataloader):
        conv_features = prototype_network_parallel.module.conv_features(inputs)
        labels = labels.cuda()

        for node in prototype_network_parallel.module.nodes_with_children:
            nodal_update_pruning_datastructures(
                node,
                conv_features,
                labels,
                k,
                log=log
            )
        

def nodal_update_pruning_datastructures(
    node,
    conv_features,
    labels,
    k,
    log=print
):
    distances = node.push_get_dist(conv_features)

    # for i in range(node.num_prototypes_per_class):
    #     for j in range(node.num_classes):
    #         if 


def nodal_init_pruning_datastructures(node, k):
    node.best_prototype_max = [float("inf")] * node.num_prototypes_per_class
    node.best_prototype_indices = [[0] * k] * node.num_prototypes_per_class
    
def nodal_clear_pruning_datastructures(node):
    del node.best_prototype_distances
    del node.best_prototype_indices

"""
This function finds the x best prototypes (based on weights) per class, and masks the rest.
It then optimizes the network with the masked prototypes.
"""
def prune_prototypes(
    prototype_network_parallel, # pytorch network with prototype_vectors
    dataloader,
    pruning_type='weights',
    k=0,
    tau=0,
    log=print,
):
    if pruning_type == 'weights':
        if prototype_network_parallel.module.mode == 3:
            for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
                nodal_prune_prototypes_weights(
                    node,
                    log=log
                )
            for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
                nodal_prune_prototypes_weights(
                    node,
                    log=log
                )
        else:
            for node in prototype_network_parallel.module.nodes_with_children:
                nodal_prune_prototypes_weights(
                    node,
                    log=log
                )
    elif pruning_type == 'quality':
        prune_prototypes_quality(
            prototype_network_parallel,
            dataloader,
            k,
            tau,
            log=log
        )
    else:
        raise ValueError('Invalid pruning type')