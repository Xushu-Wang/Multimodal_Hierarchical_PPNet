import torch
import bisect

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
    preprocess_input_function,
    k,
    tau,
    do_pruning_log=True,
    log=print
):
    if prototype_network_parallel.module.mode == 3:
        for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
            nodal_init_pruning_datastructures(node, k)
        for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
            nodal_init_pruning_datastructures(node, k)
    else:
        for node in prototype_network_parallel.module.nodes_with_children:
            nodal_init_pruning_datastructures(node, k)

    with torch.no_grad():
        for i, ((genetics, image), labels) in enumerate(dataloader):
            if prototype_network_parallel.module.mode == 1:
                search_batch_input = genetics
            elif prototype_network_parallel.module.mode == 2:
                search_batch_input = image
            else:
                search_batch_input = (genetics, image)

            """
            search_batch_input is the raw input image, this is used for generating output images.
            search_batch is the input to the network, this is used for generating the feature maps. This may be normalzied
            """
            if preprocess_input_function is not None and prototype_network_parallel.module.mode == 2:
                # print('preprocessing input for pushing ...')
                # search_batch = copy.deepcopy(search_batch_input)
                search_batch = preprocess_input_function(search_batch_input)
            elif preprocess_input_function is not None and prototype_network_parallel.module.mode == 3:
                search_batch = (search_batch_input[0], preprocess_input_function(search_batch_input[1]))
            else:
                search_batch = search_batch_input

            if prototype_network_parallel.module.mode == 3:
                search_batch = (search_batch[0].cuda(), search_batch[1].cuda())
            else:
                search_batch = search_batch.cuda()

            conv_features = prototype_network_parallel.module.conv_features(search_batch)
            labels = labels.cuda()

            if prototype_network_parallel.module.mode == 3:
                for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
                    nodal_update_pruning_datastructures(
                        node,
                        conv_features,
                        labels,
                        k,
                        log=log
                    )
                for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
                    nodal_update_pruning_datastructures(
                        node,
                        conv_features,
                        labels,
                        k,
                        log=log
                    )
            else:
                for node in prototype_network_parallel.module.nodes_with_children:
                    nodal_update_pruning_datastructures(
                        node,
                        conv_features,
                        labels,
                        k,
                        log=log
                    )
    
    if prototype_network_parallel.module.mode == 3:
        for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
            nodal_update_pruning_mask(node, k, tau)
            if do_pruning_log:
                print_quality_pruning_information(node, log)
            nodal_clear_pruning_datastructures(node)
        for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
            nodal_update_pruning_mask(node, k, tau)
            nodal_clear_pruning_datastructures(node)
    else:
        for node in prototype_network_parallel.module.nodes_with_children:
            nodal_update_pruning_mask(node, k, tau)
            nodal_clear_pruning_datastructures(node)

"""
Painfully slow function that updates the k-best samples for each prototype in a node.
"""
def nodal_update_pruning_datastructures(
    node,
    conv_features,
    labels,
    k,
    ignore_not_classified=True,
    log=print
):
    cpu_labels = labels.cpu()

    # This has shape of (batch_size, num_prototypes)
    distances = node.push_get_dist(conv_features)

    # Only consider samples applicable to this node
    applicable_mask = torch.ones(distances.shape[0], dtype=torch.bool)
    for i, class_index in enumerate(node.int_location):
        applicable_mask &= (cpu_labels[:, i] == class_index).bool()
    
    # We don't care if it's close to a prototype that isn't classified
    if ignore_not_classified:
        applicable_mask &= (cpu_labels[:, len(node.int_location)] != 0).bool()

    applicable_mask = applicable_mask.unsqueeze(1).unsqueeze(2)

    # Move distances to the cpu
    distances = distances.cpu()

    for i, kth_prototype_distance in enumerate(node.kth_prototype_distances):
        # Find all samples that have a distance to the prototype that is less than the kth prototype distance
        better_mask = distances[:, i] < kth_prototype_distance
        better_mask &= applicable_mask

        # Iterate over the better samples inserting them into the sorted list of distances
        for j, better in enumerate(better_mask):
            if not better:
                continue
            
            # Insert the distance into the sorted list of distances
            insertion_index = bisect.bisect_left(node.best_prototype_distances[i], distances[j, i])
            node.best_prototype_distances[i].insert(insertion_index, distances[j, i])
            node.best_prototype_distances[i] = node.best_prototype_distances[i][:k]
            
            # Update the class index
            node.best_prototype_classes[i].insert(insertion_index, cpu_labels[j, len(node.int_location)])
            node.best_prototype_classes[i] = node.best_prototype_classes[i][:k]
            
            # Update the kth prototype distance
            node.kth_prototype_distances[i] = node.best_prototype_distances[i][-1]

def nodal_init_pruning_datastructures(node, k):
    # node.best_prototype_distances stores the distances of the k nearest samples thus far for each prototype
    node.best_prototype_distances = [[float('inf') for _ in range(k)] for _ in range(len(node.prototype_vectors))]
    node.kth_prototype_distances = [float('inf') for _ in range(len(node.prototype_vectors))]
    # node.best_prototype_classes stores the indices of the k nearest samples thus far for each prototype
    node.best_prototype_classes = [[-1 for _ in range(k)] for _ in range(len(node.prototype_vectors))]

"""
This looks at the closest prototypes and sees if we match the quality condition
"""
def nodal_update_pruning_mask(node, k, tau):
    node.prototype_mask.data.zero_()
    for i, proto_classes in enumerate(node.best_prototype_classes):
        best_prototype_class_index = i // node.num_prototypes_per_class
        best_prototype_class_label = best_prototype_class_index + 1

        if len([x for x in proto_classes if x == best_prototype_class_label]) >= tau:
            node.prototype_mask.data[i] = 1

def nodal_clear_pruning_datastructures(node):
    del node.best_prototype_distances
    del node.kth_prototype_distances
    del node.best_prototype_classes

def print_quality_pruning_information(node, log=print):
    print(f"{node.int_location}\t{node.prototype_mask.view(node.num_prototypes_per_class, -1).sum(dim=0)}")

"""
This function finds the x best prototypes (based on weights) per class, and masks the rest.
It then optimizes the network with the masked prototypes.
"""
def prune_prototypes(
    prototype_network_parallel, # pytorch network with prototype_vectors
    dataloader,
    preprocess_input_function=None,
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
            preprocess_input_function,
            k,
            tau,
            log=log
        )
    else:
        raise ValueError('Invalid pruning type')
    
    for node in prototype_network_parallel.module.nodes_with_children:
        print(node.prototype_mask.view(node.num_prototypes_per_class, -1).sum(dim=0), node.num_prototypes_per_class)