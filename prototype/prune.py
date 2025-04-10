import torch
import bisect
from model.hierarchical import Mode

def nodal_prune_prototypes_weights(node, cfg):
    """This updates the prototype mask of a node to only include the best 
    prototypes for each class as defined by output weights.
    """
    max_num_prototypes_per_class = cfg.OPTIM.PRUNE.GENETIC_MAX_NUM_PROTOTYPES_PER_CLASS if node.mode == Mode.GENETIC else cfg.OPTIM.PRUNE.IMAGE_MAX_NUM_PROTOTYPES_PER_CLASS

    # Get the n best prototypes per class
    corresponding_weights = node.last_layer.weight.data[node.match.T.bool()]
    corresponding_weights = corresponding_weights.view(node.nprotos, -1)

    # Get the indicies of the best prototypes per class
    best_prototypes = torch.argsort(corresponding_weights, dim=0, descending=True)[:max_num_prototypes_per_class]

    # Each index will be relative to the class, so we need to add the class index * num_prototypes to get the absolute index
    diff = torch.arange(node.nclass) * node.nprotos
    # Scale diff to be the same shape as best_prototypes
    diff = diff.unsqueeze(0).expand(max_num_prototypes_per_class, node.nclass)
    diff = diff.cuda()
    best_prototypes += diff

    # Best prototypes
    node.prototype_mask.data.zero_()
    node.prototype_mask.data[best_prototypes.flatten()] = 1

    del corresponding_weights, diff, best_prototypes

def unprune_prototypes(node):
    """
    This function unmasks all prototypes.
    """
    node.prototype_mask.data[:] = 1

def prune_prototypes_quality(
    model,
    dataloader,
    cfg
):
    """
    This function prunes prototypes according to the approach described in the supplement to This Looks Like That.
    k: Number of nearest neighbors to consider
    tau: Pruning threshold
    """
    if model.mode == Mode.MULTIMODAL:
        for node in model.classifier_nodes:
            nodal_init_pruning_datastructures(node.gen_node, cfg.OPTIM.PRUNE.K)
            # nodal_init_pruning_datastructures(node.img_node, k)
    else:
        raise NotImplementedError()

    with torch.no_grad():
        # search_batch_input is the raw input image, this is used for generating output images.
        # search_batch is the input to the network, this is used for generating the feature maps. 
        # This may be normalzied
        for (genetics, image), (label, _) in dataloader:
            if model.mode == Mode.GENETIC:
                search_batch = genetics.to(cfg.DEVICE)
            elif model.mode == Mode.IMAGE:
                search_batch = image.to(cfg.DEVICE)
            else:
                gen_conv_features = model.gen_net.conv_features(genetics.to(cfg.DEVICE))
                # conv_features = model.conv_features(genetics.cuda(), image.cuda())

            label = label.cuda()

            if model.mode == Mode.MULTIMODAL:
                for node in model.classifier_nodes:
                    nodal_update_pruning_datastructures(
                        node.gen_node,
                        gen_conv_features,
                        label,
                        cfg
                    )
                    # nodal_update_pruning_datastructures(
                    #     node.img_node,
                    #     conv_features[1],
                    #     labels,
                    #     cfg.OPTIM.PRUNE.K,
                    # )
            else:
                raise NotImplementedError()
    
    if model.mode == Mode.MULTIMODAL:
        for node in model.classifier_nodes:
            nodal_update_pruning_mask(node.gen_node, cfg.OPTIM.PRUNE.TAU)
            # nodal_update_pruning_mask(node.img_node, cfg.OPTIM.PRUNE.TAU)

            print_quality_pruning_information(node.gen_node)
            # print_quality_pruning_information(node.img_node)
            
            nodal_clear_pruning_datastructures(node.gen_node)
            # nodal_clear_pruning_datastructures(node.img_node)
    else:
        raise NotImplementedError()

def nodal_update_pruning_datastructures(
    node,
    conv_features,
    label,
    cfg,
    ignore_not_classified=True
):
    """
    Painfully slow function that updates the k-best samples for each prototype in a node.
    """
    cpu_labels = label.cpu()

    # This has shape of (batch_size, num_prototypes)
    _, similarities = node.forward(conv_features)

    # Only consider samples applicable to this node
    applicable_mask = torch.all(cpu_labels[:,:node.depth] == node.idx.cpu(), dim=1)

    # Move similarities to the cpu
    similarities = similarities.cpu()

    for i, kth_prototype_similarity in enumerate(node.kth_prototype_similarities):
        # Find all samples that have a similarity to the prototype that is greater than the kth prototype similarity
        better_mask = similarities[:, i] > kth_prototype_similarity
        better_mask &= applicable_mask

        # Iterate over the better samples inserting them into the sorted list of similarities
        for j, better in enumerate(better_mask):
            if not better:
                continue
            
            # Insert the similarity into the sorted list of similarities
            insertion_index = bisect.bisect_left(node.best_prototype_similarities[i], similarities[j, i])
            node.best_prototype_similarities[i].insert(insertion_index, similarities[j, i])
            if len(node.best_prototype_similarities[i]) > cfg.OPTIM.PRUNE.K:
                node.best_prototype_similarities[i] = node.best_prototype_similarities[i][1:]
            
            # Update the class index
            node.best_prototype_classes[i].insert(insertion_index, cpu_labels[j, node.depth])
            if len(node.best_prototype_classes[i]) > cfg.OPTIM.PRUNE.K:
                node.best_prototype_classes[i] = node.best_prototype_classes[i][1:]
            
            # Update the kth prototype similarity
            node.kth_prototype_similarities[i] = node.best_prototype_similarities[i][0]
            

def nodal_init_pruning_datastructures(node, k):
    # node.best_prototype_similarities stores the similarities of the k nearest samples thus far for each prototype
    node.best_prototype_similarities = [
        [-float('inf') for _ in range(k)] for _ in range(len(node.prototype))
    ]
    node.kth_prototype_similarities = [-float('inf') for _ in range(len(node.prototype))]
    # node.best_prototype_classes stores the indices of the k nearest samples thus far for each prototype
    node.best_prototype_classes = [[-1 for _ in range(k)] for _ in range(len(node.prototype))]

def nodal_update_pruning_mask(node, tau):
    """
    This looks at the closest prototypes and sees if we match the quality condition
    """
    node.prototype_mask.data.zero_()
    for i, proto_classes in enumerate(node.best_prototype_classes):
        best_prototype_class_index = i // node.nprotos

        if len([x for x in proto_classes if x == best_prototype_class_index]) >= tau:
            node.prototype_mask.data[i] = 1

def nodal_clear_pruning_datastructures(node):
    del node.best_prototype_similarities
    del node.kth_prototype_similarities
    del node.best_prototype_classes

def print_quality_pruning_information(node):
    print(f"{node.idx}\t{node.prototype_mask.view(node.nprotos, -1).sum(dim=0)}")

"""
This function finds the x best prototypes (based on weights) per class, and masks the rest.
It then optimizes the network with the masked prototypes.
"""
def prune(
    model,
    dataloader,
    cfg,
    log=print,
):
    if cfg.OPTIM.PRUNE.TYPE == 'weights':
        if model.mode == Mode.MULTIMODAL:
            for node in model.classifier_nodes:
                nodal_prune_prototypes_weights(
                    node.gen_node,
                    cfg
                )
                nodal_prune_prototypes_weights(
                    node.img_node,
                    cfg
                )
        else:
            raise NotImplementedError()
            for node in model.classifier_nodes:
                nodal_prune_prototypes_weights(
                    node,
                )
    elif cfg.OPTIM.PRUNE.TYPE == 'quality':
        prune_prototypes_quality(
            model,
            dataloader,
            cfg
        )
    else:
        raise ValueError('Invalid pruning type')
    
    log("Pruning Info")
    for node in model.classifier_nodes:
        log(f'Genetic: {node.gen_node.prototype_mask.view(node.gen_node.nclass, -1).sum(dim=1)}, {node.gen_node.nprotos}')
        log(f'Image: {node.img_node.prototype_mask.view(node.img_node.nclass, -1).sum(dim=1)}, {node.img_node.nprotos}')
