import torch
from model.model import ProtoNode, CombinerProtoNode
from typing import Union

def get_cluster_and_sep_cost(min_distances, target, num_classes):
    target_one_hot = torch.zeros(target.size(0), num_classes).to("cuda")
    make_one_hot(target - 1, target_one_hot)
    num_prototypes_per_class = min_distances.size(1) // num_classes
    one_hot_repeat = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)
    cluster_cost = torch.mean(torch.min(min_distances * one_hot_repeat, dim=1)[0])

    flipped_one_hot_repeat = 1 - one_hot_repeat
    inverted_distances_to_nontarget_prototypes, _ = \
        torch.max((0 - min_distances) * flipped_one_hot_repeat, dim=1)
    separation_cost = torch.mean(0 - inverted_distances_to_nontarget_prototypes)

    return cluster_cost, separation_cost

def get_l1_cost(node):
    l1_mask = ( 1 - node.match.clone().t().to("cuda"))
    masked_weights = node.last_layer.weight.clone() * l1_mask
    l1 = torch.linalg.vector_norm(masked_weights.clone(), ord=1)
    return l1

def get_multi_last_layer_l1_cost(node):
    l1_mask = (
        1- torch.t(node.match).to("cuda")
    )
    l1 = torch.linalg.vector_norm(
        node.multi_last_layer.weight * l1_mask, ord=1
    )

    return l1

def get_orthogonality_cost(node):
    P = node.prototype_vectors.squeeze(-1).squeeze(-1)
    P = P / torch.linalg.norm(P, dim=1).unsqueeze(-1)
    return torch.sum(P@P.T-torch.eye(P.size(0)).cuda())

def get_correlation_matrix(genetic_min_distances, image_min_distances):
    genetic_min_distances = genetic_min_distances.view(genetic_min_distances.shape[0], 40, -1)
    image_min_distances = image_min_distances.view(genetic_min_distances.shape[0], 10, -1)

    sq_distance = torch.zeros(genetic_min_distances.shape[0], 40, 10).cuda()

    for i in range(40):
        for j in range(10):
            sq_distance[:,i,j] = (genetic_min_distances[:,i,0] - image_min_distances[:,j,0]) ** 2

    # sq_distance = (genetic_min_distances.unsqueeze(2) - image_min_distances.unsqueeze(1)) ** 2
    sq_distance = torch.sum(sq_distance, dim=0)

    return sq_distance[:,:]

# Both of these correspondence costs work by matching each of the n image prototypes to b of the bn genetic prototypes. Obviously this requires that b be an integer. This raises some issues w/ pruning. More thought is reqd.

def get_correspondence_loss_batched(
    genetic_min_distances,
    image_min_distances,
    mask,
    node
):
    if len(node.named_location) and node.named_location[-1] == "Diptera":
        # node.correlation_table += get_correlation_matrix(genetic_min_distances[mask], image_min_distances[mask])
        node.correlation_count += len(genetic_min_distances[mask])

    wrapped_genetic_min_distances = genetic_min_distances[mask].view(
        -1, genetic_min_distances.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = image_min_distances[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)

    # Calculate the dot product of the normalized distances along the batch dimension (gross)
    l2_distance = (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2
    total_dist = torch.sum(
        l2_distance,
        dim=0
    )

    # Get the maximum dot product for each image prototype
    min_correspondence_costs, min_correspondence_cost_indicies = torch.min(total_dist, dim=1)

    individual_min_indicies = torch.min(
        l2_distance,
        dim=2
    )[1]

    node.max_tracker[0].append(min_correspondence_cost_indicies)
    node.max_tracker[1].append(individual_min_indicies)

    correspondence_cost_count = len(min_correspondence_costs)
    correspondence_cost_summed = torch.sum(min_correspondence_costs)

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis, l2_distance, total_dist

    return correspondence_cost_summed, correspondence_cost_count

def get_correspondence_loss_single(
    genetic_min_distances,
    image_min_distances,
    mask,
    node
):
    wrapped_genetic_min_distances = genetic_min_distances[mask].view(
        -1, genetic_min_distances.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = image_min_distances[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)
    
    # Calculate the total correspondence cost, minimum MSE between corresponding prototypes. We will later divide this by the number of comparisons made to get the average correspondence cost
    correspondence_cost_count = len(wrapped_genetic_min_distances)
    correspondence_cost_summed = torch.sum(
        torch.min(
            (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2,
            dim=2
        )[0]
    )

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis

    return correspondence_cost_summed, correspondence_cost_count

def recursive_get_loss_multi(
        conv_features,
        node,
        target,
        prev_mask,
        level,
        global_ce,
        correct_arr,
        total_arr,
        accuracy_tree,
        parallel_mode,
        cfg
):
    """
    This is the same as recursive get loss, but uses the multi logits
    """
    logits, (genetic_min_distances, image_min_distances) = node(conv_features, get_middle_logits=parallel_mode)
    logits_size_1 = logits[0].size(1) if parallel_mode else logits.size(1) 

    # Mask out unclassified examples
    mask = target[:,level] > 0
    # Mask out examples that don't belong to the current node
    mask = prev_mask & mask
    num_parents_in_batch = 1

    # Be cognizant of a potential memory leak here
    # Populate the node with logits (for conditional probability calculation)
    genetic_logits, image_logits = logits

    # Check if the node has attribute _logits
    if hasattr(node.genetic_tree_node, "_logits"):
        del node.genetic_tree_node._logits
    if hasattr(node.image_tree_node, "_logits"):
        del node.image_tree_node._logits

    node.genetic_tree_node._logits = genetic_logits
    node.image_tree_node._logits = image_logits

    if cfg.OPTIM.CORRESPONDENCE_TYPE.lower() == "batched":
        correspondence_cost_summed, correspondence_cost_count = get_correspondence_loss_batched(
            genetic_min_distances,
            image_min_distances,
            mask,
            node
        )
    elif cfg.OPTIM.CORRESPONDENCE_TYPE.lower() == "single":
        correspondence_cost_summed, correspondence_cost_count = get_correspondence_loss_single(
            genetic_min_distances,
            image_min_distances,
            mask,
            node
        )
    else:
        raise ValueError("Invalid correspondence type")

    genetic_orthogonality_cost = get_orthogonality_cost(node.genetic_tree_node)
    image_orthogonality_cost = get_orthogonality_cost(node.image_tree_node)

    orthogonality_cost = torch.tensor([genetic_orthogonality_cost, image_orthogonality_cost])

    if mask.sum() == 0:
        for c_node in node.child_nodes:
            i = c_node.int_location[-1] - 1
            
            new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_correspondence_cost_summed, new_correspondence_cost_count, new_orthogonality_cost = recursive_get_loss_multi(
                conv_features,
                c_node,
                target,
                mask,
                level + 1,
                global_ce,
                correct_arr,
                total_arr,
                accuracy_tree["children"][i],
                parallel_mode,
                cfg
            )
            correspondence_cost_summed += new_correspondence_cost_summed
            correspondence_cost_count += new_correspondence_cost_count
            orthogonality_cost += new_orthogonality_cost
        
        del genetic_logits, image_logits, logits, genetic_min_distances, image_min_distances
        return 0, 0, 0, 0, 0, correspondence_cost_summed, correspondence_cost_count, orthogonality_cost
    else:
        if global_ce:
            cross_entropy = 0
        else:
            if parallel_mode:
                # If we are in parallel mode, we calculate each model's cross entropy separately. We ignore the last layer.
                genetic_cross_entropy = torch.nn.functional.cross_entropy(genetic_logits[mask], target[mask][:, level] - 1)
                image_cross_entropy = torch.nn.functional.cross_entropy(image_logits[mask], target[mask][:, level] - 1)
                cross_entropy = genetic_cross_entropy + image_cross_entropy
                
                del genetic_cross_entropy, image_cross_entropy        
            else:
                # If we are not in parallel mode, we calculate the cross entropy using the multi logits
                cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)

        del genetic_logits, image_logits

        genetic_cluster_cost, genetic_separation_cost = get_cluster_and_sep_cost(
            genetic_min_distances[mask], target[mask][:, level], logits_size_1)
        image_cluster_cost, image_separation_cost = get_cluster_and_sep_cost(
            image_min_distances[mask], target[mask][:, level], logits_size_1)

        cluster_cost = genetic_cluster_cost + image_cluster_cost
        separation_cost = genetic_separation_cost + image_separation_cost

        genetic_l1_cost = get_l1_cost(node.genetic_tree_node)
        image_l1_cost = get_l1_cost(node.image_tree_node)
        multi_layer_l1_cost = get_multi_last_layer_l1_cost(node)

        l1_cost = genetic_l1_cost + image_l1_cost + multi_layer_l1_cost

        # Update correct and total counts
        # TODO - Implement accuracy tree for parallel mode
        if parallel_mode:
            predicted = torch.zeros_like(target[mask][:, level])
            correct = torch.zeros_like(target[mask][:, level])
        else:
            _, predicted = torch.max(logits[mask], 1)
            correct = predicted == (target[mask][:, level] - 1)

        correct_arr[level] += correct.sum().item()
        total_arr[level] += len(predicted)

        for i in range(logits_size_1):
            class_mask = (target[mask][:, level] - 1) == i
            class_correct = correct[class_mask]
            accuracy_tree["children"][i]["correct"] += class_correct.sum()
            accuracy_tree["children"][i]["total"] += class_mask.sum()
            
        for c_node in node.child_nodes:
            i = c_node.int_location[-1] - 1
            
            applicable_mask = target[:,level] - 1 == i
            
            new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_correspondence_cost_summed, new_correspondence_cost_count, new_orthogonality_cost = recursive_get_loss_multi(
                conv_features,
                c_node,
                target,
                mask & applicable_mask,
                level + 1,
                global_ce,
                correct_arr,
                total_arr,
                accuracy_tree["children"][i],
                parallel_mode,
                cfg
            )
            
            cross_entropy = cross_entropy + new_cross_entropy
            cluster_cost = cluster_cost + new_cluster_cost 
            separation_cost = separation_cost + new_separation_cost
            l1_cost = l1_cost + new_l1_cost
            num_parents_in_batch =  num_parents_in_batch + new_num_parents_in_batch
            correspondence_cost_summed += new_correspondence_cost_summed
            correspondence_cost_count += new_correspondence_cost_count
            orthogonality_cost += new_orthogonality_cost

            del applicable_mask, new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_correspondence_cost_summed, new_correspondence_cost_count, new_orthogonality_cost

        del logits, genetic_min_distances, image_min_distances, predicted, correct, logits_size_1
        return cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch, correspondence_cost_summed, correspondence_cost_count, orthogonality_cost

def get_loss(
    conv_features,
    node: Union[ProtoNode, CombinerProtoNode],
    target
):
    """
    conv_features: The output of the convolutional layers
    Node: the current node in the tree
    target: the target labels of shape [B, 4] = [40, 4]
    prev_mask: the mask of the previous node
    global_ce: whether to use global_ce
    """
    # Given the target of 80 labels, you only want to look at the relevant ones. 
    # If your node idx is [2, 3], then you should be looking for all samples of 
    # the form [2, 3, *], and you filter out only * 

    if node.taxnode.depth == 0: 
        # this is root node that classifies order and you should consider everything 
        mask = torch.ones(target.size(0), dtype=torch.bool)
    elif node.taxnode.depth == 4: 
        return 0, 0, 0, 0, 0
    else: 
        # there will be irrelevant nodes and you should mask them 
        mask = (target[:,node.taxnode.depth-1] == node.taxnode.idx[-1])

    if mask.sum() == 0: 
        # no samples are relevant for this node since they are not in the taxonomy
        return 0, 0, 0, 0, 0

    logits, min_distances = node(conv_features)
    
    logits = logits[mask] # [B, nclasses] -> [M, nclasses] for B >= M 
    target = target[mask] # [B, 4] -> [M, 4]
    min_distances = min_distances[mask] # [B, nproto_total] -> [M, nproto_total]
    conv_features = conv_features[mask]

    cross_entropy = torch.nn.functional.cross_entropy(logits, target[:, node.taxnode.depth] - 1)

    cluster_cost, separation_cost = get_cluster_and_sep_cost(
        min_distances, target[:, node.taxnode.depth], logits.size(1)
    )

    l1_cost = get_l1_cost(node) 

    # Update correct and total counts
    _, predicted = torch.max(logits, dim=1) 

    node.predictions += len(predicted)
    node.correct += (predicted == target[:, node.taxnode.depth] - 1).sum() 

    n_classifications = 1
    # Now recurse on the new ones 
    for _, c_node in node.childs.items():

        c_cross_entropy, c_cluster_cost, c_separation_cost, c_l1_cost, c_n_classifications = get_loss(
            conv_features,
            c_node,
            target
        )

        cross_entropy += c_cross_entropy
        cluster_cost += c_cluster_cost 
        separation_cost += c_separation_cost
        l1_cost += c_l1_cost
        n_classifications += c_n_classifications

    del logits, min_distances

    return cross_entropy, cluster_cost, separation_cost, l1_cost, n_classifications

def make_one_hot(target, target_one_hot):
    target_copy = torch.LongTensor(len(target))
    target_copy.copy_(target)
    target_copy = target_copy.view(-1,1).to("cuda")
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target_copy, value=1.)

def auxiliary_costs(label,num_prototypes_per_class,num_classes,prototype_shape,min_distances):
    if label.size(0) == 0: 
        return 0, 0
    
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    target = label.to("cuda")
    target_one_hot = torch.zeros(target.size(0), num_classes)
    target_one_hot = target_one_hot.to("cuda")                
    make_one_hot(target, target_one_hot)
    one_hot_repeat = target_one_hot.unsqueeze(2).repeat(1,1,num_prototypes_per_class).\
                        view(target_one_hot.size(0),-1)
    inverted_distances, _ = torch.max((max_dist - min_distances) * one_hot_repeat, dim=1)
    cluster_cost = torch.mean(max_dist - inverted_distances)

    flipped_one_hot_repeat = 1 - one_hot_repeat
    inverted_distances_to_nontarget_prototypes, _ = \
        torch.max((max_dist - min_distances) * flipped_one_hot_repeat, dim=1)
    separation_cost = torch.mean(inverted_distances_to_nontarget_prototypes)

    return cluster_cost, separation_cost

