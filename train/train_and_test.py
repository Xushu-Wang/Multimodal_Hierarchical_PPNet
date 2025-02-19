import time
import torch
import numpy as np
from pympler.tracker import SummaryTracker
from model.model import Mode
from pprint import pprint
from typing import Union, Tuple

from dataio.dataset import TreeDataset
from utils.util import format_dictionary_nicely_for_printing

tracker = SummaryTracker()

def CE(logits, target):
     # manual definition of the cross entropy for a target which is a probability distribution    
     probs = torch.nn.functional.softmax(logits, 1)    
     return torch.sum(torch.sum(- target * torch.log(probs))) 

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
    l1_mask = (
        1 - torch.t(node.prototype_class_identity).to("cuda")
    )
    l1 = torch.linalg.vector_norm(
        node.last_layer.weight * l1_mask, ord=1
    )
    
    return l1

def get_multi_last_layer_l1_cost(node):
    l1_mask = (
        1- torch.t(node.logit_class_identity).to("cuda")
    )
    l1 = torch.linalg.vector_norm(
        node.multi_last_layer.weight * l1_mask, ord=1
    )

    return l1

def get_orthogonality_cost(node):
    P = node.prototype_vectors.squeeze(-1).squeeze(-1)
    P = P / torch.linalg.norm(P, dim=1).unsqueeze(-1)
    return torch.sum(P@P.T-torch.eye(P.size(0)).cuda())

def print_accuracy_tree(accuracy_tree, log=print):
    print_accuracy_tree_rec(accuracy_tree["children"], log)

def print_accuracy_tree_rec(accuracy_tree, log=print, level=0):
    for entry in accuracy_tree:
        if(entry["total"] == 0):
            log(f'\t{"-" * level * 2}{" " * (level > 0)}{entry["named_location"][-1]}: N/A')
        else:
            log(f'\t{"-" * level * 2}{" " * (level > 0)}{entry["named_location"][-1]}: {entry["correct"] / entry["total"]:.4f} ({entry["total"]} samples)')
        print_accuracy_tree_rec(entry["children"], log, level + 1)

def find_lowest_level_node(node, target):
    if not len(node.child_nodes):
        return node

    if len(target == 1) or target[1] == 0:
        return node
    
    return find_lowest_level_node(
        node.child_nodes[target[0] - 1],
        target[1:]
    )

def clear_accu_probs(node):
    del node.accu_probs

"""
Recursively put the conditional probabilities on the tree
"""
def recursive_update_probs_on_there(node, above_prob=1): 
    node.probs = torch.softmax(node._logits, dim=1) * above_prob  

    for child in node.child_nodes:
        recursive_update_probs_on_there(child, node.probs[:, child.int_location[-1]-1].unsqueeze(1))

"""
Gets the conditional probabilities at each level of conditioning [None, Order, Family, Genus]
"""
def get_conditional_accuracies_flat(
    model,
    target,
    dataset: TreeDataset
):
    tot_correct = [0,0,0,0]
    total = [0,0,0,0]
    
    indexed_tree = dataset.leaf_indicies  

    # Calculate the final layer outputs for the model
    out = []
    indicies = [] 

    recursive_update_probs_on_there(model.root)  

    for node in model.nodes_with_children:
        if len(node.child_nodes) == 0: # only consider leaf nodes
            for child in node.all_child_nodes: # each child is a species
                # just traverse down the tree using each location in named_location
                bit = indexed_tree
                for loc in child.named_location:
                    bit = bit[loc]

                location = bit["idx"]
                out.append(node.probs[:, child.int_location[-1]-1].unsqueeze(1)) 
                indicies.append(location)

    # sum(out) should be a Tensor of 1s

    out = [x for (_, x) in sorted(zip(indicies, out))]
    out_array = torch.stack(out, dim=1).squeeze(2)  

    for cond_level in range(4):
        if cond_level == 0:
            mask = torch.ones_like(out_array, dtype=bool).cuda()
        else:
            mask = dataset.get_species_mask(target[:, cond_level-1], cond_level-1).cuda()  

        # Accuracy
        _, predicted = torch.max(out_array*mask, 1)  # predicted = [80], target = [80, 4]
        total[cond_level] += target.size(0)
        correct = (predicted == target[:,cond_level]).sum().item()
        # correct = (predicted == target[:,-1]).sum().item()
        tot_correct[cond_level] += correct
    
    return torch.tensor(tot_correct), torch.tensor(total)

def get_correlation_matrix(
    genetic_min_distances,
    image_min_distances
):
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


def recursive_get_loss(
        conv_features,
        node,
        target,
        prev_mask,
        level,
        global_ce,
        correct_arr,
        total_arr,
        accuracy_tree
    ):
    """
    conv_features: The output of the convolutional layers
    Node: the current node in the tree
    target: the target labels
    prev_mask: the mask of the previous node
    level: the current level in the tree
    global_ce: whether to use global_ce
    correct_tuple: tuple of correct prediction counts by level
    total_tuple: tuple of total prediction counts by level
    accuracy_tree: object to keep track of accuracy at each split of the tree
    """
    logits, min_distances = node(conv_features)
    
    node.logits = logits

    # Mask out unclassified examples
    mask = target[:,level] > 0
    # Mask out examples that don't belong to the current node
    mask = prev_mask & mask
    num_parents_in_batch = 1

    if mask.sum() == 0:
        return 0, 0, 0, 0, 0

    if global_ce:
        cross_entropy = 0
    else:
        cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)

    cluster_cost, separation_cost = get_cluster_and_sep_cost(
        min_distances[mask], target[mask][:, level], logits.size(1)
    )

    l1_cost = get_l1_cost(node)

    # Update correct and total counts
    _, predicted = torch.max(logits[mask], 1)
    correct = predicted == (target[mask][:, level] - 1)
    correct_arr[level] += correct.sum().item()
    total_arr[level] += len(predicted)

    for i in range(logits.shape[1]):
        class_mask = (target[mask][:, level] - 1) == i
        class_correct = correct[class_mask]
        accuracy_tree["children"][i]["correct"] += class_correct.sum()
        accuracy_tree["children"][i]["total"] += class_mask.sum()
    
    for c_node in node.child_nodes:
        i = c_node.int_location[-1] - 1

        applicable_mask = target[:,level] - 1 == i
        
        if applicable_mask.sum() == 0:
            continue

        new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch = recursive_get_loss(
            conv_features,
            c_node,
            target,
            mask & applicable_mask,
            level + 1,
            global_ce,
            correct_arr,
            total_arr,
            accuracy_tree["children"][i],
        )
        
        cross_entropy = cross_entropy + new_cross_entropy
        cluster_cost = cluster_cost + new_cluster_cost 
        separation_cost = separation_cost + new_separation_cost
        l1_cost = l1_cost + new_l1_cost
        num_parents_in_batch =  num_parents_in_batch + new_num_parents_in_batch

    del logits, min_distances

    return cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch

def construct_accuract_tree_rec(node):
    return {
        "int_location": node.int_location,
        "named_location": node.named_location,
        "correct": 0,
        "total": 0,
        "children": [
            construct_accuract_tree_rec(child) for child in node.all_child_nodes
        ]
    }

def construct_accuracy_tree(root):
    """
    This creates an object that can be used to keep track of the accuracy at each split of the tree.
    """
    return {
        "int_location": [],
        "children": [
            construct_accuract_tree_rec(child) for child in root.all_child_nodes
        ]}

def get_correspondence_proportions(model, cfg):
    props = [[] for i in range(4)]
    top_props = [[] for i in range(4)]
    for node in model.module.nodes_with_children:
        try:
            individual_min_indicies = node.max_tracker[1]
        except Exception as e:
            print("No max tracker found. Huh.")
            continue

        individual_min_indicies = torch.cat(individual_min_indicies, dim=0)
        if individual_min_indicies.shape[0] == 0:
            continue

        modes = torch.mode(individual_min_indicies, dim=0)[0]
        mode_truth = individual_min_indicies == modes
        # print(mode_truth.shape)
        mode_counts = torch.sum(mode_truth, dim=0)
        mode_counts_folded_by_10 = mode_counts.reshape((cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS,-1)).float()

        # if node.correlation_count > 0:
        #     print(node.correlation_table / node.correlation_count)

        props[len(node.int_location)].append(torch.mean(mode_counts / individual_min_indicies.shape[0]).item())
        top_props[len(node.int_location)].append(torch.sort(torch.mean(mode_counts_folded_by_10, dim=1), descending=True)[0] / individual_min_indicies.shape[0])
    
    props = torch.tensor([torch.mean(torch.tensor(prop)) for prop in props])
    a = [torch.stack(t) for t in top_props]
    top_props = torch.stack([torch.stack(t).float().mean(dim=0) for t in top_props])
    top_3_props = top_props[:, :3].mean(dim=1)

    return props, top_props, top_3_props

def _train_or_test(
    model,
    dataloader,
    global_ce,
    parallel_mode,
    run,
    optimizer=None,
    coefs = None,
    log=print,
    batch_mult = 1,
    cfg=None,
):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    
    is_train = optimizer is not None
    
    print(is_train) 

    if not is_train:
        model.eval()
    else:
        model.train()
        
    start = time.time()
    
    n_batches = 0

    correct_arr = np.zeros(len(model.module.levels)) # This is the number of correct predictions at each level
    total_arr = np.zeros(len(model.module.levels)) # This is the total number of predictions at each level
    
    if model.module.mode == Mode.MULTIMODAL:
        # I can't be bothered to implement all_child_nodes for the multimodal model. This will work, though it's gross. 
        accuracy_tree = construct_accuracy_tree(model.module.genetic_hierarchical_ppnet.root) 
    else:
        accuracy_tree = construct_accuracy_tree(model.module.root)

    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_l1 = 0
    total_correspondence_cost = 0
    
    try:
        for node in model.module.nodes_with_children:
            for el1, el2 in node.max_tracker:
                del el1, el2
            
            node.max_tracker = ([], []) 
    except:
        pass

    if parallel_mode:
        total_probabalistic_correct_count = torch.zeros((2,4))
    else:
        total_probabalistic_correct_count = torch.zeros(4)
    total_probabilistic_total_count = torch.zeros(4)

    for i, ((genetics, image), (label, flat_label)) in enumerate(dataloader): 
        if model.module.mode == Mode.GENETIC:
            input = genetics.to("cuda")
        elif model.module.mode == Mode.IMAGE:
            input = image.to("cuda")
        else:
            input = (genetics.to("cuda"), image.to("cuda"))
            # raise NotImplementedError("Multimodal not implemented")

        target = label.type(torch.LongTensor)
        target = target.to("cuda")

        batch_size = len(target)

        cross_entropy = 0
        cluster_cost = 0
        separation_cost = 0
        l1 = 0

        num_parents_in_batch = 0

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            conv_features = model.module.conv_features(input)
            
            # compute the loss
            if model.module.mode == Mode.MULTIMODAL:
                # Freeze last layer multi
                cross_entropy, cluster_cost, separation_cost, l1, num_parents_in_batch, correspondence_cost_summed, correspondence_cost_count, orthogonality_cost = recursive_get_loss_multi( 
                    conv_features=conv_features,
                    node=model.module.root,
                    target=target,
                    prev_mask=torch.ones(batch_size, dtype=bool).to("cuda"),
                    level=0,
                    global_ce=global_ce,
                    correct_arr=correct_arr,
                    total_arr=total_arr,
                    accuracy_tree=accuracy_tree,
                    parallel_mode=parallel_mode,
                    cfg=cfg
                )

                correspondence_cost = correspondence_cost_summed / correspondence_cost_count
            else:
                cross_entropy, cluster_cost, separation_cost, l1, num_parents_in_batch = recursive_get_loss(
                    conv_features=conv_features,
                    node=model.module.root,
                    target=target,
                    prev_mask=torch.ones(batch_size, dtype=bool).to("cuda"),
                    level=0,
                    global_ce=global_ce,
                    correct_arr=correct_arr,
                    total_arr=total_arr,
                    accuracy_tree=accuracy_tree
                )
            
            # compute the conditional accuracies
            if parallel_mode:
                genetic_probabalistic_correct_counts, genetic_probabilistic_total_counts = get_conditional_accuracies_flat(
                    model=model.module.genetic_hierarchical_ppnet,
                    target=flat_label.cuda(),
                    dataset=dataloader.dataset
                )
                image_probabalistic_correct_counts, _ = get_conditional_accuracies_flat(
                    model=model.module.image_hierarchical_ppnet,
                    target=flat_label.cuda(),
                    dataset=dataloader.dataset
                ) 
                # print("-------------------------------------")
                # total_probabalistic_correct_count += torch.stack([genetic_probabalistic_correct_counts, image_probabalistic_correct_counts])
                # print(total_probabilistic_total_count)
                # total_probabilistic_total_count += genetic_probabilistic_total_counts
                # print(total_probabalistic_correct_count)
                # print(genetic_probabilistic_total_counts)
                # print(total_probabilistic_total_count)
                # print("-------------------------------------")

                for node in model.module.nodes_with_children:
                    del node.genetic_tree_node._logits, node.image_tree_node._logits
                    del node.genetic_tree_node.probs, node.image_tree_node.probs
            else:
                probabalistic_correct_counts, probabilistic_total_counts = get_conditional_accuracies_flat(
                    model=model.module,
                    target=flat_label.cuda(),
                    dataset=dataloader.dataset
                )
                total_probabalistic_correct_count += probabalistic_correct_counts
                total_probabilistic_total_count += probabilistic_total_counts 


        if is_train:
            loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
            
            if model.module.mode == Mode.MULTIMODAL:
                loss += coefs['correspondence'] * correspondence_cost
                loss += (coefs['orthogonality'] * orthogonality_cost).sum()
                          
            loss.backward()
            
            if (i+1) % batch_mult == 0:
                optimizer.step()
                optimizer.zero_grad()

        n_batches += 1

        total_cross_entropy += cross_entropy
        total_cluster_cost += cluster_cost / num_parents_in_batch
        total_separation_cost += separation_cost / num_parents_in_batch
        total_l1 += l1

        if model.module.mode == Mode.MULTIMODAL:
            total_correspondence_cost += correspondence_cost
            del correspondence_cost
        # total_noise_cross_ent += noise_cross_ent.item() if CEDA else 0
            
        del input, target, conv_features

        if i % 512 == 0:
            log(f"[{i}] VRAM Usage: {torch.cuda.memory_reserved()/1024/1024/1024:.2f}GB")

    props, top_props, top_3_props = get_correspondence_proportions(model, cfg)

    mode_str = "train" if is_train else "val"

    batch = {
        f"{mode_str}-cross_ent": total_cross_entropy / n_batches,
        f"{mode_str}-cluster": total_cluster_cost / n_batches,
        f"{mode_str}-separation": total_separation_cost / n_batches,
        f"{mode_str}-l1": total_l1 / n_batches,
        f"{mode_str}-correspondence": total_correspondence_cost / n_batches,
        f"{mode_str}-orthogonality-genetic": orthogonality_cost[0].item(),
        f"{mode_str}-orthogonality-image": orthogonality_cost[1].item(),
    }

    for i, level in enumerate(["base", "order", "family", "genus"]): 
        if parallel_mode:
            batch[f"{mode_str}-genetic-{level}-conditional-prob-accuracy"] = total_probabalistic_correct_count[0,i] / total_probabilistic_total_count[i]
            batch[f"{mode_str}-image-{level}-conditional-prob-accuracy"] = total_probabalistic_correct_count[1,i] / total_probabilistic_total_count[i]
        else:
            batch[f"{mode_str}-{level}-conditional-prob-accuracy"] = total_probabalistic_correct_count[i] / total_probabilistic_total_count[i]

        batch[f"{mode_str}-{level}-mean-top-3-correspondence-agreement"] = top_3_props[i].item()

    run.log(batch, commit=False)
    log(format_dictionary_nicely_for_printing(batch))

    # If is parallel mode
    if parallel_mode:
        overall_accuracy = total_probabalistic_correct_count[1,0] / total_probabilistic_total_count[0]
    else:
        overall_accuracy = total_probabalistic_correct_count[0] / total_probabilistic_total_count[0]
    return overall_accuracy

def train(
    model, 
    dataloader, 
    optimizer, 
    coefs, 
    parallel_mode, 
    run,
    global_ce=True, 
    cfg=None,
    log=print,
): 

    assert(optimizer is not None)
    log('train')
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        global_ce=global_ce,
        parallel_mode=parallel_mode,
        optimizer=optimizer,
        coefs=coefs,
        log=log,
        cfg=cfg,
        run=run
    )

def valid(
    model, 
    dataloader, 
    run,
    coefs = None, 
    parallel_mode=False, 
    global_ce=True, 
    cfg=None,
    log=print,
    ):

    log('valid')
    return _train_or_test(
        model=model, 
        dataloader=dataloader, 
        parallel_mode=parallel_mode, 
        global_ce = global_ce,
        optimizer=None, 
        coefs=coefs,
        log=log,
        cfg=cfg,
        run=run
    )

def test(
    model, 
    dataloader, 
    run,
    coefs = None, 
    parallel_mode=False, 
    global_ce=True, 
    cfg=None,
    log=print 
):

    return _train_or_test(
        model=model, 
        dataloader=dataloader, 
        parallel_mode=parallel_mode, 
        global_ce=global_ce, 
        optimizer=None, 
        coefs = coefs,
        log=log, 
        cfg=cfg,
        run=run
    )

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
    

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True    

    prototype_vecs = model.module.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = True

    layers = model.module.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = False

    for p in model.module.get_last_layer_multi_parameters():
        p.requires_grad = False


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False

    for p in model.module.get_prototype_parameters():
        p.requires_grad = False
    
    for l in model.module.get_last_layer_parameters():
        l.requires_grad = True
    
# joint opts

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True

    prototype_vecs = model.module.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = True

    layers = model.module.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = True

    if model.module.mode == Mode.MULTIMODAL:
        for p in model.module.get_last_layer_multi_parameters():
            p.requires_grad = False
    
def multi_last_layer(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False

    prototype_vecs = model.module.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = False

    layers = model.module.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = False

    if model.module.mode == Mode.MULTIMODAL:
        for p in model.module.get_last_layer_multi_parameters():
            p.requires_grad = True
