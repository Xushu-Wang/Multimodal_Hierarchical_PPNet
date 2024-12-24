import time
import torch
import numpy as np
from pympler.tracker import SummaryTracker
from model.hierarchical_ppnet import Mode
from typing import Union, Tuple

tracker = SummaryTracker()

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def CE(logits,target):
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

def get_conditional_prob_accuracies(
    conv_features,
    model,
    target,
    global_ce=False,
    parallel_mode=False
):
    """
    This returns the predicted classes for each input using the conditional probability method.

    If global_ce is true, the cross entropy is calculated using the conditional probabilities of each class.
    """
    
    recursive_put_accuracy_probs(
        conv_features,
        model.root,
        target,
        0,
        parallel_mode,
        scale = (1,1) if parallel_mode else 1
    )
    

    cpu_target = target.cpu()

    level_dict = {}

    for node in model.nodes_with_children:
        if not level_dict.get(len(node.int_location), False):
            level_dict[len(node.int_location)] = []
        
        level_dict[len(node.int_location)].append(node)
    
    if parallel_mode:
        agreements = ([] , [])
    else:
        agreements = []

    cross_entropies = []
    for level in level_dict:
        if parallel_mode:
            node_probs = ([], [])
        else:
            node_probs = []

        node_labels = []
    
        best_probs = torch.zeros(cpu_target.shape[0]).cpu()
        for node in level_dict[level]:
            if parallel_mode:
                accu_probs = (node.accu_probs[0].cpu(), node.accu_probs[1].cpu())

                node_probs[0].append(accu_probs[0])
                node_probs[1].append(accu_probs[1])
            else:
                accu_probs = node.accu_probs.cpu()
                node_probs.append(accu_probs)

            child_count = node.accu_probs[0].shape[1] if parallel_mode else node.accu_probs.shape[1]

            for i in range(child_count):
                node_label = torch.tensor(node.int_location + [i + 1]).cpu()
                node_labels.append(node_label)

                # Find if this node corresponds to any samples, if so add them to best_probs
                # Get a mask of all rows of cpu_target[:, :len(node_label)] that match node_label
                # TODO - Move this up a level for a small speedup
                if global_ce:
                    if parallel_mode:
                        raise NotImplementedError("Parallel mode not implemented for global_ce")
                    mask = (cpu_target[:, :len(node_label)] == node_label).all(dim=1).cpu()
                    best_probs[mask] = accu_probs[mask, i]

        if parallel_mode:
            labels = torch.stack(node_labels,dim=1)

            genetic_probs = torch.cat(node_probs[0], dim=1)

            genetic_best_indicies = torch.argmax(genetic_probs, dim=1).cpu()
            genetic_best_labels = labels[:, genetic_best_indicies].permute(1, 0)

            genetic_diff = genetic_best_labels - cpu_target[:, :genetic_best_labels.shape[1]]
            genetic_agreement = (genetic_diff == 0).all(dim=1)
            agreements[0].append(genetic_agreement)

            image_probs = torch.cat(node_probs[1], dim=1)

            image_best_indicies = torch.argmax(image_probs, dim=1).cpu()
            image_best_labels = labels[:, image_best_indicies].permute(1, 0)

            image_diff = image_best_labels - cpu_target[:, :image_best_labels.shape[1]]
            image_agreement = (image_diff == 0).all(dim=1)
            agreements[1].append(image_agreement)

            del genetic_probs, genetic_best_indicies, genetic_best_labels, genetic_diff, genetic_agreement

            if global_ce:
                # Calculate cross entropy, but don't use torch cross entropy because we have already softmaxed the logits
                # Handle the case where the best_probs are 0
                raise NotImplementedError("Parallel mode not implemented for global_ce")
                best_probs[best_probs == 0] = 1e-10
                cross_entropy = -torch.log(best_probs)
                cross_entropies.append(cross_entropy)
        else:
            labels = torch.stack(node_labels,dim=1)
            probs = torch.cat(node_probs, dim=1)

            best_indicies = torch.argmax(probs, dim=1).cpu()
            best_labels = labels[:, best_indicies].permute(1, 0)

            diff = best_labels - cpu_target[:, :best_labels.shape[1]]
            agreement = (diff == 0).all(dim=1)
            agreements.append(agreement)

            if global_ce:
                # Calculate cross entropy, but don't use torch cross entropy because we have already softmaxed the logits
                # Handle the case where the best_probs are 0
                best_probs[best_probs == 0] = 1e-10
                cross_entropy = -torch.log(best_probs)
                cross_entropies.append(cross_entropy)

    if parallel_mode:
        genetic_agreements = torch.stack(agreements[0], dim=1)
        image_agreements = torch.stack(agreements[1], dim=1)
        
        if global_ce:
            raise NotImplementedError("Parallel mode not implemented for global_ce")
            cross_entropies = torch.stack(cross_entropies, dim=1)
        
        # This is gross, I know. I'm sorry.
        # Find the lowest level classified
        lowest_level_classified = torch.argmax(
            (torch.cat((cpu_target, torch.zeros((cpu_target.shape[0], 1))), dim=1) == 0).int(), dim=1
        ) - 1
        
        # Index into aggreements with lowest_level_classified
        genetic_agreements = genetic_agreements[range(genetic_agreements.shape[0]), lowest_level_classified]
        image_agreements = image_agreements[range(image_agreements.shape[0]), lowest_level_classified]

        if global_ce:
            raise NotImplementedError("Parallel mode not implemented for global_ce")
            cross_entropies = cross_entropies[range(cross_entropies.shape[0]), lowest_level_classified]
            total_cross_entropy = torch.mean(cross_entropies)
        else:
            total_cross_entropy = 0

        for node in model.nodes_with_children:
            clear_accu_probs(node)

        return torch.tensor([genetic_agreements.sum().item(), image_agreements.sum().item()]), len(genetic_agreements), total_cross_entropy
    else:
        agreements = torch.stack(agreements, dim=1)
        
        if global_ce:
            cross_entropies = torch.stack(cross_entropies, dim=1)
        
        # This is gross, I know. I'm sorry.
        # Find the lowest level classified
        lowest_level_classified = torch.argmax(
            (torch.cat((cpu_target, torch.zeros((cpu_target.shape[0], 1))), dim=1) == 0).int(), dim=1
        ) - 1
        
        # Index into aggreements with lowest_level_classified
        agreements = agreements[range(agreements.shape[0]), lowest_level_classified]

        if global_ce:
            cross_entropies = cross_entropies[range(cross_entropies.shape[0]), lowest_level_classified]
            total_cross_entropy = torch.mean(cross_entropies)
        else:
            total_cross_entropy = 0

        for node in model.nodes_with_children:
            clear_accu_probs(node)

        del labels, probs, best_indicies, best_labels, diff, agreement

        return agreements.sum().item(), len(agreements), total_cross_entropy

def recursive_put_accuracy_probs(
    conv_features,
    node,
    target,
    level,
    parallel_mode,
    scale: Union[int, Tuple[int, int]],
):
    """
    This puts the softmax probabilities for each class into the node object. It scales the probabilities by the previous node's probability.
    """
    # TODO - This is calculated twice! BAD! We can't cache it frgenetic_logits, image_logitsom the main_train_loop step, because that doesn't evaluate the whole tree. This should be run first, then it should cache the logits within the tree. The get_loss function should then use the logits from the tree.
    if parallel_mode:
        (genetic_logits, image_logits), _ = node(conv_features, get_middle_logits=parallel_mode)
        
        node.accu_probs = (
            torch.nn.functional.softmax(genetic_logits, dim=1) * scale[0],
            torch.nn.functional.softmax(image_logits, dim=1) * scale[1]
        )

        scale = (node.accu_probs[0][:, -1].unsqueeze(1), node.accu_probs[1][:, -1].unsqueeze(1))

        del genetic_logits, image_logits
    else:
        node.accu_probs = torch.nn.functional.softmax(node(conv_features)[0], dim=1) * scale
        scale = node.accu_probs[:, -1].unsqueeze(1)



    for c_node in node.child_nodes:
        i = c_node.int_location[-1] - 1
        recursive_put_accuracy_probs(
            conv_features,
            c_node,
            target,
            level + 1,
            parallel_mode=parallel_mode,
            scale=scale
        )

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
        parallel_mode
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

    if mask.sum() == 0:
        return 0, 0, 0, 0, 0, 0, 0

    if global_ce:
        cross_entropy = 0
    else:
        if parallel_mode:
            # If we are in parallel mode, we calculate each model's cross entropy separately. We ignore the last layer.
            genetic_logits, image_logits = logits
            genetic_cross_entropy = torch.nn.functional.cross_entropy(genetic_logits[mask], target[mask][:, level] - 1)
            image_cross_entropy = torch.nn.functional.cross_entropy(image_logits[mask], target[mask][:, level] - 1)
            cross_entropy = genetic_cross_entropy + image_cross_entropy
            del genetic_cross_entropy, image_cross_entropy, genetic_logits, image_logits
        else:
            # If we are not in parallel mode, we calculate the cross entropy using the multi logits
            cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)

    genetic_cluster_cost, genetic_separation_cost = get_cluster_and_sep_cost(
        genetic_min_distances[mask], target[mask][:, level], logits_size_1)
    image_cluster_cost, image_separation_cost = get_cluster_and_sep_cost(
        image_min_distances[mask], target[mask][:, level], logits_size_1)

    wrapped_genetic_min_distances = genetic_min_distances[mask].view(
        -1, genetic_min_distances.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = image_min_distances[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)
    
    # Calculate the total correspondence cost, minimum MSE between corresponding prototypes. We will later divide this by the number of comparisons made to get the average correspondence cost
    summed_correspondence_cost = torch.sum(
        torch.min(
            (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2,
            dim=2
        )[0]
    )
    summed_correspondence_cost_count = wrapped_genetic_min_distances.shape[0]

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis

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
        
        if applicable_mask.sum() == 0:
            continue

        new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_summed_correspondence_cost, new_summed_correspondence_cost_count = recursive_get_loss_multi(
            conv_features,
            c_node,
            target,
            mask & applicable_mask,
            level + 1,
            global_ce,
            correct_arr,
            total_arr,
            accuracy_tree["children"][i],
            parallel_mode
        )
        
        cross_entropy = cross_entropy + new_cross_entropy
        cluster_cost = cluster_cost + new_cluster_cost 
        separation_cost = separation_cost + new_separation_cost
        l1_cost = l1_cost + new_l1_cost
        num_parents_in_batch =  num_parents_in_batch + new_num_parents_in_batch
        summed_correspondence_cost = summed_correspondence_cost + new_summed_correspondence_cost
        summed_correspondence_cost_count = summed_correspondence_cost_count + new_summed_correspondence_cost_count

        del applicable_mask, new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_summed_correspondence_cost, new_summed_correspondence_cost_count

    del logits, genetic_min_distances, image_min_distances, predicted, correct, logits_size_1

    return cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch, summed_correspondence_cost, summed_correspondence_cost_count


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

def _train_or_test(
    model,
    dataloader,
    global_ce,
    parallel_mode,
    optimizer=None,
    coefs = None,
    log=print,
    batch_mult = 1):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    
    is_train = optimizer is not None
    
    if not is_train: 
        model.eval()
        
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
    # torch.autograd.set_detect_anomaly(True)
    
    for i, ((genetics, image), label) in enumerate(dataloader):
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

        if parallel_mode:
            total_probabalistic_correct_count = torch.zeros(2)
        else:
            total_probabalistic_correct_count = 0
        total_probabilistic_total_count = 0

        num_parents_in_batch = 0

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            conv_features = model.module.conv_features(input)
            
            if model.module.mode == Mode.MULTIMODAL:
                # Freeze last layer multi
                cross_entropy, cluster_cost, separation_cost, l1, num_parents_in_batch, summed_correspondence_cost, summed_correspondence_cost_count = recursive_get_loss_multi( 
                    conv_features=conv_features,
                    node=model.module.root,
                    target=target,
                    prev_mask=torch.ones(batch_size, dtype=bool).to("cuda"),
                    level=0,
                    global_ce=global_ce,
                    correct_arr=correct_arr,
                    total_arr=total_arr,
                    accuracy_tree=accuracy_tree,
                    parallel_mode=parallel_mode
                )
                correspondence_cost = summed_correspondence_cost / summed_correspondence_cost_count
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
            
            probabalistic_correct_count, probabilistic_total_count, global_cross_entropy = get_conditional_prob_accuracies(
                conv_features=conv_features,
                model=model.module,
                target=target,
                global_ce=global_ce,
                parallel_mode=parallel_mode,
            )
            # print(is_train, probabalistic_correct_count, probabilistic_total_count, total_probabilistic_total_count)

            if global_ce:
                cross_entropy = global_cross_entropy

            total_probabalistic_correct_count += probabalistic_correct_count
            total_probabilistic_total_count += probabilistic_total_count

            # TODO - Maybe warm up

        if is_train:          
            loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
            
            if model.module.mode == Mode.MULTIMODAL:
                loss += coefs['correspondence'] * correspondence_cost
                          
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
            
    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))

    train_or_test_string = 'train' if is_train else 'test'

    log("\t[%s]\ttorch.cuda.memory_allocated: %fGB"%(train_or_test_string,torch.cuda.memory_allocated(0)/1024/1024/1024))
    log("\t[%s]\ttorch.cuda.memory_reserved: %fGB"%(train_or_test_string,torch.cuda.memory_reserved(0)/1024/1024/1024))
    log("\t[%s]\ttorch.cuda.max_memory_reserved: %fGB"%(train_or_test_string, torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    log('\t[{0}]\tcross ent: \t{1:.2f}'.format(train_or_test_string, total_cross_entropy / n_batches))
    # log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\t[{0}]\tcluster: \t{1:.2f}'.format(train_or_test_string, total_cluster_cost / n_batches))
    log('\t[{0}]\tseparation: \t{1:.2f}'.format(train_or_test_string, total_separation_cost / n_batches))
    log('\t[{0}]\tl1: \t{1:.2f}'.format(train_or_test_string, total_l1 / n_batches))
    if model.module.mode == Mode.MULTIMODAL:
        log('\t[{0}]\tcorrespondence: \t{1:.2f}'.format(train_or_test_string, total_correspondence_cost / n_batches))
    
    if parallel_mode:
        log('\t[{0}]\tgenetic probabilistic accuracy: \t{1:.5f}'.format(train_or_test_string, total_probabalistic_correct_count[0] / total_probabilistic_total_count)) 
        log('\t[{0}]\timage probabilistic accuracy: \t{1:.5f}'.format(train_or_test_string, total_probabalistic_correct_count[1] / total_probabilistic_total_count)) 
    else:
        log('\t[{0}]\tprobabilistic accuracy: \t{1:.5f}'.format(train_or_test_string, total_probabalistic_correct_count / total_probabilistic_total_count)) 

    for i, level in enumerate(model.module.levels):
        log(f'\t[{train_or_test_string}]\t{level + " level accuracy:":<23} \t{correct_arr[i] / total_arr[i]:.5f} ({int(total_arr[i])} samples)')

    overall_accuracy = torch.min(total_probabalistic_correct_count / total_probabilistic_total_count) if parallel_mode else total_probabalistic_correct_count / total_probabilistic_total_count
    return overall_accuracy

def train(
    model, 
    dataloader, 
    optimizer, 
    coefs, 
    parallel_mode, 
    global_ce=True, 
    log=print
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
        log=log
    )

def valid(
    model, 
    dataloader, 
    coefs = None, 
    parallel_mode=False, 
    global_ce=True, 
    log=print
    ):

    log('valid')
    return _train_or_test(
        model=model, 
        dataloader=dataloader, 
        parallel_mode=parallel_mode, 
        global_ce = global_ce,
        optimizer=None, 
        coefs=coefs,
        log=log
    )

def test(
    model, 
    dataloader, 
    coefs = None, 
    parallel_mode=False, 
    global_ce=True, 
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
    log('warm')
   

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False

    for p in model.module.get_prototype_parameters():
        p.requires_grad = False
    
    for l in model.module.get_last_layer_parameters():
        l.requires_grad = True
    
    log('\tlast layer')

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
    
    log('joint')

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

    log('multi last layer')

