import time
import torch
import numpy as np

from torchvision.transforms.functional import normalize

from model.node import Node


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def CE(logits,target):
     # manual definition of the cross entropy for a target which is a probability distribution    
     probs = torch.nn.functional.softmax(logits, 1)    
     return torch.sum(torch.sum(- target * torch.log(probs))) 

def get_cluster_and_sep_cost(min_distances, target, num_classes):
    target_one_hot = torch.zeros(target.size(0), num_classes).cuda()
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
        1 - torch.t(node.prototype_class_identity).cuda()
    )
    l1 = torch.linalg.vector_norm(
        node.last_layer.weight * l1_mask, ord=1
    )
    
    return l1

def get_multi_last_layer_l1_cost(node):
    l1_mask = (
        1- torch.t(node.logit_class_identity).cuda()
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
    node.accu_probs = None

def get_conditional_prob_accuracies(
        conv_features,
        model,
        target
):
    """
    This returns the predicted classes for each input using the conditional probability method.
    """
    recursive_put_accuracy_probs(
        conv_features,
        model.root,
        target,
        0,
        target
    )

    cpu_target = target.cpu()

    level_dict = {}

    for node in model.nodes_with_children:
        if not level_dict.get(len(node.int_location), False):
            level_dict[len(node.int_location)] = []
        
        level_dict[len(node.int_location)].append(node)
    
    agreements = []
    for level in level_dict:
        node_probs = []
        node_labels = []
        for node in level_dict[level]:
            node_probs.append(node.accu_probs)
            for i in range(node.accu_probs.shape[1]):
                node_labels.append(torch.tensor(node.int_location + [i + 1]))
    
        labels = torch.stack(node_labels,dim=1)
        probs = torch.cat(node_probs, dim=1)

        best_indicies = torch.argmax(probs, dim=1).cpu()
        best_labels = labels[:, best_indicies].permute(1, 0)

        diff = best_labels - cpu_target[:, :best_labels.shape[1]]
        agreement = (diff == 0).all(dim=1)
        agreements.append(agreement)

    agreements = torch.stack(agreements, dim=1)
    # This is gross, I know. I'm sorry.
    lowest_level_classified = torch.argmax(
        (torch.cat((cpu_target, torch.zeros((cpu_target.shape[0], 1))), dim=1) == 0).int(), dim=1
    ) - 1
    # Index into aggreements with lowest_level_classified
    agreements = agreements[range(agreements.shape[0]), lowest_level_classified]

    for node in model.nodes_with_children:
        clear_accu_probs(node)

    return agreements.sum().item(), len(agreements)

def recursive_put_accuracy_probs(
    conv_features,
    node,
    target,
    level,
    location_info,
    scale=1
):
    """
    This puts the softmax probabilities for each class into the node object. It scales the probabilities by the previous node's probability.
    """
    node.accu_probs = torch.nn.functional.softmax(node(conv_features)[0], dim=1) * scale
    
    for c_node in node.child_nodes:
        i = c_node.int_location[-1] - 1
        recursive_put_accuracy_probs(
            conv_features,
            c_node,
            target,
            level + 1,
            location_info,
            scale=node.accu_probs[:, i].unsqueeze(1)
        )

def recursive_get_loss_multi(
        conv_features,
        node,
        target,
        prev_mask,
        level,
        correct_arr,
        total_arr,
        accuracy_tree
    ):
    """
    This is the same as recursive get loss, but uses the multi logits
    """
    logits, (genetic_min_distances, image_min_distances) = node(conv_features)
    
    # Mask out unclassified examples
    mask = target[:,level] > 0
    # Mask out examples that don't belong to the current node
    mask = prev_mask & mask
    num_parents_in_batch = 1

    if mask.sum() == 0:
        return 0, 0, 0, 0, 0, 0

    cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)
    
    genetic_cluster_cost, genetic_separation_cost = get_cluster_and_sep_cost(
        genetic_min_distances[mask], target[mask][:, level], logits.size(1))
    image_cluster_cost, image_separation_cost = get_cluster_and_sep_cost(
        image_min_distances[mask], target[mask][:, level], logits.size(1))

    wrapped_genetic_min_distances = genetic_min_distances[mask].view(
        -1, genetic_min_distances.shape[1] // node.prototype_ratio, node.prototype_ratio
    )
    repeated_image_min_distances_along_the_third_axis = image_min_distances[mask].unsqueeze(2).expand(-1, -1, node.prototype_ratio)
    
    correspondence_cost = torch.mean(
        torch.min(
            (wrapped_genetic_min_distances - repeated_image_min_distances_along_the_third_axis) ** 2,
            dim=2
        )[0]
    )

    del wrapped_genetic_min_distances, repeated_image_min_distances_along_the_third_axis

    cluster_cost = genetic_cluster_cost + image_cluster_cost
    separation_cost = genetic_separation_cost + image_separation_cost

    genetic_l1_cost = get_l1_cost(node.genetic_tree_node)
    image_l1_cost = get_l1_cost(node.image_tree_node)
    multi_layer_l1_cost = get_multi_last_layer_l1_cost(node)

    l1_cost = genetic_l1_cost + image_l1_cost + multi_layer_l1_cost

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

        new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch, new_correspondence_cost = recursive_get_loss_multi(
            conv_features,
            c_node,
            target,
            mask & applicable_mask,
            level + 1,
            correct_arr,
            total_arr,
            accuracy_tree["children"][i],
        )
        
        cross_entropy = cross_entropy + new_cross_entropy
        cluster_cost = cluster_cost + new_cluster_cost 
        separation_cost = separation_cost + new_separation_cost
        l1_cost = l1_cost + new_l1_cost
        num_parents_in_batch =  num_parents_in_batch + new_num_parents_in_batch
        correspondence_cost = correspondence_cost + new_correspondence_cost

        del applicable_mask, new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch

    del logits, genetic_min_distances, image_min_distances, predicted, correct

    return cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch, correspondence_cost


def recursive_get_loss(conv_features, node, target, prev_mask, level, correct_arr, total_arr, accuracy_tree):
    """
    conv_features: The output of the convolutional layers
    Node: the current node in the tree
    target: the target labels
    prev_mask: the mask of the previous node
    level: the current level in the tree
    correct_tuple: tuple of correct prediction counts by level
    total_tuple: tuple of total prediction counts by level
    accuracy_tree: object to keep track of accuracy at each split of the tree
    """
    logits, min_distances = node(conv_features)
    
    # Mask out unclassified examples
    mask = target[:,level] > 0
    # Mask out examples that don't belong to the current node
    mask = prev_mask & mask
    num_parents_in_batch = 1

    if mask.sum() == 0:
        return 0, 0, 0, 0, 0

    cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)
    cluster_cost, separation_cost = get_cluster_and_sep_cost(
        min_distances[mask], target[mask][:, level], logits.size(1))

    l1_cost = get_l1_cost(node)

    # Update correct and total counts
    _, predicted = torch.max(logits[mask], 1)
    correct = predicted == (target[mask][:, level] - 1)
    # if level >= 1:
    #     print(level, correct, predicted, target[mask][:, level] - 1)
    correct_arr[level] += correct.sum().item()
    total_arr[level] += len(predicted)

    for i in range(logits.shape[1]):
        class_mask = (target[mask][:, level] - 1) == i
        class_correct = correct[class_mask]
        accuracy_tree["children"][i]["correct"] += class_correct.sum()
        accuracy_tree["children"][i]["total"] += class_mask.sum()
        
    for c_node in node.child_nodes:
        i = c_node.int_location[-1] - 1

        c_logits, c_min_distances = c_node(conv_features)
        applicable_mask = target[:,level] - 1 == i
        
        if applicable_mask.sum() == 0:
            continue

        new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch = recursive_get_loss(
            conv_features,
            c_node,
            target,
            mask & applicable_mask,
            level + 1,
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

def _train_or_test(model, dataloader, optimizer=None, coefs = None, class_specific=False, log=print, warm_up = False, CEDA = False, batch_mult = 1, class_acc = False):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    
    if not is_train: 
        model.eval()
        
    start = time.time()
    
    n_examples = 0
    n_batches = 0

    correct_arr = np.zeros(len(model.module.levels)) # This is the number of correct predictions at each level
    total_arr = np.zeros(len(model.module.levels)) # This is the total number of predictions at each level
    
    if model.module.mode == 3:
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
        if model.module.mode == 1:
            input = genetics.cuda()
        elif model.module.mode == 2:
            input = image.cuda()
        else:
            input = (genetics.cuda(), image.cuda())
            # raise NotImplementedError("Multimodal not implemented")
        
        target = label.type(torch.LongTensor)
        target = target.cuda()

        batch_size = len(target)
        batch_start = time.time()   

        cross_entropy = 0
        cluster_cost = 0
        separation_cost = 0
        l1 = 0

        total_probabalistic_correct_count = 0
        total_probabilistic_total_count = 0

        num_parents_in_batch = 0

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()

        with grad_req:
            conv_features = model.module.conv_features(input)
            if model.module.mode == 3:
                # Freeze last layer multi
                cross_entropy, cluster_cost, separation_cost, l1, num_parents_in_batch, correspondence_cost = recursive_get_loss_multi( 
                    conv_features=conv_features,
                    node=model.module.root,
                    target=target,
                    prev_mask=torch.ones(batch_size, dtype=bool).cuda(),
                    level=0,
                    correct_arr=correct_arr,
                    total_arr=total_arr,
                    accuracy_tree=accuracy_tree
                )
            else:
                cross_entropy, cluster_cost, separation_cost, l1, num_parents_in_batch =recursive_get_loss(
                    conv_features=conv_features,
                    node=model.module.root,
                    target=target,
                    prev_mask=torch.ones(batch_size, dtype=bool).cuda(),
                    level=0,
                    correct_arr=correct_arr,
                    total_arr=total_arr,
                    accuracy_tree=accuracy_tree
                )
            
            probabalistic_correct_count, probabilistic_total_count = get_conditional_prob_accuracies(
                conv_features=conv_features,
                model=model.module,
                target=target
            )
            total_probabalistic_correct_count += probabalistic_correct_count
            total_probabilistic_total_count += probabilistic_total_count

            # TODO - Maybe do this weird warm up BS

                # if warm_up:
                #     if node.name == "root":
                #         cross_entropy += torch.nn.functional.cross_entropy(node_logits,node_y) 
                #         cluster_cost_, separation_cost_ = auxiliary_costs(node_y,node.num_prototypes_per_class,node.num_children(),node.prototype_shape,node.min_distances[children_idx,:])                                    
                #         cluster_cost += cluster_cost_
                #         separation_cost += separation_cost_  
                #         l1 += elastic_net_reg(model,node)            
                # else:
                #     cross_entropy += torch.nn.functional.cross_entropy(node_logits,node_y) * len(node_y) / dataloader.batch_size if len(node_y) > 0 else 0                    
                                
                #     cluster_cost_, separation_cost_ = auxiliary_costs(node_y,node.num_prototypes_per_class,node.num_children(),node.prototype_shape,node.min_distances[children_idx,:])                                    
                #     cluster_cost += cluster_cost_
                #     separation_cost += separation_cost_               
                #     l1 += elastic_net_reg(model, node)

            # preds_root, preds_joint = model.module.get_joint_distribution()
            
            # # evaluation statistics
            # _, coarse_predicted = torch.max(preds_root.data, 1)
            # _, fine_predicted = torch.max(preds_joint.data, 1)
            
            # n_examples += target.size(0)
            # coarse_correct = coarse_predicted == root_y
            # fine_correct = fine_predicted == target
            # n_coarse_correct += coarse_correct.sum().item()
            # n_fine_correct += fine_correct.sum().item()    

            
            # if class_acc:
            #     for j in range(len(target)):
            #         coarse_class_correct[root_y[j]] += (1 if coarse_correct[j] else 0)
            #         coarse_class_total[root_y[j]] += 1
            #         fine_class_correct[target[j]] += (1 if fine_correct[j] else 0)
            #         fine_class_total[target[j]] += 1

        if is_train:          
            loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
            
            if model.module.mode == 3:
                loss += coefs['correspondence'] * correspondence_cost
                          
            loss.backward()
            
            # if CEDA:
            #     noise = torch.stack([normalize(torch.rand((3,32,32)), mean, std) for n in range(batch_size)]).cuda()
            #     _ = model(noise) 
            #     for node in model.module.root.nodes_with_children():
            #         noise_cross_ent += 1 * CE(node.logits,node.unif) # 1/10
            #     noise_cross_ent.backward()

            
            # optimizer.step()
            if (i+1) % batch_mult == 0:
                optimizer.step()
                optimizer.zero_grad()

        n_batches += 1        

        total_cross_entropy += cross_entropy
        total_cluster_cost += cluster_cost / num_parents_in_batch
        total_separation_cost += separation_cost / num_parents_in_batch
        total_l1 += l1

        if model.module.mode == 3:
            total_correspondence_cost += correspondence_cost
        # total_noise_cross_ent += noise_cross_ent.item() if CEDA else 0
            
        del input
        del target

        batch_end = time.time()

        if i%256 == 0:
            log(f"[{i}] Memory: {torch.cuda.memory_reserved()/1024/1024/1024:.2f}GB")
            
    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))

    log("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    log("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    log("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    # log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcluster: \t{0:.2f}'.format(total_cluster_cost / n_batches))
    log('\tseparation: \t{0:.2f}'.format(total_separation_cost / n_batches))
    log('\tl1: \t{0:.2f}'.format(total_l1 / n_batches))
    if model.module.mode == 3:
        log('\tcorrespondence: \t{0:.2f}'.format(total_correspondence_cost / n_batches))
    log('\tprobabilistic accuracy: \t{0:.5f}'.format(total_probabalistic_correct_count / total_probabilistic_total_count)) 
    for i, level in enumerate(model.module.levels):
        log(f'\t{level + " level accuracy:":<23} \t{correct_arr[i] / total_arr[i]:.5f} ({int(total_arr[i])} samples)')

    # print_accuracy_tree(accuracy_tree, log)

    return correct_arr / total_arr

def train(model, dataloader, optimizer, coefs, class_specific=False, log=print, warm_up = False):
    assert(optimizer is not None)
    log('train')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs,
                          class_specific=class_specific, log=log, warm_up = warm_up, CEDA=coefs['CEDA'])

def valid(model, dataloader, coefs = None, class_specific=False, log=print):
    log('valid')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs=coefs,
                          class_specific=class_specific, log=log)

def test(model, dataloader, coefs = None, class_specific=False, log=print, class_acc = False):
    #log('test')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs = coefs,
                      class_specific=class_specific, log=log, class_acc=class_acc)

def make_one_hot(target, target_one_hot):
    target_copy = torch.LongTensor(len(target))
    target_copy.copy_(target)
    target_copy = target_copy.view(-1,1).cuda()
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target_copy, value=1.)


def auxiliary_costs(label,num_prototypes_per_class,num_classes,prototype_shape,min_distances):
    if label.size(0) == 0: 
        return 0, 0
    
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    target = label.cuda()
    target_one_hot = torch.zeros(target.size(0), num_classes)
    target_one_hot = target_one_hot.cuda()                
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

    if model.module.mode == 3:
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

    if model.module.mode == 3:
        for p in model.module.get_last_layer_multi_parameters():
            p.requires_grad = True

    log('multi last layer')


# last layer opts

def last_layers(model, log=print):
    raise NotImplementedError("Last layers not implemented")
    pass
    # for p in model.module.features.parameters():
    #     p.requires_grad = False
    # for p in model.module.add_on_layers.parameters():
    #     p.requires_grad = False
    # for node in model.module.root.nodes_with_children():
    #     vecs = getattr(model.module,node.name + "_prototype_vectors")
    #     vecs.requires_grad = False
    #     layer = getattr(model.module,node.name + "_layer")
    #     for p in layer.parameters():
    #         p.requires_grad = True                  
    # log('last layers')

