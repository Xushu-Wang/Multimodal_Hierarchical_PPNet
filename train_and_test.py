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
    l1_mask = (1 - torch.t(node.prototype_class_identity))
    l1_mask = l1_mask.cuda()
    l1 = (node.last_layer.weight * l1_mask).norm(p=1)
    
    return l1

def recursive_get_loss(node, logits, min_distances, child_output, target, prev_mask, level):
    # Mask out unclassified examples
    mask = target[:,level] > 0
    mask = prev_mask & mask
    num_parents_in_batch = 1

    # TODO - I am suspicious that we do this twice. Hmmmm.
    cross_entropy = torch.nn.functional.cross_entropy(logits[mask], target[mask][:, level] - 1)
    cluster_cost, separation_cost = get_cluster_and_sep_cost(
        min_distances[mask], target[mask][:, level], logits.size(1))

    l1_cost = get_l1_cost(node)
    for i, (c_logits, c_min_distances, c_child_output) in enumerate(child_output):
        # Mask out irrelevant examples
        correct_mask = (target[:,level+1] - 1) == i
        new_mask = mask & correct_mask

        if new_mask.sum() == 0:
            continue

        new_cross_entropy, new_cluster_cost, new_separation_cost, new_l1_cost, new_num_parents_in_batch = recursive_get_loss(
            node.child_nodes[i], c_logits, c_min_distances, c_child_output, target, new_mask, level + 1)
        
        cross_entropy += new_cross_entropy
        cluster_cost += new_cluster_cost
        separation_cost += new_separation_cost
        l1_cost += new_l1_cost
        num_parents_in_batch += new_num_parents_in_batch

    return cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch

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
    n_coarse_correct = 0
    n_fine_correct = 0
    n_batches = 0
    
    total_cross_entropy = 0
    total_noise_cross_ent = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_l1 = 0
        
    for i, ((genetics, image), label) in enumerate(dataloader):
        input = image.cuda()
        target = label.type(torch.LongTensor).cuda()

        batch_size = len(target)
        batch_start = time.time()   

        cross_entropy = 0
        cluster_cost = 0
        separation_cost = 0
        l1 = 0

        num_parents_in_batch = 0

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            logits, min_distances, child_output = model(input)
            cross_entropy, cluster_cost, separation_cost, l1_cost, num_parents_in_batch = recursive_get_loss(
                node=model.module.root,
                logits=logits,
                min_distances=min_distances,
                child_output=child_output,
                target=target,
                prev_mask=torch.ones(batch_size, dtype=bool).cuda(),
                level=0
            )

            print(cluster_cost, separation_cost, l1_cost)

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

            del logits
            del min_distances
            del child_output

        if is_train:          
            loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                          
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
        # total_noise_cross_ent += noise_cross_ent.item() if CEDA else 0
            
        del input
        del target

        batch_end = time.time()

        print(f"Batch {i} took {batch_end - batch_start:.2f} seconds")
        
    end = time.time()

    log('\ttime: \t{0:.2f}'.format(end -  start))

    log('\tcross ent: \t{0:.2f}'.format(total_cross_entropy / n_batches))
    # log('\tnoise cross ent: \t{0:.2f}'.format(total_noise_cross_ent / n_batches))
    log('\tcluster: \t{0:.2f}'.format(total_cluster_cost / n_batches))
    log('\tseparation: \t{0:.2f}'.format(total_separation_cost / n_batches))
    log('\tl1: \t{0:.2f}'.format(total_l1 / n_batches))        

    return 0,0    

def train(model, dataloader, optimizer, coefs, class_specific=False, log=print, warm_up = False):
    assert(optimizer is not None)
    log('train')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, coefs=coefs,
                          class_specific=class_specific, log=log, warm_up = warm_up, CEDA=coefs['CEDA'])

def valid(model, dataloader, label2name, coefs = None, class_specific=False, log=print):
    log('valid')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs=coefs,
                          class_specific=class_specific, log=log)

def test(model, dataloader, label2name, coefs = None, class_specific=False, log=print, class_acc = False):
    #log('test')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None, coefs = coefs,
                      class_specific=class_specific, log=log, class_acc=class_acc)

def OOD_test(model, dataloader, label2name, IDroot, OODroot, args = None, class_specific=False, log=print, class_acc = True):
    #log('test')
    return _OOD_test(model=model, dataloader=dataloader, label2name=label2name,IDroot=IDroot, OODroot = OODroot, optimizer=None, 
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
    

def get_l1(model,node, mask = True):

    if mask:
        identity = torch.eye(node.num_children()).cuda()
        repeated_identity = identity.unsqueeze(2).repeat(1,1,node.num_prototypes_per_class).\
                                view(node.num_children(), -1)
        l1_mask = 1 - repeated_identity
        weight = getattr(getattr(model.module,node.name + "_layer"),'weight') * l1_mask

    else:
        weight = getattr(getattr(model.module,node.name + "_layer"),'weight')
    
    return weight.norm(p=1)    


def elastic_net_reg(model,node):
    # l1 reg on off-prototype weights, l2 on on-proto weights

    identity = torch.eye(node.num_children()).cuda()
    repeated_identity = identity.unsqueeze(2).repeat(1,1,node.num_prototypes_per_class).\
                            view(node.num_children(), -1)
    l1_mask = 1 - repeated_identity
    weight = getattr(getattr(model.module,node.name + "_layer"),'weight')
    off_weight = weight * l1_mask
    on_weight = weight * repeated_identity
    
    return off_weight.norm(p=1) + 1/10 * on_weight.norm(p=2)           

# warm only opts

def coarse_warm(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module, node.name + "_prototype_vectors")
            vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('coarse warm')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True    

    prototype_vecs = model.module.root.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = True

    layers = model.module.root.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = False            
    log('warm')
   
# up to protos opts

def coarse_up_to_protos(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module,node.name + "_prototype_vectors")
            vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('coarse up to protos')

def up_to_protos(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = True
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = False                     
    log('through protos')


# joint opts

def coarse_joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.root_prototype_vectors.requires_grad = True    
    for p in model.module.root_layer.parameters():
        p.requires_grad = True  
    for node in model.module.root.nodes_with_children():
        if node.name != "root":
            vecs = getattr(model.module,node.name + "_prototype_vectors")
            vecs.requires_grad = False
            layer = getattr(model.module,node.name + "_layer")
            for p in layer.parameters():
                p.requires_grad = False                     
    log('coarse joint')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = True
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = True                     
    log('joint')


# last layer opts

def last_layers(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    for node in model.module.root.nodes_with_children():
        vecs = getattr(model.module,node.name + "_prototype_vectors")
        vecs.requires_grad = False
        layer = getattr(model.module,node.name + "_layer")
        for p in layer.parameters():
            p.requires_grad = True                  
    log('last layers')


