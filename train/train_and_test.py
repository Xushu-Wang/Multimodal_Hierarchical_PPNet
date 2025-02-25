import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from yacs.config import CfgNode
from model.hierarchical import Mode, HierProtoPNet
from model.multimodal import MultiHierProtoPNet
from typing import Union
from train.loss import Objective, get_cluster_and_sep_cost, get_l1_cost, get_ortho_cost, get_correspondence_loss_batched
from tqdm import tqdm
from enum import Enum

Model = Union[HierProtoPNet, MultiHierProtoPNet, torch.nn.DataParallel]

class OptimMode(Enum): 
    """
    Enumeration object to specify which parameters to train. 
    """
    WARM = 1 
    MULTI_LAST = 2 
    JOINT = 3 
    LAST = 4 

def warm_only(model: Model):
    if isinstance(model, torch.nn.DataParallel): 
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True    

    prototype_vecs = model.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = True

    layers = model.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = False

def last_only(model: Model):
    if isinstance(model, torch.nn.DataParallel): 
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False

    for p in model.get_prototype_parameters():
        p.requires_grad = False
    
    for l in model.get_last_layer_parameters():
        l.requires_grad = True
    
def joint(model: Model):
    if isinstance(model, torch.nn.DataParallel): 
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True

    prototype_vecs = model.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = True

    layers = model.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = True

def multi_last_layer(model: Model): 
    if isinstance(model, torch.nn.DataParallel): 
        model = model.module 
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False

    prototype_vecs = model.get_prototype_parameters()
    for p in prototype_vecs:
        p.requires_grad = False

    layers = model.get_last_layer_parameters()
    for l in layers:
        l.requires_grad = False

def train(
    model: Model, 
    dataloader, 
    optimizer: Optimizer, 
    cfg: CfgNode, 
    optim_mode: OptimMode,
    log = print
): 
    """
    Train wrapper function for all models. 
    It first sets specific parameter gradients = True depending on optim_mode 
    Then depending on the model mode, it calls the specific train functions
    """
    print("Trainin'")
    mode = Mode(model.mode) 

    match optim_mode: 
        case OptimMode.WARM: 
            warm_only(model)
        case OptimMode.MULTI_LAST: 
            multi_last_layer(model)
        case OptimMode.JOINT: 
            joint(model)
        case OptimMode.LAST: 
            last_only(model) 

    model.train()

    match mode: 
        case Mode.GENETIC: 
            return train_genetic(model, dataloader, optimizer, cfg, log) 
        case Mode.IMAGE: 
            return train_image(model, dataloader, optimizer, cfg, log) 
        case Mode.MULTIMODAL: 
            return train_multimodal(model, dataloader, optimizer, cfg, log) 


def train_genetic(model, dataloader, optimizer, cfg, log):  
    # compute the losses 
    total_obj = Objective(cfg)

    for (genetics, _), (label, _) in tqdm(dataloader): 

        input = genetics.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg)

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            logits, min_dist = node.forward(conv_features)

            n_classified += torch.sum(mask)

            # masked input and output
            m_label = label[mask][:, node.depth] - 1
            m_logits, m_min_dist = logits[mask], min_dist[mask]
            if len(m_logits) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label)  

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation = separation
            batch_obj.lasso += get_l1_cost(node) 

        # calculate cross entropy with unmasked logits 
        # softmax node.logits for each node to compute node.prob 
        model.conditional_normalize() 

        total_loss = batch_obj.total()  
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified
        total_obj += batch_obj


    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

def train_image(model, dataloader, optimizer, cfg, log): 
    total_obj = Objective(cfg)

    for (_, image), (label, _) in tqdm(dataloader): 

        input = image.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg)

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1)
            n_classified += torch.sum(mask)

            # masked input and output
            m_conv_features = conv_features[mask]
            m_label = label[mask][:, node.depth] - 1
            m_logits, m_min_dist = node.forward(m_conv_features) 
            if len(m_logits) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label)  

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation = separation
            batch_obj.lasso += get_l1_cost(node)

        total_loss = batch_obj.total()  
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified
        total_obj += batch_obj


    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

def train_multimodal(model, dataloader, optimizer, cfg, log): 
    total_obj = Objective(cfg)

    for (genetics, image), (label, _) in tqdm(dataloader):

        gen_input = genetics.cuda()
        img_input = image.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg)

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)

        # # track number of classifications across all nodes in this batch
        n_classified = 0 
        total_corr_count = 0


        for node in model.classifier_nodes:  
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1)
            n_classified += torch.sum(mask)

            # masked input and output
            # m_label = label[mask][:, node.depth] - 1

            # forward pass through the node 
            (gen_logits, img_logits), (gen_min_dist, img_min_dist) = node.forward(gen_conv_features, img_conv_features) 
            del gen_logits, img_logits, gen_min_dist, img_min_dist

        #     m_gen_logits = gen_logits[mask]
        #     m_img_logits = img_logits[mask]

        #     m_gen_min_dist = gen_min_dist[mask]
        #     m_img_min_dist = img_min_dist[mask]

        #     # cross entropy loss 
        #     if len(m_gen_logits) != 0: 
        #         gen_ce = F.cross_entropy(m_gen_logits, m_label)  
        #         img_ce = F.cross_entropy(m_img_logits, m_label)
        #         batch_obj.cross_entropy += gen_ce + img_ce

        #     # cluster and separation loss 
        #     gen_cluster, gen_separation = get_cluster_and_sep_cost(m_gen_min_dist, m_label, node.nclass)
        #     img_cluster, img_separation = get_cluster_and_sep_cost(m_img_min_dist, m_label, node.nclass)
        #     batch_obj.cluster += gen_cluster + img_cluster
        #     batch_obj.separation = gen_separation + img_separation

        #     # lasso loss
        #     batch_obj.lasso += get_l1_cost(node.gen_node) + get_l1_cost(node.img_node) 

        #     # correspondence loss 
        #     corr_sum, corr_count = get_correspondence_loss_batched(m_gen_min_dist, m_img_min_dist, node)
        #     batch_obj.correspondence += corr_sum
        #     total_corr_count += corr_count

        #     # orthogonality loss 
        #     batch_obj.gen_orthogonality = get_ortho_cost(node.gen_node)
        #     batch_obj.img_orthogonality = get_ortho_cost(node.img_node)

        #     del gen_logits, img_logits, m_gen_logits, m_img_logits, gen_min_dist, img_min_dist, m_gen_min_dist, m_img_min_dist

        # batch_obj.correspondence /= total_corr_count
        # total_loss = batch_obj.total()  
        # total_loss.backward()
        # optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        # batch_obj.cluster /= n_classified
        # batch_obj.separation /= n_classified 

        # add batch objective values to total objective values for logging
        # total_obj += batch_obj

        # Free scoped tensors
        del gen_conv_features, img_conv_features
        # for node in model.classifier_nodes:
        #     node.clear_cache()

        # Print torch memory usage
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")
        torch.cuda.empty_cache()

    # normalize the losses after running through entire dataset
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

def test(
    model: Model, 
    dataloader: DataLoader, 
    cfg: CfgNode, 
    log = print
): 
    print("Validatin'")
    mode = Mode(model.mode) 
    model.eval()
    match mode: 
        case Mode.GENETIC: 
            return test_genetic(model, dataloader, cfg, log) 
        case Mode.IMAGE: 
            return test_image(model, dataloader, cfg, log) 
        case Mode.MULTIMODAL: 
            return test_multimodal(model, dataloader, cfg, log) 

def test_genetic(model, dataloader, cfg, log): 
    # compute the loss? 
    total_obj = Objective(cfg, "test")

    for (genetics, _), (label, _) in tqdm(dataloader): 

        input = genetics.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg, "test")

        with torch.no_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1)
            n_classified += len(mask)

            # masked input and output
            m_conv_features = conv_features[mask]
            m_label = label[mask][:, node.depth] - 1
            m_logits, m_min_dist = node.forward(m_conv_features) 
            if len(m_logits) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label)  

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation = separation
            batch_obj.lasso += get_l1_cost(node)

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified
        total_obj += batch_obj

    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

def test_image(model, dataloader, cfg, log): 
    total_obj = Objective(cfg, "test")

    for (_, image), (label, _) in tqdm(dataloader): 

        input = image.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg, "test")

        with torch.no_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1)
            n_classified += len(mask)

            # masked input and output
            m_conv_features = conv_features[mask]
            m_label = label[mask][:, node.depth] - 1
            m_logits, m_min_dist = node.forward(m_conv_features) 
            if len(m_logits) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label)  

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation = separation
            batch_obj.lasso += get_l1_cost(node)

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified
        total_obj += batch_obj

        # Free scoped tensors
        del gen_conv_features, img_conv_features, m_gen_logits, m_img_logits, m_gen_min_dist, m_img_min_dist

        # Print torch memory usage
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

def test_multimodal(model, dataloader, cfg, log): 
    return
    total_obj = Objective(cfg)

    for (genetics, image), (label, _) in tqdm(dataloader): 
        gen_input = genetics.cuda()
        img_input = image.cuda()
        label = label.cuda()
        batch_obj = Objective(cfg)

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 
        total_corr_count = 0

        for node in model.classifier_nodes:  
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1)
            n_classified += torch.sum(mask)

            # masked input and output
            m_gen_conv_features = gen_conv_features[mask]
            m_img_conv_features = img_conv_features[mask]
            m_label = label[mask][:, node.depth] - 1

            # forward pass through the node 
            (m_gen_logits, m_img_logits), (m_gen_min_dist, m_img_min_dist) = node.forward(m_gen_conv_features, m_img_conv_features) 

            # cross entropy loss 
            if len(m_gen_logits) != 0: 
                gen_ce = F.cross_entropy(m_gen_logits, m_label)  
                img_ce = F.cross_entropy(m_img_logits, m_label)
                batch_obj.cross_entropy += gen_ce + img_ce

            # cluster and separation loss 
            gen_cluster, gen_separation = get_cluster_and_sep_cost(m_gen_min_dist, m_label, node.nclass)
            img_cluster, img_separation = get_cluster_and_sep_cost(m_img_min_dist, m_label, node.nclass)
            batch_obj.cluster += gen_cluster + img_cluster
            batch_obj.separation = gen_separation + img_separation

            # lasso loss
            batch_obj.lasso += get_l1_cost(node.gen_node) + get_l1_cost(node.img_node) 

            # correspondence loss 
            corr_sum, corr_count = get_correspondence_loss_batched(m_gen_min_dist, m_img_min_dist, node)
            batch_obj.correspondence += corr_sum
            total_corr_count += corr_count

            # orthogonality loss 
            batch_obj.gen_orthogonality = get_ortho_cost(node.gen_node)
            batch_obj.img_orthogonality = get_ortho_cost(node.img_node)

            # Free everything up for this node
            del m_gen_conv_features, m_img_conv_features, m_gen_logits, m_img_logits, m_gen_min_dist, m_img_min_dist

        batch_obj.correspondence /= total_corr_count

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified 

        # add batch objective values to total objective values for logging
        total_obj += batch_obj

        # Free everything up for this batch
        del gen_conv_features, img_conv_features

    # normalize the losses after running through entire dataset
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))

