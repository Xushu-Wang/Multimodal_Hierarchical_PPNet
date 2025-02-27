import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Optimizer
from yacs.config import CfgNode
from model.hierarchical import Mode, HierProtoPNet
from model.multimodal import MultiHierProtoPNet
from typing import Union
from train.loss import Objective, MultiObjective, get_cluster_and_sep_cost, get_l1_cost, get_ortho_cost, get_correspondence_loss_batched
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
            train_genetic(model, dataloader, optimizer, cfg, log) 
        case Mode.IMAGE: 
            train_image(model, dataloader, optimizer, cfg, log) 
        case Mode.MULTIMODAL: 
            train_multimodal(model, dataloader, optimizer, cfg, log) 


def train_genetic(model, dataloader, optimizer, cfg, log):  
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "train")

    for (genetics, _), (label, flat_label) in tqdm(dataloader): 

        input = genetics.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "train")

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            logits, min_dist = node.forward(conv_features)
            
            # softmax the nodes to initialize node.probs
            node.softmax() 

            node.npredictions += torch.sum(mask) 
            n_classified += torch.sum(mask)

            # masked input and output
            m_label = label[mask][:, node.depth]
            m_logits, m_min_dist = logits[mask], min_dist[mask]
            if len(m_label) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                predictions = torch.argmax(m_logits, dim=1)
                node.n_next_correct += torch.sum(predictions == m_label) 
                batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation = separation
            batch_obj.lasso += get_l1_cost(node) 

        model.conditional_normalize(model.root)  

        # calculate species predictions over this batch, which is done above
        last_classifiers = [node for node in model.classifier_nodes if node.depth == 3] 
        last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
        logits = torch.concat([node.probs for node in last_classifiers], dim=1) # 80x113

        for node in model.classifier_nodes: 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            m_flat_label = flat_label[mask][:, -1]
            cond_m_logits = logits[mask][:, node.min_species_idx:node.max_species_idx]
            cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
            node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
            batch_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

        total_loss = batch_obj.total()  
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified 
        total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

def train_image(model, dataloader, optimizer, cfg, log): 
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "train")

    for (_, image), (label, flat_label) in tqdm(dataloader): 

        input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "train")

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

        # track number of classifications across all nodes in this batch
        n_classified = 0 

        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            logits, min_dist = node.forward(conv_features)
            
            # softmax the nodes to initialize node.probs
            node.softmax() 

            node.npredictions += torch.sum(mask) 
            n_classified += torch.sum(mask)

            # masked input and output
            m_label = label[mask][:, node.depth]
            m_logits, m_min_dist = logits[mask], min_dist[mask]
            if len(m_logits) != 0: 
                batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                predictions = torch.argmax(m_logits, dim=1)
                node.n_next_correct += torch.sum(predictions == m_label) 
                batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

            cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
            batch_obj.cluster += cluster
            batch_obj.separation += separation
            batch_obj.lasso += get_l1_cost(node) 

        model.conditional_normalize(model.root)  

        # calculate species predictions over this batch, which is done above
        last_classifiers = [node for node in model.classifier_nodes if node.depth == 3] 
        last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
        logits = torch.concat([node.probs for node in last_classifiers], dim=1) # 80x113

        for node in model.classifier_nodes: 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            m_flat_label = flat_label[mask][:, -1]
            cond_m_logits = logits[mask][:, node.min_species_idx:node.max_species_idx]
            cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
            node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
            batch_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

        total_loss = batch_obj.total()  
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.cluster /= n_classified
        batch_obj.separation /= n_classified 
        total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

def train_multimodal(model, dataloader, optimizer, cfg, log): 
    total_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "train")

    for (genetics, image), (label, flat_label) in tqdm(dataloader): 

        gen_input = genetics.cuda()
        img_input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "train")

        with torch.enable_grad(): 
            # can't forward the entire model here since it 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)

        # # track number of classifications across all nodes in this batch
        n_classified = 0 
        total_corr_count = 0


        for node in model.classifier_nodes: 
            # filter out the irrelevant samples in batch 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            gen_logits, gen_min_dist = node.gen_node.forward(gen_conv_features) 
            img_logits, img_min_dist = node.img_node.forward(img_conv_features) 
            node.gen_node.softmax()
            node.img_node.softmax() 

            node.gen_node.npredictions += torch.sum(mask)
            node.img_node.npredictions += torch.sum(mask)
            n_classified += torch.sum(mask)

            # masked input and output
            m_label = label[mask][:, node.depth]
            m_gen_logits, m_gen_min_dist = gen_logits[mask], gen_min_dist[mask]
            m_img_logits, m_img_min_dist = img_logits[mask], img_min_dist[mask]
            if len(m_label) != 0: 
                batch_obj.gen_obj.cross_entropy += F.cross_entropy(m_gen_logits, m_label)  
                batch_obj.img_obj.cross_entropy += F.cross_entropy(m_img_logits, m_label)  

                gen_predictions = torch.argmax(m_gen_logits, dim=1) 
                img_predictions = torch.argmax(m_img_logits, dim=1) 
                node.gen_node.n_next_correct += torch.sum(gen_predictions == m_label)
                node.img_node.n_next_correct += torch.sum(img_predictions == m_label)
                batch_obj.gen_obj.n_next_correct[node.depth] += torch.sum(gen_predictions == m_label) 
                batch_obj.img_obj.n_next_correct[node.depth] += torch.sum(img_predictions == m_label) 

            # cluster and separation loss 
            gen_cluster, gen_separation = get_cluster_and_sep_cost(m_gen_min_dist, m_label, node.nclass)
            img_cluster, img_separation = get_cluster_and_sep_cost(m_img_min_dist, m_label, node.nclass)
            batch_obj.gen_obj.cluster += gen_cluster 
            batch_obj.gen_obj.separation += gen_separation 
            batch_obj.img_obj.cluster += img_cluster 
            batch_obj.img_obj.separation += img_separation 

            # lasso loss
            batch_obj.gen_obj.lasso += get_l1_cost(node.gen_node)
            batch_obj.img_obj.lasso += get_l1_cost(node.img_node)
            
            # correspondence loss 
            corr_sum, corr_count = get_correspondence_loss_batched(m_gen_min_dist, m_img_min_dist, node)
            batch_obj.correspondence += corr_sum
            total_corr_count += corr_count

            # orthogonality loss 
            batch_obj.gen_obj.orthogonality += get_ortho_cost(node.gen_node)
            batch_obj.img_obj.orthogonality += get_ortho_cost(node.img_node)

        batch_obj.correspondence /= total_corr_count

        model.gen_net.conditional_normalize(model.gen_net.root)  
        model.img_net.conditional_normalize(model.img_net.root)  

        # calculate genetic accuracies
        gen_last_classifiers = [node for node in model.gen_net.classifier_nodes if node.depth == 3] 
        gen_last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
        gen_logits = torch.concat([node.probs for node in gen_last_classifiers], dim=1) # 80x113

        for node in model.gen_net.classifier_nodes: 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            m_flat_label = flat_label[mask][:, -1]
            cond_m_logits = gen_logits[mask][:, node.min_species_idx:node.max_species_idx]
            cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
            node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
            batch_obj.gen_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

        # calculate image accuracies
        img_last_classifiers = [node for node in model.img_net.classifier_nodes if node.depth == 3] 
        img_last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
        img_logits = torch.concat([node.probs for node in img_last_classifiers], dim=1) # 80x113

        for node in model.img_net.classifier_nodes: 
            mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
            m_flat_label = flat_label[mask][:, -1]
            cond_m_logits = img_logits[mask][:, node.min_species_idx:node.max_species_idx]
            cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
            node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
            batch_obj.img_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

        # print(f"GPU : {torch.cuda.memory_allocated() / 1024 ** 2 :.2f} MB")

        total_loss = batch_obj.total()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        # Divide the cls/sep costs before (?) adding to total objective cost 
        batch_obj.gen_obj.cluster /= n_classified
        batch_obj.gen_obj.separation /= n_classified 
        batch_obj.img_obj.cluster /= n_classified
        batch_obj.img_obj.separation /= n_classified 
        total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

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
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "test")

    for (genetics, _), (label, flat_label) in tqdm(dataloader): 

        input = genetics.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "test")

        with torch.no_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

            # track number of classifications across all nodes in this batch
            n_classified = 0 

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                logits, min_dist = node.forward(conv_features)
                
                # softmax the nodes to initialize node.probs
                node.softmax() 

                node.npredictions += torch.sum(mask) 
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_logits, m_min_dist = logits[mask], min_dist[mask]
                if len(m_logits) != 0: 
                    batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                    predictions = torch.argmax(m_logits, dim=1)
                    node.n_next_correct += torch.sum(predictions == m_label) 
                    batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

                cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
                batch_obj.cluster += cluster
                batch_obj.separation = separation
                batch_obj.lasso += get_l1_cost(node) 

            model.conditional_normalize(model.root) 

            # calculate species predictions over this batch, which is done above
            last_classifiers = [node for node in model.classifier_nodes if node.depth == 3] 
            last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
            logits = torch.concat([node.probs for node in last_classifiers], dim=1) # 80x113

            for node in model.classifier_nodes: 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                m_flat_label = flat_label[mask][:, -1]
                cond_m_logits = logits[mask][:, node.min_species_idx:node.max_species_idx]
                cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
                node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
                batch_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

            # Divide the cls/sep costs before (?) adding to total objective cost 
            batch_obj.cluster /= n_classified
            batch_obj.separation /= n_classified 
            total_obj += batch_obj

    model.zero_pred()

    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

def test_image(model, dataloader, cfg, log): 
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "test")

    for (_, image), (label, flat_label) in tqdm(dataloader): 

        input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "test")

        with torch.no_grad(): 
            # can't forward the entire model here since it 
            conv_features = model.conv_features(input)

            # track number of classifications across all nodes in this batch
            n_classified = 0 

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                logits, min_dist = node.forward(conv_features)
                
                # softmax the nodes to initialize node.probs
                node.softmax() 

                node.npredictions += torch.sum(mask) 
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_logits, m_min_dist = logits[mask], min_dist[mask]
                if len(m_logits) != 0: 
                    batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                    predictions = torch.argmax(m_logits, dim=1)
                    node.n_next_correct += torch.sum(predictions == m_label) 
                    batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

                cluster, separation = get_cluster_and_sep_cost(m_min_dist, m_label, node.nclass)
                batch_obj.cluster += cluster
                batch_obj.separation += separation
                batch_obj.lasso += get_l1_cost(node) 
        
            model.conditional_normalize(model.root)  

            # calculate species predictions over this batch, which is done above
            last_classifiers = [node for node in model.classifier_nodes if node.depth == 3] 
            last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
            logits = torch.concat([node.probs for node in last_classifiers], dim=1) # 80x113

            for node in model.classifier_nodes: 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                m_flat_label = flat_label[mask][:, -1]
                cond_m_logits = logits[mask][:, node.min_species_idx:node.max_species_idx]
                cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
                node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
                batch_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

            # Divide the cls/sep costs before (?) adding to total objective cost 
            batch_obj.cluster /= n_classified
            batch_obj.separation /= n_classified 
            total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

def test_multimodal(model, dataloader, cfg, log): 
    total_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), "test")

    for (genetics, image), (label, flat_label) in tqdm(dataloader): 

        gen_input = genetics.cuda()
        img_input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, "test")

        with torch.no_grad(): 
            # can't forward the entire model here since it 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)

            # track number of classifications across all nodes in this batch
            n_classified = 0 
            total_corr_count = 0

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                gen_logits, gen_min_dist = node.gen_node.forward(gen_conv_features) 
                img_logits, img_min_dist = node.img_node.forward(img_conv_features) 
                node.gen_node.softmax()
                node.img_node.softmax() 

                node.gen_node.npredictions += torch.sum(mask)
                node.img_node.npredictions += torch.sum(mask)
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_gen_logits, m_gen_min_dist = gen_logits[mask], gen_min_dist[mask]
                m_img_logits, m_img_min_dist = img_logits[mask], img_min_dist[mask]
                if len(m_label) != 0: 
                    batch_obj.gen_obj.cross_entropy += F.cross_entropy(m_gen_logits, m_label)  
                    batch_obj.img_obj.cross_entropy += F.cross_entropy(m_img_logits, m_label)  

                    gen_predictions = torch.argmax(m_gen_logits, dim=1) 
                    img_predictions = torch.argmax(m_img_logits, dim=1) 
                    node.gen_node.n_next_correct += torch.sum(gen_predictions == m_label)
                    node.img_node.n_next_correct += torch.sum(img_predictions == m_label)
                    batch_obj.gen_obj.n_next_correct[node.depth] += torch.sum(gen_predictions == m_label) 
                    batch_obj.img_obj.n_next_correct[node.depth] += torch.sum(img_predictions == m_label) 

                # cluster and separation loss 
                gen_cluster, gen_separation = get_cluster_and_sep_cost(m_gen_min_dist, m_label, node.nclass)
                img_cluster, img_separation = get_cluster_and_sep_cost(m_img_min_dist, m_label, node.nclass)
                batch_obj.gen_obj.cluster += gen_cluster 
                batch_obj.gen_obj.separation += gen_separation 
                batch_obj.img_obj.cluster += img_cluster 
                batch_obj.img_obj.separation += img_separation 

                # lasso loss
                batch_obj.gen_obj.lasso += get_l1_cost(node.gen_node)
                batch_obj.img_obj.lasso += get_l1_cost(node.img_node)
                
                # correspondence loss (memory leak in correspondence loss)
                corr_sum, corr_count = get_correspondence_loss_batched(m_gen_min_dist, m_img_min_dist, node)
                batch_obj.correspondence += corr_sum
                total_corr_count += corr_count

                # orthogonality loss 
                batch_obj.gen_obj.orthogonality += get_ortho_cost(node.gen_node)
                batch_obj.img_obj.orthogonality += get_ortho_cost(node.img_node)

            batch_obj.correspondence /= total_corr_count

            model.gen_net.conditional_normalize(model.gen_net.root)  
            model.img_net.conditional_normalize(model.img_net.root)  

            # calculate genetic accuracies
            gen_last_classifiers = [node for node in model.gen_net.classifier_nodes if node.depth == 3] 
            gen_last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
            gen_logits = torch.concat([node.probs for node in gen_last_classifiers], dim=1) # 80x113

            for node in model.gen_net.classifier_nodes: 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                m_flat_label = flat_label[mask][:, -1]
                cond_m_logits = gen_logits[mask][:, node.min_species_idx:node.max_species_idx]
                cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
                node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
                batch_obj.gen_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

            # calculate image accuracies
            img_last_classifiers = [node for node in model.img_net.classifier_nodes if node.depth == 3] 
            img_last_classifiers.sort(key = lambda node : node.flat_idx[-1]) 
            img_logits = torch.concat([node.probs for node in img_last_classifiers], dim=1) # 80x113

            for node in model.img_net.classifier_nodes: 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                m_flat_label = flat_label[mask][:, -1]
                cond_m_logits = img_logits[mask][:, node.min_species_idx:node.max_species_idx]
                cond_predictions = torch.argmax(cond_m_logits, dim=1) + node.min_species_idx
                node.n_species_correct += torch.sum(cond_predictions == m_flat_label)
                batch_obj.img_obj.n_cond_correct[node.depth] += torch.sum(cond_predictions == m_flat_label)

            # Divide the cls/sep costs before (?) adding to total objective cost 
            batch_obj.gen_obj.cluster /= n_classified
            batch_obj.gen_obj.separation /= n_classified 
            batch_obj.img_obj.cluster /= n_classified
            batch_obj.img_obj.separation /= n_classified 
            total_obj += batch_obj
            batch_obj.clear()

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    wandb.log(total_obj.to_dict())
    log(str(total_obj))
    total_obj.clear()

