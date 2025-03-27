import torch
import torch.nn.functional as F
from yacs.config import CfgNode
from model.hierarchical import Mode, HierProtoPNet
from model.multimodal import MultiHierProtoPNet
from train.loss import Objective, MultiObjective, get_cluster_and_sep_cost, get_l1_cost, get_ortho_cost, get_correspondence_loss_batched
from train.optimizer import Optim, OptimMode
from tqdm import tqdm
from typing import Union

Model = Union[HierProtoPNet, MultiHierProtoPNet]

def warm_only(model: Model):
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
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False

    for p in model.get_prototype_parameters():
        p.requires_grad = False
    
    for l in model.get_last_layer_parameters():
        l.requires_grad = True
    
def joint(model: Model):
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

def traintest(
    model: Model, 
    dataloader, 
    optimizer: Optim, 
    cfg: CfgNode
): 
    """
    Train wrapper function for all models. 
    It first sets specific parameter gradients = True depending on optim_mode 
    Then depending on the model mode, it calls the specific train functions
    """
    mode = Mode(model.mode) 

    match optimizer.mode: 
        case OptimMode.WARM: 
            warm_only(model)
            model.train()
        case OptimMode.JOINT: 
            joint(model)
            model.train()
        case OptimMode.LAST: 
            last_only(model) 
            model.train()
        case OptimMode.TEST: 
            model.eval()

    match mode: 
        case Mode.GENETIC: 
            return _traintest_genetic   (model, dataloader, optimizer, cfg) 
        case Mode.IMAGE: 
            return _traintest_image     (model, dataloader, optimizer, cfg) 
        case Mode.MULTIMODAL: 
            return _traintest_multi     (model, dataloader, optimizer, cfg) 

def _traintest_genetic(model, dataloader, optimizer, cfg): 
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), optimizer.mode.value)

    for (genetics, _), (label, flat_label) in tqdm(dataloader): 
        input = genetics.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, optimizer.mode.value)

        with torch.no_grad() if optimizer.mode == OptimMode.TEST else torch.enable_grad(): 
            conv_features = model.conv_features(input)

            # track number of classifications across all nodes in this batch
            n_classified = 0 

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                logits, max_sim = node.forward(conv_features)
                
                # softmax the nodes to initialize node.probs
                node.softmax() 

                node.npredictions += torch.sum(mask) 
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_logits, m_max_sim = logits[mask], max_sim[mask]
                if len(m_label) != 0: 
                    batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                    predictions = torch.argmax(m_logits, dim=1)
                    node.n_next_correct += torch.sum(predictions == m_label) 
                    batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

                cluster, separation = get_cluster_and_sep_cost(m_max_sim, m_label, node.nclass)
                batch_obj.cluster[node.depth] = cluster + batch_obj.cluster[node.depth]
                batch_obj.separation[node.depth] = separation + batch_obj.separation[node.depth]
                batch_obj.cluster_sep_count[node.depth] += len(m_label)

                batch_obj.lasso += get_l1_cost(node) 

            if not optimizer.mode == OptimMode.TEST: 
                total_loss = batch_obj.total()  
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

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
        # batch_obj.cluster /= cluster_sep_count
        # batch_obj.separation /= cluster_sep_count 
        total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader) 
    return total_obj

def _traintest_image(model, dataloader, optimizer, cfg): 
    total_obj = Objective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), optimizer.mode.value)

    for (_, image), (label, flat_label) in tqdm(dataloader): 
        input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = Objective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, optimizer.mode.value)
        
        with torch.no_grad() if optimizer.mode == OptimMode.TEST else torch.enable_grad(): 
            conv_features = model.conv_features(input)

            # track number of classifications across all nodes in this batch
            n_classified = 0 

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                logits, max_sim = node.forward(conv_features)
                
                # softmax the nodes to initialize node.probs
                node.softmax() 

                node.npredictions += torch.sum(mask) 
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_logits, m_max_sim = logits[mask], max_sim[mask]
                if len(m_logits) != 0: 
                    batch_obj.cross_entropy += F.cross_entropy(m_logits, m_label) 
                    predictions = torch.argmax(m_logits, dim=1)
                    node.n_next_correct += torch.sum(predictions == m_label) 
                    batch_obj.n_next_correct[node.depth] += torch.sum(predictions == m_label) 

                cluster, separation = get_cluster_and_sep_cost(m_max_sim, m_label, node.nclass)
                batch_obj.cluster[node.depth] = cluster + batch_obj.cluster[node.depth]
                batch_obj.separation[node.depth] = separation + batch_obj.separation[node.depth]
                batch_obj.cluster_sep_count[node.depth] += len(m_label)

                batch_obj.lasso += get_l1_cost(node) 

            if not optimizer.mode == OptimMode.TEST: 
                total_loss = batch_obj.total()  
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

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
        batch_obj.cluster /= cluster_sep_count
        batch_obj.separation /= cluster_sep_count
        total_obj += batch_obj

    model.zero_pred()
    # normalize the losses
    total_obj /= len(dataloader)
    return total_obj

def _traintest_multi(model, dataloader, optimizer, cfg): 
    total_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, len(dataloader.dataset), optimizer.mode.value)

    for (genetics, image), (label, flat_label) in tqdm(dataloader): 
        gen_input = genetics.cuda()
        img_input = image.cuda()
        label = label.cuda()
        flat_label = flat_label.cuda()
        batch_obj = MultiObjective(model.mode, cfg.OPTIM.COEFS, dataloader.batch_size, optimizer.mode.value)

        with torch.no_grad() if optimizer.mode == OptimMode.TEST else torch.enable_grad(): 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)

            # # track number of classifications across all nodes in this batch
            n_classified = 0 
            total_corr_count = 0

            for node in model.classifier_nodes: 
                # filter out the irrelevant samples in batch 
                mask = torch.all(label[:,:node.depth] == node.idx.cuda(), dim=1) 
                gen_logits, gen_max_sim = node.gen_node.forward(gen_conv_features) 
                img_logits, img_max_sim = node.img_node.forward(img_conv_features) 
                node.gen_node.softmax()
                node.img_node.softmax() 

                node.gen_node.npredictions += torch.sum(mask)
                node.img_node.npredictions += torch.sum(mask)
                n_classified += torch.sum(mask)

                # masked input and output
                m_label = label[mask][:, node.depth]
                m_gen_logits, m_gen_max_sim = gen_logits[mask], gen_max_sim[mask]
                m_img_logits, m_img_max_sim = img_logits[mask], img_max_sim[mask]
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
                gen_cluster, gen_separation = get_cluster_and_sep_cost(m_gen_max_sim, m_label, node.nclass)
                img_cluster, img_separation = get_cluster_and_sep_cost(m_img_max_sim, m_label, node.nclass)
                batch_obj.gen_obj.cluster[node.depth] = batch_obj.gen_obj.cluster[node.depth] + gen_cluster
                batch_obj.gen_obj.separation[node.depth] = batch_obj.gen_obj.separation[node.depth] + gen_separation 
                batch_obj.gen_obj.cluster_sep_count[node.depth] = batch_obj.gen_obj.cluster_sep_count[node.depth]+ len(m_label) 
                batch_obj.img_obj.cluster[node.depth] = batch_obj.img_obj.cluster[node.depth] + img_cluster 
                batch_obj.img_obj.separation[node.depth] = batch_obj.img_obj.separation[node.depth] + img_separation 
                batch_obj.img_obj.cluster_sep_count[node.depth] = batch_obj.img_obj.cluster_sep_count[node.depth] + len(m_label) 

                # lasso loss
                batch_obj.gen_obj.lasso += get_l1_cost(node.gen_node)
                batch_obj.img_obj.lasso += get_l1_cost(node.img_node)
                
                # correspondence loss 
                corr_sum, corr_count = get_correspondence_loss_batched(m_gen_max_sim, m_img_max_sim, node)
                batch_obj.correspondence += corr_sum
                total_corr_count += corr_count

                # orthogonality loss 
                batch_obj.gen_obj.orthogonality += get_ortho_cost(node.gen_node)
                batch_obj.img_obj.orthogonality += get_ortho_cost(node.img_node)

            batch_obj.correspondence /= total_corr_count

            if not optimizer.mode == OptimMode.TEST: 
                total_loss = batch_obj.total() 
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

        model.gen_net.conditional_normalize(model.gen_net.root)  
        model.img_net.conditional_normalize(model.img_net.root)  

        # calculate genetic accuracies
        gen_last_classifiers = [node for node in model.gen_net.all_classifier_nodes if node.depth == 3] 
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
        img_last_classifiers = [node for node in model.img_net.all_classifier_nodes if node.depth == 3] 
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
        # batch_obj.gen_obj.cluster /= n_classified
        # batch_obj.gen_obj.separation /= n_classified 
        # batch_obj.img_obj.cluster /= n_classified
        # batch_obj.img_obj.separation /= n_classified 
        batch_obj.gen_obj.cluster /= batch_obj.gen_obj.cluster_sep_count
        batch_obj.gen_obj.separation /= batch_obj.gen_obj.cluster_sep_count
        batch_obj.img_obj.cluster /= batch_obj.img_obj.cluster_sep_count
        batch_obj.img_obj.separation /= batch_obj.img_obj.cluster_sep_count
        total_obj += batch_obj

    model.zero_pred()
    total_obj /= len(dataloader)
    return total_obj
