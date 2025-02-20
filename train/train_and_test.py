import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np
from yacs.config import CfgNode
from pympler.tracker import SummaryTracker
from model.model import Mode, HierProtoPNet, MultiHierProtoPNet
from typing import Union
from dataio.dataset import TreeDataset
from utils.util import format_dictionary_nicely_for_printing
from train.loss import get_loss
from tqdm import tqdm

Model = Union[HierProtoPNet, MultiHierProtoPNet, torch.nn.DataParallel]

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

    if model.mode == Mode.MULTIMODAL:
        for p in model.get_last_layer_multi_parameters():
            p.requires_grad = False

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

    if model.mode == Mode.MULTIMODAL:
        for p in model.get_last_layer_multi_parameters():
            p.requires_grad = False
    
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

    if model.mode == Mode.MULTIMODAL:
        for p in model.get_last_layer_multi_parameters():
            p.requires_grad = True


def train(
    model: Model, 
    dataloader: DataLoader, 
    optimizer: Optimizer, 
    cfg: CfgNode, 
    run, 
    log = print
): 
    mode = Mode(model.mode) 
    match mode: 
        case Mode.GENETIC: 
            return train_genetic(model, dataloader, optimizer, cfg, run, log) 
        case Mode.IMAGE: 
            return train_image(model, dataloader, optimizer, cfg, run, log) 
        case Mode.MULTIMODAL: 
            return train_multimodal(model, dataloader, optimizer, cfg, run, log) 


def train_genetic(model, dataloader, optimizer, cfg, run, log): 

    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_l1 = 0
    torch.autograd.set_detect_anomaly(True)

    model.train()
    for (genetics, _), (label, _) in tqdm(dataloader): 

        input = genetics.cuda()
        label = label.cuda()

        with torch.enable_grad(): 
            conv_features = model.conv_features(input)
            cross_entropy, cluster_cost, separation_cost, l1_cost, n_classifications = get_loss(
                conv_features = conv_features,
                node = model.root,
                target = label
            )

        loss = (
            cfg.OPTIM.COEFS.CRS_ENT * cross_entropy + 
            cfg.OPTIM.COEFS.CLST * cluster_cost + 
            cfg.OPTIM.COEFS.SEP * separation_cost + 
            cfg.OPTIM.COEFS.L1 * l1_cost 
        )
                      
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        total_cross_entropy += cross_entropy
        total_cluster_cost += cluster_cost / n_classifications
        total_separation_cost += separation_cost / n_classifications
        total_l1 += l1_cost 

        del conv_features


def train_image(model, dataloader, optimizer, cfg, run, log): 
    model.train()

def train_multimodal(model, dataloader, optimizer, cfg, run, log): 
    model.train()

