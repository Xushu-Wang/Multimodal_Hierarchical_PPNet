import argparse, os
import torch
from prototype.prune import prune_prototypes
from os import mkdir

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, save_model_w_condition
from model.model import Mode
from dataio.dataloader import get_dataloaders
from dataio.dataset import get_datasets 
from model.model import construct_tree_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
import prototype.push as push       
from pprint import pprint
from torchinfo import summary 
from torchviz import make_dot 

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/parallel.yaml")
run_id_accumulator(cfg) 

tree_ppnet = construct_tree_ppnet(cfg).to("cuda") 

genetic_root = tree_ppnet.genetic_hierarchical_ppnet.root
image_root = tree_ppnet.image_hierarchical_ppnet.root 
