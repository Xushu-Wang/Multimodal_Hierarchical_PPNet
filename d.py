import argparse, os
import torch
from prototype.prune import prune_prototypes
from os import mkdir

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, save_model_w_condition
from model.model import Mode
from dataio.dataloader import get_dataloaders
from dataio.dataset import get_datasets, Hierarchy
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

json_file = cfg.DATASET.TREE_SPECIFICATION_FILE 

h = Hierarchy(json_file) 
print(h)

print(h.root)
for c in h.root.children: 
    print("  " + c.__repr__()) 
    for c2 in c.children: 
        print("    " + c2.__repr__())

