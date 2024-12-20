from configs.cfg import get_cfg_defaults
from utils.util import handle_run_name_weirdness

from dataio.tree import get_dataloaders
from model.model import construct_tree_ppnet
from utils.util import create_logger
import os, json
import pandas as pd

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/image.yaml") 
train_loader, train_push_loader, val_loader, test_loader, image_normalizer = get_dataloaders(cfg, print)

