import torch
from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, save_model_w_condition
from dataio.dataset import get_datasets, Hierarchy
from pprint import pprint 

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/parallel.yaml")
run_id_accumulator(cfg) 

train, push, val, test, normalize = get_datasets(cfg) 
