import torch
from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, save_model_w_condition
from dataio.dataset import get_datasets, Hierarchy
from pprint import pprint 
from model.model import construct_tree_ppnet

cfg = get_cfg_defaults()
cfg.merge_from_file("configs/parallel.yaml")
run_id_accumulator(cfg) 

mmppnet = construct_tree_ppnet(cfg).to("cuda")

def cos_sim(self, x): 
    """
    x - convolutional output features: img=(80, 2048, 8, 8) gen=(80, 64, 1, 40) 
    """
    sqrt_D = (self.pshape[1] * self.pshape[2]) ** 0.5 
    x = F.normalize(x, dim=1) / sqrt_D 
    normalized_proto = F.normalize(self.prototypes, dim=1) / sqrt_D 

    if self.mode == Mode.GENETIC: 
        pass 

    return F.conv2d(x, normalized_proto)

