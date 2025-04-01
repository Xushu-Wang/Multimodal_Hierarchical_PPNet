import torch
import numpy as np
import argparse, os
import time
from yacs.config import CfgNode

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator
from dataio.dataloader import get_dataloaders
from model.multimodal import construct_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
from prototype import push, prune
from typing import Callable
from torchvision.transforms import transforms
from tqdm import tqdm
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def main(cfg: CfgNode):
    _, push_loader, _, _ = get_dataloaders(cfg, print)
    model = construct_ppnet(cfg, print).to(cfg.MODEL.DEVICE)
    my_dict = torch.load(os.path.join("..", "output", "joint", "unittest_push_008", "before_push_weights.pth"), weights_only=True)
    model.load_state_dict(my_dict) 

    model.eval() 
    for node in model.classifier_nodes: 
        node.init_push()

    # current implementation of push 
    with torch.no_grad(): 
        normalize = transforms.Normalize(
            mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            std=cfg.DATASET.IMAGE.TRANSFORM_STD
        )
        for ((genetics, raw_image), (label, _)) in tqdm(push_loader):
            gen_input = genetics.to(cfg.MODEL.DEVICE)
            img_input = normalize(raw_image).to(cfg.MODEL.DEVICE)
            raw_image = raw_image.to(cfg.MODEL.DEVICE)

            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)  

            # for each node, find the prototype that it should project to 
            for node in model.classifier_nodes: 
                # filter out the conv features and inputs for node A
                filter = torch.all(label[:, :len(node.gen_node.taxnode.idx)] == node.gen_node.taxnode.idx, dim=1)
                f_label = label[filter] 
                f_gen_input = gen_input[filter]
                f_raw_image = raw_image[filter]
                f_gen_conv_features = gen_conv_features[filter]
                f_img_conv_features = img_conv_features[filter]

                push.find_closest_conv_feature(
                    model=model.img_net,
                    node=node.img_node,
                    conv_features=f_img_conv_features,
                    input=f_raw_image,
                    label=f_label,
                    stride=1,
                    epoch=1,
                    cfg=cfg
                )
                push.find_closest_conv_feature(
                    model=model.gen_net,
                    node=node.gen_node,
                    conv_features=f_gen_conv_features,
                    input=f_gen_input,
                    label=f_label,
                    stride=1,
                    epoch=1,
                    cfg=cfg
                ) 
    
    # new implementation of push with for loops 
    with torch.no_grad(): 

        # Just grab the entire dataset and put it into one tensor. 
        normalize = transforms.Normalize(
            mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            std=cfg.DATASET.IMAGE.TRANSFORM_STD
        ) 
        dataset = torch.zeros(1130, 64, 1, 40).cuda()
        labels = torch.zeros(1130, 4).cuda()
        i = 0 
        for ((genetics, raw_image), (label, _)) in tqdm(push_loader): 
            gen_input = genetics.to(cfg.MODEL.DEVICE)
            img_input = normalize(raw_image).to(cfg.MODEL.DEVICE)
            raw_image = raw_image.to(cfg.MODEL.DEVICE)

            gen_conv_features, _ = model.conv_features(gen_input, img_input) 
            assert gen_conv_features.size() == torch.Size([113, 64, 1, 40])

            dataset[i:i+113,:,:,:] = gen_conv_features
            labels[i:i+113,:] = label
            i += 113

        # manually pick an arbitrary node 
        gen_node = model.gen_net.root.childs[IDX]
        max_sims = torch.full((gen_node.nclass,), -float('inf'))
        best_protos = torch.zeros(gen_node.nclass, 64)

        counts = torch.zeros(gen_node.nclass) 

        # PyTorch implementation of cossim--different from ours since we do conv2d, 
        # but in theory should be same 
        cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6) 
        
        # integer index of the class hierarchy, e.g. [1, 2] represents 
        # order 1, genus 2 
        tax_idx = gen_node.taxnode.idx.cuda() 
        print("Number of Classes: ", gen_node.nclass)
        
        # only one prototype for each class C in each node 
        for proto_idx in range(gen_node.nclass): 
            # grab the prototype 
            prototype = gen_node.prototype[proto_idx]  

            # filter dataset to consider only samples that match 
            # up to class C 
            mask = torch.all(labels[:, :len(gen_node.taxnode.idx)] == tax_idx, dim=1)
            f_dataset = dataset[mask] 
            
            counts[proto_idx] = f_dataset.size(0)

            for s in range(f_dataset.size(0)):
                for p in range(40):     # patch  
                    cos_sim = cossim(prototype.squeeze(), f_dataset[s,:,:,p].squeeze()) 
                    if cos_sim > max_sims[proto_idx]: 
                        max_sims[proto_idx] = cos_sim 
                        best_protos[proto_idx,:] = dataset[s,:,:,p].squeeze()

    print("After Push")
    print("Differences between Max Cosine Similarities")
    print(torch.tensor(model.gen_net.root.childs[IDX].global_max_proto_sim) - max_sims)
    
    print("Differences between Best Prototypes to Project To")
    ours = torch.tensor(np.squeeze(model.gen_net.root.childs[IDX].global_max_fmap_patches))
    truth = best_protos 
    print(torch.norm(ours - truth, dim=1)) 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='multi.yaml') 
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()


    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("configs", args.config))
    run_id_accumulator(cfg) 
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    # IDX determines which node to look at 
    IDX = args.idx
    main(cfg)
