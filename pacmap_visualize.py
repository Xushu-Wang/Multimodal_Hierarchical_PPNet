import pickle
import torch
import numpy as np
from tqdm import tqdm
import argparse, os, wandb
from yacs.config import CfgNode

from configs.cfg import get_cfg_defaults 
from dataio.dataloader import get_dataloaders
from model.multimodal import construct_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

run_mode = {1 : "genetic", 2 : "image", 3 : "multimodal"} 
log = print

def main(cfg: CfgNode):
    train_loader, _, _, _ = get_dataloaders(cfg, log)
    log("Dataloaders Constructed")

    model = construct_ppnet(cfg, log).cuda()
    log("ProtoPNet constructed")

    all_vectors = {
    }
    location = []
    depth = len(location)

    node = [n for n in model.all_classifier_nodes if n.depth == 0][0]
    all_vectors["prototype"] = node.prototype.detach().cpu().numpy()

    for (genetics, image), (label, flat_label) in tqdm(train_loader):
        if cfg.MODE == 2:
            raise NotImplementedError("Image mode is not implemented yet.")
        
        genetics = genetics.cuda()
        label = label.cuda()

        # Forward pass
        with torch.no_grad():
            gen_conv_features = model.features(genetics)
            gen_conv_features = model.add_on_layers(gen_conv_features)
            vector = gen_conv_features.squeeze()
            vector = vector.cpu().numpy()

            # Find max location for each prototype
            best_loc = node.cos_sim(gen_conv_features).argmax(dim=3)
            best_loc = best_loc.view(best_loc.shape[0], -1)
            
            most_common_loc = best_loc.mode(dim=1)[0]
            print(most_common_loc)

            for i in range(genetics.shape[0]):
                do_cont = False
                for j, loc in enumerate(location):
                    if label[i][j] != loc:
                        do_cont = True
                        break
                if do_cont:
                    continue
                if label[i][depth].item() not in all_vectors:
                    all_vectors[label[i][depth].item()] = []
                all_vectors[label[i][depth].item()].append(vector[i])
    
    # Save the vectors to a file
    with open("vectors.pkl", 'wb') as file:
        pickle.dump(all_vectors, file)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="Path to the source model")
    parser.add_argument('--config', type=str, default='multi.yaml')
    parser.add_argument('--name', type=str, default="pacmap")
    parser.add_argument('--type', type=str, default="genetic", choices=["genetic", "image"])
    
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("configs", args.config))

    cfg.MODE = (1 if args.type == "genetic" else 2)

    if cfg.MODE == 1:
        cfg.MODEL.GENETIC.PPNET_PATH = args.src
    elif cfg.MODE == 2:
        cfg.MODEL.IMAGE.PPNET_PATH = args.src

    main(cfg)  
