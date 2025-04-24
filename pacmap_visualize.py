import pickle
import torch
import numpy as np
from tqdm import tqdm
import argparse, os, wandb
from yacs.config import CfgNode

from configs.cfg import get_cfg_defaults 
from dataio.dataloader import get_dataloaders
from model.multimodal import construct_ppnet
from prototype.push import decode_onehot
from train.optimizer import get_optimizers
import train.train_and_test as tnt
import warnings

from utils.util import does_it_match, find_node, find_node_hierarchical
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

run_mode = {1 : "genetic", 2 : "image", 3 : "multimodal"} 
log = print

def get_last_layer_weight_matrix(node):
    return node.last_layer.weight


def record(cfg, dataloader, name):
    all_vectors = {
    }
    data_object = {}
    location = [0, 2, 0]
    # location = [0]
    depth = len(location)

    model = construct_ppnet(cfg, log).cuda()
    log("ProtoPNet constructed")

    node = find_node(model, location)
    print("Node:", node.taxonomy)
    print("Children:", len(node.childs))
    all_vectors["prototype"] = node.prototype.detach().cpu().numpy()


    child_names = [
        child.taxonomy for child in node.childs
    ]

    data_object["class_names"] = child_names
    data_object["location"] = location

    for (genetics, image), (label, flat_label) in tqdm(dataloader):
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

            last_layer_weight_matrix = get_last_layer_weight_matrix(node)
            last_layer_weight_matrix = last_layer_weight_matrix.cpu().numpy()


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
    
    data_object["classes"] = all_vectors
    # Save the vectors to a file
    with open(name, 'wb') as file:
        pickle.dump(data_object, file)

def position(cfg, dataloader, root=[]):
    hierarchy = dataloader.dataset.hierarchy
    
    node = find_node_hierarchical(hierarchy, [])

    if node is None:
        raise ValueError("Node not found")
    
    # Num classes
    num_classes = len(node.childs)

    COUNT_PER_CLASS = 4
    class_strings = [[] for _ in range(num_classes)]
    depth = len(root)

    for (genetics, _), (label, _) in dataloader:
        for i in range(genetics.shape[0]):
            # if not does_it_match(label[i], root):
            #     continue

            if len(class_strings[label[i][len(root)].item()]) < COUNT_PER_CLASS:
                class_strings[label[i][len(root)].item()].append(decode_onehot(genetics[i]))

    print ("Class strings:", class_strings)

    width = 18

    for i in range(720 // width):
        print("Location:", i)
        for arr in class_strings:
            print(" ".join([sample[i*width:(i+1)*width] for sample in arr]))
        
        # Wait for the user to hit enter
        input()

def main(cfg: CfgNode, script: str, name:str):
    train_loader, _, _, _ = get_dataloaders(cfg, log)
    log("Dataloaders Constructed")

    if script == "record":
        record(cfg, train_loader, name)
    elif script == "position":
        position(cfg, train_loader)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=str, help="What to do? [record, position]", choices=["record", "position"])
    parser.add_argument("src", type=str, help="Path to the source model")
    parser.add_argument('--type', type=str, default="genetic", choices=["genetic", "image"])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--name', type=str, default="vectors")

    args = parser.parse_args()

    # Extract filename from args.src using os.path.basename
    src_name = os.path.basename(args.src)[4:]

    # Extract the path to the directory containing the file
    src_dir = os.path.dirname(args.src)

    cfg = get_cfg_defaults()
    if not args.config:
        cfg.merge_from_file(os.path.join(src_dir, "config.yaml"))
    else:
        cfg.merge_from_file(os.path.join("configs", args.config))

    cfg.MODE = (1 if args.type == "genetic" else 2)

    if cfg.MODE == 1:
        cfg.MODEL.GENETIC.PPNET_PATH = os.path.join(src_dir,f"gen_{src_name}")
    elif cfg.MODE == 2:
        cfg.MODEL.IMAGE.PPNET_PATH = os.path.join(src_dir,f"img_{src_name}")

    cfg.DATASET.CACHED_DATASET_FOLDER = "pre_existing_datasets/reasonable_dataset"

    name = os.path.join("pacmap_vectors", args.name + ".pkl")

    main(cfg, args.script, name)  
