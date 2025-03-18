import argparse
import math
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from configs.cfg import get_cfg_defaults
from dataio.dataloader import get_dataloaders
from prototype.push import decode_onehot

import pandas as pd

LOCATION = [0, 1]
NAMED_LOCATION = ["Diptera", "Chironomidae"]
PROTOTYPE = 0

def get_genetic_prototype_string(df, class_index, prototype_index):
    return df[(df["class_index"] == class_index) & (df["prototype_index"] == prototype_index)].iloc[0]["patch"]

def save_class_img_prototypes(visualization_path, model_dir, epoch):
    class_index = PROTOTYPE // 40

    paths = [
        os.path.join(model_dir, "images", f"epoch-{epoch}", *NAMED_LOCATION, "prototypes", f"prototype-img{i+10*class_index}.png") for i in range(10)
    ]

    w = math.floor(math.sqrt(len(paths)))
    h = math.ceil(len(paths) / w)

    fig, axs = plt.subplots(h, w, figsize=(20, 20))
    for i, path in enumerate(paths):
        img = plt.imread(path)
        ax = axs[i // w, i % w]
        ax.imshow(img)
        ax.axis('off')
    fig.suptitle(f"{'>'.join(NAMED_LOCATION)} Prototypes", fontsize=30)
    fig.tight_layout()
    # Save the figure
    fig.savefig(os.path.join(visualization_path, "image-prototypes.png"))
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("epoch", type=str)
    parser.add_argument('--configs', type=str, default='configs/multi.yaml')
    parser.add_argument('--same-class-only', action='store_true')
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.configs)

    _, _, val_loader, _, image_normalizer = get_dataloaders(cfg, print)

    print("Loading Model")
    # model = construct_tree_ppnet(cfg).cuda()
    epoch = args.epoch
    model = torch.load(os.path.join(args.model_dir, f"{epoch}_push_full.pth")).cuda()
    relevant_csv = os.path.join(args.model_dir, "images", f"epoch-{epoch}", *NAMED_LOCATION, "prototypes", "prototype-img.csv")

    # Open relevant csv as a pandas dataframe
    genetic_prototypes = pd.read_csv(relevant_csv)

    print("Model Loaded")
    prototype_node = find_node(model, LOCATION)

    # Save the image prototypes
    model_name = args.model_dir.split("/")[-1]
    visualziation_path = f"visualizations/{model_name}/{epoch}/{'_'.join(NAMED_LOCATION)}" 
    os.makedirs(visualziation_path, exist_ok=True)

    save_class_img_prototypes(visualziation_path, args.model_dir, args.epoch)

    samples, min_distances = find_best_samples_and_their_activations(model, val_loader, args.same_class_only, prototype_node, n=10)

    print("Data:")
    md_strs = [get_genetic_prototype_string(genetic_prototypes, i // 40, i % 40) for i in range(len(min_distances[0][0]))]
    md_strs_is_padding = [s.count("N") > 8 for s in md_strs]
    md_strs_is_padding_tensor = torch.tensor(md_strs_is_padding).cuda()
    for s,md in zip(samples, min_distances):
        gen_md = md[0] + md_strs_is_padding_tensor * 1000
        top_10 = torch.argsort(gen_md, descending=False)[:10]

        full_str = decode_onehot(s[0].cpu())

        for i in top_10:
            prototype_index = i.item() % 40
            class_index = i.item() // 40
            print(class_index, prototype_index, full_str[prototype_index*18:(prototype_index+1)*18], md_strs[i], prototype_node.gen_node.last_layer.weight[:,i], prototype_node.gen_node.last_layer.weight.flatten().sort(descending=True)[0][:10])
        print("")

def find_node(model, location):
    for node in model.classifier_nodes:
        if len(node.idx) == len(location) and all([a == b for a, b in zip(node.idx, location)]):
            return node
    return None

def matches(label, location):
    return all([a == b for a, b in zip(label, location)])

def find_best_samples_and_their_activations(model, dataloader, same_class_only, prototype_node, n=10):
    with torch.no_grad():
        model.eval()

        smallest_distances = torch.ones(n) * float("inf")
        best_samples = [None] * n
        distance_objects = [None] * n

        for i, ((genetics, image), (label, _)) in enumerate(dataloader):
            input = (genetics.to("cuda"), image.to("cuda"))
            conv_features = model.conv_features(input[0], input[1])

            logits, min_distances = prototype_node(conv_features[0], conv_features[1])

            relevant_distance = min_distances[1][:, PROTOTYPE]

            for i in range(len(relevant_distance)):
                if relevant_distance[i] < smallest_distances[-1] and (matches(label[i], LOCATION) or not same_class_only):
                    print(relevant_distance[i], label[i])
                    smallest_distances[-1] = relevant_distance[i]
                    best_samples[-1] = (input[0][i], input[1][i], label[i])
                    distance_objects[-1] = (min_distances[0][i], min_distances[1][i])
                    sorted_indices = torch.argsort(smallest_distances)
                    smallest_distances = smallest_distances[sorted_indices]
                    best_samples = [best_samples[j] for j in sorted_indices]
                    distance_objects = [distance_objects[j] for j in sorted_indices]
            
            if (distance_objects[0] != None) and (not matches(label[0], LOCATION)):
                break
    
    return best_samples, distance_objects





if __name__ == '__main__':
    main()
