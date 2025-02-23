import argparse
import torch
import numpy as np
from configs.cfg import get_cfg_defaults
from model.model import Mode, construct_tree_ppnet
from dataio.dataloader import get_dataloaders

LOCATION = [1]
PROTOTYPE = 4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/analysis.yaml')
    parser.add_argument('--same-class-only', action='store_true')
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.configs)

    _, _, val_loader, _, image_normalizer = get_dataloaders(cfg, print)

    print("Loading Model")
    # model = construct_tree_ppnet(cfg).cuda()
    model = torch.load(cfg.MODEL.MULTI.MULTI_PPNET_PATH)

    print("Model Loaded")
    prototype_node = find_node(model, LOCATION)
    print(prototype_node.genetic_tree_node.last_layer.weight)
    print(prototype_node.image_tree_node.last_layer.weight)
    samples, min_distances = find_best_samples_and_their_activations(model, val_loader, args.same_class_only, prototype_node, n=10)

    # Do some image visualizations with this correspondence

    for s,md in zip(samples, min_distances):
        top_10 = torch.argsort(md[0])[:10]
        print(s[2])
        print(top_10)
        # print(prototype_node.genetic_tree_node.last_layer.weight[:, top_10])

def find_node(model, location):
    for node in model.get_nodes_with_children():
        print(node.int_location)
        if len(node.int_location) == len(location) and all([a == b for a, b in zip(node.int_location, location)]):
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
            conv_features = model.conv_features(input)

            logits, min_distances = prototype_node.get_logits(conv_features[0], conv_features[1])

            relevant_distance = min_distances[1][:, PROTOTYPE]

            for i in range(len(relevant_distance)):
                if relevant_distance[i] < smallest_distances[-1] and (matches(label[i], LOCATION) or not same_class_only):
                    print(relevant_distance[i], label[i])
                    print(torch.argmax(logits, dim=1)[i])
                    smallest_distances[-1] = relevant_distance[i]
                    best_samples[-1] = (input[0][i], input[1][i], label[i])
                    distance_objects[-1] = (min_distances[0][i], min_distances[1][i])
                    sorted_indices = torch.argsort(smallest_distances)
                    smallest_distances = smallest_distances[sorted_indices]
                    best_samples = [best_samples[j] for j in sorted_indices]
                    distance_objects = [distance_objects[j] for j in sorted_indices]
    
    return best_samples, distance_objects





if __name__ == '__main__':
    main()