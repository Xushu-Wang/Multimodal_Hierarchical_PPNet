"""
This is a temporary script to correctly calculate probabilistic accuracy.

The probabilistic accuracy printed during training is incorrect. This (slowly) calculates it correctly. 

This should be gone soon.
"""

import argparse

import torch

from configs.cfg import get_cfg_defaults
from dataio.tree import get_dataloaders
from model.model import construct_tree_ppnet


MODEL_PATHS = (
    # "../output/image_only/image_species_new_backbone/images/10_push_weights.pth",
    # "../output/image_only/image_species_120_80train_warm_001/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_no_init_warm_001/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_no_init_warm_002/images/10_push_weights.pth",
    "../output/joint/parallel_new_backbone_10epoch_init/images/5_push_weights.pth",
    # "../output/joint/parallel_new_backbone_10epoch_init_warm/images/5_push_weights.pth",
)
MODEL_NAMES = (
    # "image_bare",
    # "image_bare_warm",
    # "image_no_init",
    # "image_no_init_warm",
    "image_init",
    # "image_init_warm"
)
IS_HIERARCHICAL = (
    # False,
    # False,
    # True,
    # True,
    True,
    # True
)

def recursive_throw_probs_on_there(tree, conv_features, above_prob=1):
    logits, _ = tree(conv_features)
    probs = torch.softmax(logits, dim=1) * above_prob
    tree.probs = probs

    for child in tree.all_child_nodes:
        if child.parent:
            recursive_throw_probs_on_there(child, conv_features, probs[:, child.int_location[-1]-1].unsqueeze(1))
        else:
            child.prob = probs[:, child.int_location[-1]-1]


def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image_species.yaml')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    cfg.merge_from_file(args.configs)

    # Get dataloaders
    train_loader, train_push_loader, val_loader, test_loader, image_normalizer = get_dataloaders(cfg, print, flat_class=True)

    indexed_tree = val_loader.dataset.leaf_indicies

    for name, path, is_hierarchical in zip(MODEL_NAMES, MODEL_PATHS, IS_HIERARCHICAL):
        cfg.DATASET.IMAGE.PPNET_PATH = path

        if is_hierarchical:
            print("AAA")
            cfg.DATASET.MODE = 3
            cfg.MODEL.MULTI.MULTI_PPNET_PATH = path
            joint_tree_ppnet = construct_tree_ppnet(cfg)
            tree_ppnet = joint_tree_ppnet.image_hierarchical_ppnet.to("cuda")
        else:
            cfg.DATASET.IMAGE.PPNET_PATH = path
            tree_ppnet = construct_tree_ppnet(cfg).to("cuda")

        with torch.no_grad():
            tot_correct = 0
            total = 0

            for ((_, image), label) in train_loader:
                image = image.to("cuda")
                label = label.to("cuda")

                conv_features = tree_ppnet.conv_features(image)

                recursive_throw_probs_on_there(tree_ppnet.root, conv_features)

                out = []
                indecies = []

                for node in tree_ppnet.nodes_with_children:
                    if len(node.child_nodes) == 0:
                        for child in node.all_child_nodes:
                            bit = indexed_tree
                            for loc in child.named_location:
                                bit = bit[loc]
                            
                            location = bit["idx"]
                            out.append(child.prob)
                            indecies.append(location)

                # Sort out by index
                out = [x for _, x in sorted(zip(indecies, out))]
                out_array = torch.stack(out, dim=1)
                
                # Accuracy
                _, predicted = torch.max(out_array, 1)
                total += label.size(0)
                correct = (predicted == label).sum().item()
                tot_correct += correct

            print(f"{name}: {tot_correct/total:.6f}")

main()