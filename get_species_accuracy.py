"""
This is a temporary script to correctly calculate probabilistic accuracy.

The probabilistic accuracy printed during training is incorrect. This (slowly) calculates it correctly. 

This should be gone soon.
"""
import argparse

import torch

from configs.cfg import get_cfg_defaults
from dataio.dataloader import get_dataloaders
from model.model import construct_tree_ppnet
from torchvision.models import resnet50

MODEL_PATHS = (
    # "../output/image_only/image_species_new_backbone/images/10_push_weights.pth",
    # "../output/image_only/image_species_120_80train_warm_001/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_no_init_warm_001/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_no_init_warm_002/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_10epoch_init/images/5_push_weights.pth",
    # "../output/joint/parallel_new_backbone_10epoch_init_warm/images/10_push_weights.pth",
    # "../output/joint/parallel_new_backbone_10epoch_init_nopush_1000_002/50nopush0.3333.pth", 
    # "../output/joint/parallel_new_backbone_10epoch_init_nopush_80/50nopush0.3333.pth",
    "../output/joint/parallel_new_backbone_10epoch_init_067/50nopush0.3333.pth",

)
MODEL_NAMES = (
    # "image_bare",
    # "image_bare_warm",
    # "image_no_init",
    # "image_no_init_warm",
    # "image_init",
    # "image_init_warm",
    "1000",
    "80"
)
IS_HIERARCHICAL = (
    # False,
    # False,
    # True,
    # True,
    # True,
    # True,
    True,
    True
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

def get_level_mask(ppnet, indexed_tree, level, label):
    mask = torch.ones((label.shape[0], 113)).int().cuda()

    for node in ppnet.nodes_with_children:
        if len(node.child_nodes) == 0:
            for child in node.all_child_nodes:
                bit = indexed_tree
                for loc in child.named_location:
                    bit = bit[loc]
                
                location = bit["idx"]

                for i in range(level):
                    matching = child.int_location[i] == label[:, i]
                    mask[:,location] = mask[:,location] * matching

    return mask

def get_blackbox(path):
    model = resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 113)

    # Load weights
    model.load_state_dict(torch.load(path))

    return model.cuda()

def get_model(cfg, name, path, is_hierarchical):
    if is_hierarchical:
        cfg.DATASET.MODE = 3
        cfg.MODEL.MULTI.MULTI_PPNET_PATH = path
        joint_tree_ppnet = construct_tree_ppnet(cfg)
        return joint_tree_ppnet.image_hierarchical_ppnet.to("cuda")
    else:
        cfg.DATASET.IMAGE.PPNET_PATH = path
        return construct_tree_ppnet(cfg).to("cuda")

def get_tree_out_array(image, tree_ppnet, indexed_tree):
    conv_features = tree_ppnet.conv_features(image)

    recursive_throw_probs_on_there(
        tree_ppnet.root,
        conv_features,
    )

    out = []
    indicies = []

    for node in tree_ppnet.nodes_with_children:
        if len(node.child_nodes) == 0:
            for child in node.all_child_nodes:
                bit = indexed_tree
                for loc in child.named_location:
                    bit = bit[loc]
                
                location = bit["idx"]
                out.append(child.prob)
                indicies.append(location)

    # Sort out by index
    out = [x for _, x in sorted(zip(indicies, out))]
    out_array = torch.stack(out, dim=1)

    return out_array

def eval_model(eval_blackbox, indexed_tree, name, model, dataloader):
    with torch.no_grad():
        tot_correct = [0,0,0,0]
        total = [0,0,0,0]

        for ((_, image), label) in dataloader:
            # print(label.shape)
            full_label = label[:, :4]
            # label = label[:, 4]

            image = image.to("cuda")
            label = label.to("cuda")

            full_label = full_label.to("cuda")

            if not eval_blackbox:
                out_array = get_tree_out_array(image, model, indexed_tree)
            else:
                out_array = model(image)
                out_array = torch.nn.functional.softmax(out_array, dim=1)

            for cond_level in range(4):
                if not eval_blackbox:
                    mask = get_level_mask(model, indexed_tree, cond_level, full_label)
                else:
                    mask = dataloader.dataset.get_species_mask(label[:, cond_level], cond_level).cuda()
                    # mask = torch.ones_like(mask)

                # Accuracy
                # print(f"----{cond_level}----")
                # print(label)
                # print(out_array[0, :100])
                # print((out_array * mask)[0, :100])
                _, predicted = torch.max(out_array * mask, 1)
                total[cond_level] += label.size(0)
                correct = (predicted == label[:,cond_level]).sum().item()
                tot_correct[cond_level] += correct
                
        for cond_level in range(4):
            print(f"[{cond_level}] {name}: {tot_correct[cond_level]/total[cond_level]:.6f}")


def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image_species.yaml')
    parser.add_argument('--eval-blackbox', action="store_true")
    args = parser.parse_args()
    cfg.merge_from_file(args.configs)

    # Get dataloaders
    train_loader, train_push_loader, val_loader, test_loader, image_normalizer = get_dataloaders(cfg, print, flat_class=True)

    indexed_tree = train_loader.dataset.leaf_indicies

    blackbox = get_blackbox("/home/users/cjb131/school/cs474/Multimodal_Hierarchical_PPNet/backbones/image_species_120_80train_028/image_species_120_80train_028_best.pth")

    if not args.eval_blackbox:
        for name, path, is_hierarchical in zip(MODEL_NAMES, MODEL_PATHS, IS_HIERARCHICAL):
            cfg.DATASET.IMAGE.PPNET_PATH = path
            tree_ppnet = get_model(cfg, name, path, is_hierarchical)

            eval_model(False, indexed_tree, name, tree_ppnet, val_loader)
    else:
        eval_model(True, indexed_tree, "blackbox", blackbox, val_loader)

main()
