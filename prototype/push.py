import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from torchvision.transforms import transforms
from yacs.config import CfgNode

from model.hierarchical import Mode, ProtoNode, HierProtoPNet
from model.multimodal import MultiHierProtoPNet
from typing import Union

from prototype.receptive_field import compute_rf_prototype
from utils.util import find_high_activation_crop

Model = Union[HierProtoPNet, MultiHierProtoPNet, torch.nn.DataParallel]

nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

def decode_onehot(onehot, three_dim=True):
    """
    Convert a one-hot encoded nucleotide sequence to a string
    """
    if three_dim:
        onehot = onehot[:, 0, :]
    # Add another row encoding whether the nucleotide is unknown
    onehot = np.vstack([np.zeros(onehot.shape[1]), onehot])
    # Make the unknown nucleotide 1 if all other nucleotides are 0
    onehot[0] = 1 - onehot[1:].sum(0)
    return "".join([list(nucleotides.keys())[list(nucleotides.values()).index(i)] for i in onehot.argmax(0)])

def find_closest_conv_feature(
    model:Model, 
    node: ProtoNode, 
    conv_features: Tensor,      # [B, 2048, 8, 8]
    input: Tensor,              # [B, 3, 256, 256]
    label: Tensor,              # [B, 4]
    stride: int, 
    epoch: int, 
    cfg: CfgNode
):
    """
    For each prototype, find its best patch in the backbone outputs and calculate 
    maximum similarity batch. But not projecting yet.  
    Node A predicting X, Y, Z, each with 10 prototypes, so a total of 30 prototypes 
    For each node, loop through dataset of samples in clases X, Y, Z, and find the 
    global minimum distance prototype. 
    label - [20, 4] = [batch size, levels] 
    """ 
    if conv_features.size(0) != input.size(0) and input.size(0) != label.size(0): 
        raise Exception("Batch sizes do not match")

    mode = node.mode 
    level = node.taxnode.depth 
    protoL_input_ = np.copy(conv_features.detach().cpu().numpy())         # 80x3x256x256
    proto_sim_ = np.copy(node.cos_sim(conv_features).detach().cpu().numpy())  # [20, 10*6, 8, 8]

    # class idx -> sample idx (in filtered batch)
    class_to_img_idx = {key: [] for key in range(node.nclass)}
    for img_index, img_y in enumerate(label):
        img_label = int(img_y[level].item())
        class_to_img_idx[img_label].append(img_index)

    proto_h, proto_w = node.pshape[1], node.pshape[2] 

    # Create a list of dictionaries to store the patch for saving
    patch_df_list = []
    node_file_dir = os.path.join(cfg.OUTPUT.IMG_DIR, f'epoch-{epoch}',*node.taxnode.full_taxonomy, "prototypes") 
    for c in range(node.nclass): 
        subclass_mask = label[:,level] == c 
        if len(label[subclass_mask]) == 0: 
            continue 
        for p in range(node.nprotos): 
            j = c * node.nprotos + p  
            # [B, 10c, 8, 8] -> [B, 8, 8] 
            # only consider cosine similarity of jth prototype with 
            # all patches (8x8) over all samples in batch (B)
            proto_sim_j = proto_sim_[class_to_img_idx[c]][:,j,:,:]
            batch_max_proto_sim_j = np.amax(proto_sim_j)

            if batch_max_proto_sim_j > node.global_max_proto_sim[j]:  
                node.global_max_proto_sim[j] = batch_max_proto_sim_j 

                # now find the argmax index in subclass batch 
                if mode == Mode.GENETIC:
                    batch_argmax_proto_sim_j = [np.argmax(proto_sim_j[:,0,0]), 0, p] 
                else:
                    # [0, 4, 3]
                    batch_argmax_proto_sim_j = \
                        list(np.unravel_index(np.argmax(proto_sim_j), proto_sim_j.shape))  

                # change argmax index from index among images of target class 
                # to the index in the entire search batch. 
                batch_argmax_proto_sim_j[0] = class_to_img_idx[c][batch_argmax_proto_sim_j[0]]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_sim_j[0]
                fmap_height_start_index = batch_argmax_proto_sim_j[1] * stride
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmax_proto_sim_j[2] * stride
                fmap_width_end_index = fmap_width_start_index + proto_w

                batch_max_fmap_patch_j = protoL_input_[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index
                ]

                node.global_max_fmap_patches[j] = batch_max_fmap_patch_j

                if not cfg.OUTPUT.SAVE_IMAGES: 
                    continue

                # Add patch information to patch_df_list to allow for saving patches
                if mode == Mode.GENETIC:
                    assert conv_features.shape[2] == 1
                    protoL_rf_info = model.proto_layer_rf_info

                    # get the whole image
                    original_img_j = input[batch_argmax_proto_sim_j[0]]
                    original_img_j = original_img_j.cpu().numpy()

                    # crop out the receptive field
                    rf_img_j = original_img_j[:, 0, (j % (node.nprotos)) * protoL_rf_info[1]: (j % (node.nprotos) + 1) * protoL_rf_info[1]]

                    string_prototype = decode_onehot(rf_img_j, False)

                    patch_df_list.append(
                        {
                            "key": j,
                            "class_index": j // (node.nprotos),
                            "prototype_index": j % (node.nprotos),
                            "patch": string_prototype
                        }
                    )
                else: 
                    # Save Images
                    # Get the receptive field boundary of the image patch
                    # that generates the representation
                    protoL_rf_info = model.proto_layer_rf_info
                    rf_prototype_j = compute_rf_prototype(input.size(2), batch_argmax_proto_sim_j, protoL_rf_info)

                    # Get the whole image
                    original_img_j = input[rf_prototype_j[0]]
                    original_img_j = original_img_j.cpu().numpy()
                    original_img_j = np.transpose(original_img_j, (1, 2, 0))
                    original_img_size = original_img_j.shape[0]

                    # Crop out the receptive field
                    rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                            rf_prototype_j[3]:rf_prototype_j[4], :]
                    
                    # Find the highly activated region of the original image
                    proto_sim_img_j = proto_sim_[img_index_in_batch, j, :, :]
                    proto_act_img_j = node.pshape[0] * node.pshape[1] * node.pshape[2] - proto_sim_img_j
                    
                    upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                    interpolation=cv2.INTER_CUBIC)

                    proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                    
                    # Crop out the image patch with high activation as prototype image
                    proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                                proto_bound_j[2]:proto_bound_j[3], :]
                    
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(node_file_dir,
                                        cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + str(j) + '.npy'),
                            proto_act_img_j)
                    
                    if cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX is not None:
                        # save the image of the prototype
                        cv2.imwrite(os.path.join(node_file_dir,
                                                cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + str(j) + '.png'),
                                    cv2.cvtColor(proto_img_j, cv2.COLOR_RGB2BGR))
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[...,::-1]
                        # Clamp heatmap to 0-1
                        heatmap = np.clip(heatmap, 0.0, 1.0)
                        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

                        plt.imsave(os.path.join(node_file_dir,
                                                cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + '-original_with_self_act' + str(j) + '.png'),
                                overlayed_original_img_j,
                                vmin=0.0,
                                vmax=1.0)
                        
                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                            plt.imsave(os.path.join(node_file_dir,
                                                    cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + '-receptive_field' + str(j) + '.png'),
                                    rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                            overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                        rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(node_file_dir,
                                                    cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + '-receptive_field_with_self_act' + str(j) + '.png'),
                                    overlayed_rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                        
                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(node_file_dir,
                                                cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + str(j) + '.png'),
                                proto_img_j,
                                vmin=0.0,
                                vmax=1.0)

    del conv_features

    # Save the genetic prototypes
    # Get the directory for saving prototypes for this node for this epcoh
    if len(patch_df_list) and cfg.OUTPUT.SAVE_IMAGES:
        patch_df = pd.DataFrame(patch_df_list, columns=["key", "class_index", "prototype_index", "patch"])
        if os.path.isfile(os.path.join(node_file_dir, cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + ".csv")):
            # Update old file
            existing_df = pd.read_csv(os.path.join(node_file_dir, cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + ".csv"))
            existing_df = existing_df.set_index("key")
            patch_df = patch_df.set_index("key")

            # Combine the two
            existing_df = existing_df.drop(patch_df.index, errors="ignore")
            existing_df = pd.concat([existing_df,patch_df])

            patch_df = existing_df.reset_index()
        # Create the parent directory if it does not exist
        os.makedirs(node_file_dir, exist_ok=True)
        # Save the patch information
        patch_df.to_csv(os.path.join(node_file_dir, cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX + ".csv"), index=False)

def push(
    model: Model, 
    dataloader, 
    cfg,
    stride: int = 1,
    epoch = 0
):
    model.eval() 
    mode = Mode(model.mode) 

    for node in model.classifier_nodes:
        node.init_push()

    with torch.no_grad(): 
        match mode: 
            case Mode.GENETIC: 
                return push_genetic(model, dataloader, stride, epoch, cfg) 
            case Mode.IMAGE: 
                return push_image(model, dataloader, stride, epoch, cfg) 
            case Mode.MULTIMODAL: 
                return push_multimodal(model, dataloader, stride, epoch, cfg) 

def push_genetic(model, dataloader, stride, epoch, cfg): 
    for (genetics, _), (label, _) in dataloader:
        input = genetics.to(cfg.DEVICE)

        conv_features = model.conv_features(input) 

        for node in model.classifier_nodes:  
            find_closest_conv_feature(
                model=model,
                node=node,
                input=input,
                conv_features=conv_features,
                label=label,
                stride=stride,
                epoch=epoch,
                cfg=cfg
            )

    for node in model.classifier_nodes:
        prototype_update = np.reshape(node.global_max_fmap_patches, node.prototype.shape)
        node.prototype.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(cfg.DEVICE))

def push_image(model, dataloader, stride, epoch, cfg): 
    normalize = transforms.Normalize(
        mean=cfg.DATASET.TRANSFORMS.IMAGE.MEAN,
        std=cfg.DATASET.TRANSFORMS.IMAGE.STD
    )

    for (_, raw_image), (label, _) in dataloader: 
        input = normalize(raw_image).to(cfg.DEVICE)
        raw_image = raw_image.to(cfg.DEVICE)

        conv_features = model.conv_features(input) 

        for node in model.classifier_nodes:
            find_closest_conv_feature(
                model=model,
                node=node,
                input=raw_image,
                conv_features=conv_features,
                label=label,
                stride=stride,
                epoch=epoch,
                cfg=cfg
            )

    for node in model.classifier_nodes:
        prototype_update = np.reshape(node.global_max_fmap_patches, node.prototype.shape)
        node.prototype.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(cfg.DEVICE))

def push_multimodal(model, dataloader, stride, epoch, cfg): 
    normalize = transforms.Normalize(
        mean=cfg.DATASET.TRANSFORMS.IMAGE.MEAN,
        std=cfg.DATASET.TRANSFORMS.IMAGE.STD
    )

    for ((genetics, raw_image), (label, _)) in tqdm(dataloader):
        gen_input = genetics.to(cfg.DEVICE)
        img_input = normalize(raw_image).to(cfg.DEVICE)
        raw_image = raw_image.to(cfg.DEVICE)

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

            find_closest_conv_feature(
                model=model.gen_net,
                node=node.gen_node,
                conv_features=f_gen_conv_features,
                input=f_gen_input,
                label=f_label,
                stride=stride,
                epoch=epoch,
                cfg=cfg
            )
            find_closest_conv_feature(
                model=model.img_net,
                node=node.img_node,
                conv_features=f_img_conv_features,
                input=f_raw_image,
                label=f_label,
                stride=stride,
                epoch=epoch,
                cfg=cfg
            )

    for node in model.classifier_nodes: 
        # project the prototypes in each node to prototype_update 
        node.gen_node.prototype.data.copy_(
            torch.tensor(node.gen_node.global_max_fmap_patches, dtype=torch.float32).to(cfg.DEVICE)
        )

        node.img_node.prototype.data.copy_(
            torch.tensor(node.img_node.global_max_fmap_patches, dtype=torch.float32).to(cfg.DEVICE)
        )
