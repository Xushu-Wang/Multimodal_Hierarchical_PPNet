import torch
import numpy as np
from torch import Tensor

from model.hierarchical import Mode, ProtoNode, HierProtoPNet
from model.multimodal import MultiHierProtoPNet
from typing import Union

Model = Union[HierProtoPNet, MultiHierProtoPNet, torch.nn.DataParallel]

def find_closest_conv_feature(node: ProtoNode, conv_features: Tensor, label: Tensor, stride: int):
    """
    For each prototype, find its best patch in the backbone outputs and calculate 
    minimum distance batch. But not projecting yet. 
    """
    mode = node.mode 

    # mask out the samples in the batch that are not relevant
    mask = torch.ones(label.size(0), dtype=torch.bool)      # [80]
    for i, class_index in enumerate(node.taxnode.idx):     # 
        mask &= (label[:,i] == class_index).bool()
    mask = mask.bool()
    label = label[mask]

    level = node.taxnode.depth

    num_classes = node.nclass

    with torch.no_grad():
        # this computation currently is not parallelized
        proto_dist_torch = node.push_get_dist(conv_features)

    protoL_input_ = np.copy(conv_features.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del conv_features, proto_dist_torch

    # class idx -> sample idx (in filtered batch)
    class_to_img_idx = {key+1: [] for key in range(num_classes)}
    for img_index, img_y in enumerate(label):
        img_label = int(img_y[level].item())
        if img_label > 0:
            class_to_img_idx[img_label].append(img_index)

    proto_h, proto_w = node.pshape[1], node.pshape[2] 

    for j in range(node.nprotos):
        # We assume class_specifc is true
        # target_class is the class of the class_specific prototype
        target_class = int(torch.argmax(node.match[j]).item()) + 1
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_idx[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_idx[target_class]][:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < node.global_min_proto_dist[j]:
            if mode == Mode.GENETIC:
                batch_argmin_proto_dist_j = [np.argmin(proto_dist_j[:,0,0]),0,(j % (node.nprotos))]
            else:
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                        proto_dist_j.shape))
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist_j[0] = class_to_img_idx[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                :,
                                                fmap_height_start_index:fmap_height_end_index,
                                                fmap_width_start_index:fmap_width_end_index]

            node.global_min_proto_dist[j] = batch_min_proto_dist_j
            node.global_min_fmap_patches[j] = batch_min_fmap_patch_j

def push(
    model: Model, 
    dataloader, 
    preprocessor = None, 
    stride: int = 1
):
    model.eval() 
    mode = Mode(model.mode) 
    match mode: 
        case Mode.GENETIC: 
            return push_genetic(model, dataloader, None, stride) 
        case Mode.IMAGE: 
            return push_image(model, dataloader, preprocessor, stride) 
        case Mode.MULTIMODAL: 
            return push_multimodal(model, dataloader, preprocessor, stride) 


def push_genetic(model, dataloader, _, stride): 
    for push_iter, ((genetics, _), (label, _)) in enumerate(dataloader):
        input = genetics.cuda() 

        with torch.no_grad(): 
            conv_features = model.conv_features(input) 

        for node in model.classifier_nodes:  
            find_closest_conv_feature(
                node=node,
                conv_features=conv_features,
                label=label,
                stride=stride,
            )

    for node in model.classifier_nodes:
        prototype_update = np.reshape(node.global_min_fmap_patches, node.prototype.shape)
        node.prototype.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

def push_image(model, dataloader, preprocessor, stride): 
    for push_iter, ((_, image), (label, _)) in enumerate(dataloader): 
        image = preprocessor(image) if preprocessor else image
        input = image.cuda()

        with torch.no_grad(): 
            conv_features = model.conv_features(input) 

        for node in model.classifier_nodes:  
            find_closest_conv_feature(
                node=node,
                conv_features=conv_features,
                label=label,
                stride=stride,
            )

    for node in model.classifier_nodes:
        prototype_update = np.reshape(node.global_min_fmap_patches, node.prototype.shape)
        node.prototype.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

def push_multimodal(model, dataloader, preprocessor, stride): 
    for push_iter, ((genetics, image), (label, _)) in enumerate(dataloader):
        image = preprocessor(image) if preprocessor else image 
        gen_input = genetics.cuda()
        img_input = image.cuda()

        with torch.no_grad(): 
            gen_conv_features, img_conv_features = model.conv_features(gen_input, img_input)
        
        # for each node, find the prototype that it should project to
        for node in model.classifier_nodes:  
            find_closest_conv_feature(
                node=node.gen_node,
                conv_features=gen_conv_features,
                label=label,
                stride=stride
            )
            find_closest_conv_feature(
                node=node.img_node,
                conv_features=img_conv_features,
                label=label,
                stride=stride
            )

    for node in model.classifier_nodes: 
        # project the prototypes in each node to prototype_update 

        # genetic prototypes projection 
        gen_prototype_update = np.reshape(node.gen_node.global_min_fmap_patches, node.gen_node.prototype.shape)
        node.gen_node.prototype.data.copy_(torch.tensor(gen_prototype_update, dtype=torch.float32).cuda())

        # img prototypes projection 
        img_prototype_update = np.reshape(node.img_node.global_min_fmap_patches, node.img_node.prototype.shape)
        node.img_node.prototype.data.copy_(torch.tensor(img_prototype_update, dtype=torch.float32).cuda())


