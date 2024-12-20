import json
import torch

from model.features.genetic_features import GeneticCNN2D
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.hierarchical_ppnet import Hierarchical_PPNet, Multi_Hierarchical_PPNet, Mode

from model.features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features, resnet_bioscan_features
from model.features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model.features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'resnetbioscan': resnet_bioscan_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

def construct_genetic_tree_ppnet(cfg): 
    class_specification = json.load(open(cfg.DATASET.TREE_SPECIFICATION_FILE, "r")) 

    if cfg.DATASET.GENETIC.PPNET_PATH != "NA":
        genetic_ppnet = torch.load(cfg.DATASET.GENETIC.PPNET_PATH)
    else:
        # Genetics Mode
        m = GeneticCNN2D(720, 1, include_connected_layer=False)
    
        # Remove the fully connected layer
        weights = torch.load(cfg.MODEL.GENETIC_BACKBONE_PATH)

        for k in list(weights.keys()):
            if "conv" not in k:
                del weights[k]
        
        m.load_state_dict(weights)

        # NOTE - Layer_paddings is different from the padding in the image models
        layer_filter_sizes, layer_strides, layer_paddings = m.conv_info()

        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=720,
                                                        layer_filter_sizes=layer_filter_sizes,
                                                        layer_strides=layer_strides,
                                                        layer_paddings=layer_paddings,
                                                        prototype_kernel_size=cfg.DATASET.GENETIC.PROTOTYPE_SHAPE[1])
        genetic_ppnet = Hierarchical_PPNet(
            features=m,
            img_size=0,
            prototype_shape=cfg.DATASET.GENETIC.PROTOTYPE_SHAPE,
            num_prototypes_per_class=cfg.DATASET.GENETIC.NUM_PROTOTYPES_PER_CLASS,
            class_specification=class_specification,
            proto_layer_rf_info=proto_layer_rf_info,
            mode=Mode.GENETIC,
            max_num_prototypes_per_class=cfg.DATASET.GENETIC.MAX_NUM_PROTOTYPES_PER_CLASS,
        ) 

    return genetic_ppnet

def construct_image_tree_ppnet(cfg): 
    class_specification = json.load(open(cfg.DATASET.TREE_SPECIFICATION_FILE, "r")) 

    if cfg.DATASET.IMAGE.PPNET_PATH != "NA":
        image_ppnet = torch.load(cfg.DATASET.IMAGE.PPNET_PATH)
        if image_ppnet.mode == 3:
            image_ppnet = image_ppnet.image_hierarchical_ppnet
    else:
        # Image Mode
        features = base_architecture_to_features["resnetbioscan"](pretrained=True)
        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=cfg.DATASET.IMAGE.SIZE,
                                                            layer_filter_sizes=layer_filter_sizes,
                                                            layer_strides=layer_strides,
                                                            layer_paddings=layer_paddings,
                                                            prototype_kernel_size=cfg.DATASET.IMAGE.PROTOTYPE_SHAPE[1])

        image_ppnet = Hierarchical_PPNet(
            features=features,
            img_size=cfg.DATASET.IMAGE.SIZE,
            prototype_shape=cfg.DATASET.IMAGE.PROTOTYPE_SHAPE,
            num_prototypes_per_class=cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS,
            class_specification=class_specification,
            proto_layer_rf_info=proto_layer_rf_info,
            mode=Mode.IMAGE,
            max_num_prototypes_per_class=cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS,
        )

    return image_ppnet

def construct_tree_ppnet(cfg, log=print):
    mode = Mode(cfg.DATASET.MODE) 

    if mode == Mode.GENETIC:
        return construct_genetic_tree_ppnet(cfg)
    elif mode == Mode.IMAGE:
        return construct_image_tree_ppnet(cfg)
    elif mode == Mode.MULTIMODAL: 
        multi = Multi_Hierarchical_PPNet(
            construct_genetic_tree_ppnet(cfg), 
            construct_image_tree_ppnet(cfg) 
        )
        if cfg.MODEL.MULTI.MULTI_PPNET_PATH != "NA":
            if cfg.DATASET.GENETIC.PPNET_PATH != "NA" or cfg.DATASET.IMAGE.PPNET_PATH != "NA":
                log("Warning: Loading MultiModalNetwork, other pretrained networks are ignored...")
            # Load from file, not from state_dict
            multi = torch.load(cfg.MODEL.MULTI.MULTI_PPNET_PATH)
        return multi
    else: 
        raise ValueError("Mode is not valid. This should never happen. ")
