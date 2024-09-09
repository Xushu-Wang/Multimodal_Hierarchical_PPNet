import json
import torch

from model.features.genetic_features import GeneticCNN2D
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.hierarchical_ppnet import Hierarchical_PPNet, Multi_Hierarchical_PPNet

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



def construct_image_ppnet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=512, num_prototypes_per_class=8, 
                    root=None,
                    prototype_distance_function='l2', 
                    prototype_activation_function='log'):
    
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape)
    return Hierarchical_PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_prototypes_per_class=num_prototypes_per_class, 
                 root = root, 
                 proto_layer_rf_info=proto_layer_rf_info,
                 init_weights=True,
                 prototype_distance_function=prototype_distance_function,
                 prototype_activation_function=prototype_activation_function
                 )


def construct_genetic_ppnet(length:int, num_classes:int, prototype_shape, model_path:str, prototype_distance_function = 'cosine', prototype_activation_function='log', fix_prototypes=True):
    m = GeneticCNN2D(length, num_classes, include_connected_layer=False)

    # Remove the fully connected layer
    weights = torch.load(model_path)

    for k in list(weights.keys()):
        if "conv" not in k:
            del weights[k]
    
    m.load_state_dict(weights)

    # NOTE - Layer_paddings is different from the padding in the image models
    layer_filter_sizes, layer_strides, layer_paddings = m.conv_info()

    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=length,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

    return Hierarchical_PPNet(features=m, 
                 img_size=(4, 1, length), 
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info, 
                 num_classes=num_classes,
                 init_weights=True, 
                 prototype_distance_function=prototype_distance_function,
                 prototype_activation_function="linear", 
                 genetics_mode=True,
                 fix_prototypes=fix_prototypes
    )
    
def construct_tree_ppnet(cfg, log=print):
    class_specification = json.load(open(cfg.DATASET.TREE_SPECIFICATION_FILE, "r"))

    if cfg.DATASET.MODE & 1:
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
                mode=1,
                max_num_prototypes_per_class=cfg.DATASET.GENETIC.MAX_NUM_PROTOTYPES_PER_CLASS,
            )

    if cfg.DATASET.MODE >> 1:
        if cfg.DATASET.IMAGE.PPNET_PATH != "NA":
            image_ppnet = torch.load(cfg.DATASET.IMAGE.PPNET_PATH)
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
                mode=2,
                max_num_prototypes_per_class=cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS,
            )

    if cfg.DATASET.MODE == 1:
        return genetic_ppnet
    elif cfg.DATASET.MODE == 2:
        return image_ppnet
    elif cfg.DATASET.MODE == 3:
        multi = Multi_Hierarchical_PPNet(genetic_ppnet, image_ppnet)
        if cfg.MODEL.MULTI.MULTI_PPNET_PATH != "NA":
            if cfg.DATASET.GENETIC.PPNET_PATH != "NA" or cfg.DATASET.IMAGE.PPNET_PATH != "NA":
                log("Warning: Loading MultiModalNetwork, other pretrained networks are ignored...")
            multi.load_state_dict(torch.load(cfg.MODEL.MULTI.MULTI_PPNET_PATH))
        return multi