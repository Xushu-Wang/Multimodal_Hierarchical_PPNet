from model.features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features, resnet_bioscan_features
from model.features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model.features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

base_architecture_to_features = {
    'resnet18': resnet18_features,
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
    'vgg19_bn': vgg19_bn_features
}

