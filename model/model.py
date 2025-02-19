import json
import torch
from typing import Union
from yacs.config import CfgNode
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from enum import Enum

from model.features.genetic_features import GeneticCNN2D
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.backbones import base_architecture_to_features

import sys

class Mode(Enum):  
    '''
    Enumeration object for labeling ppnet mode.  
    '''
    GENETIC = 1
    IMAGE = 2 
    MULTIMODAL = 3 

class LeafNode(nn.Module):
    def __init__(self, int_location, named_location):
        super().__init__()
        self.int_location = int_location
        self.named_location = named_location
        self.child_nodes = []
        self.all_child_nodes = []
        self.parent = False

class TreeNode(nn.Module): 
    def __init__(
        self,
        int_location: List[int],
        named_location,
        num_prototypes_per_class,
        prototype_shape,
        tree_specification: Dict,
        fix_prototypes,
        max_num_prototypes_per_class
    ):
        """
        int_location: the 2nd child of the 1st class will look like (1,2) (1 indexing)
        named_location: the 2nd child of the 1st class will look like ("class_1", "class_2")
        num_prototypes_per_class: the number of prototypes per class
        prototype_shape: the shape of the prototype (minus the number of prototypes) length 3
        tree_specification: the tree specification
        fix_prototypes: whether to fix the prototypes or not. True when doing genetic and false if image. 
        max_num_prototypes_per_class: for pruning the prototypes, since tends to overfit for genetics. 
        """
        super().__init__()

        if len(prototype_shape) != 3:
            raise ValueError("Prototype_shape must be of length 3, leave out the number of prototypes man, come on.")

        self.int_location = int_location
        self.named_location = named_location
        self.num_classes = len(tree_specification)
        self.tree_specification = tree_specification
        self.num_prototypes = self.num_classes * num_prototypes_per_class
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_shape = prototype_shape
        self.full_prototype_shape = (self.num_prototypes, prototype_shape[0], prototype_shape[1], prototype_shape[2]) 
        self.fix_prototypes = fix_prototypes
        self.parent = True # True for all tree nodes since it's a parent, not a leaf node. 
        self.level = len(int_location)
        self.max_num_prototypes_per_class = max_num_prototypes_per_class
        self._logits = None

        # a 0/1 block of maps between prototype and its corresponding class label
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # may have combination of leaf and non-leaf nodes if some groups are filtered.
        self.child_nodes = [] # a list of TreeNode objects that are direct children. 
        self.all_child_nodes = [] # All child nodes includes leafs, child nodes does not

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.full_prototype_shape),
            requires_grad=True
        )
        # This is used for pruning. We mask the prototypes that are not used.
        # originall all 1s, but iteratively prunes by setting to 0. 
        self.prototype_mask = nn.Parameter(
            torch.ones(self.num_prototypes, 1, 1),
            requires_grad=False
        )

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)
        
        self.set_last_layer_incorrect_connection(-0.5)
        # basically sets self.last_layer to be self.prototype_class_identity but 0s replaced with -0.5s
        self.create_children()

    def get_prototype_parameters(self):
        # Forgive me for this confusing black magic. It just puts all the parameters into a list
        return [self.prototype_vectors] + [param for child in self.child_nodes for param in child.get_prototype_parameters()]

    def get_last_layer_parameters(self):
        return [p for p in self.last_layer.parameters()] + [param for child in self.child_nodes for param in child.get_last_layer_parameters()]

    def create_children(self):
        i = 1 # 0 Is reserved for not_classified
        if self.tree_specification is None:
            return
        for name, child in self.tree_specification.items():
            if child is None:
                self.all_child_nodes.append(LeafNode(int_location=self.int_location + [i],
                                            named_location=self.named_location + [name]))
            else:
                node = TreeNode(int_location=self.int_location + [i],
                                            named_location=self.named_location + [name],
                                            num_prototypes_per_class=self.num_prototypes_per_class,
                                            prototype_shape=self.prototype_shape,
                                            tree_specification=child,
                                            fix_prototypes=self.fix_prototypes,
                                            max_num_prototypes_per_class=self.max_num_prototypes_per_class
                                            )
                self.child_nodes.append(node)
                self.all_child_nodes.append(node)
            i += 1
        
        self.all_child_nodes.sort(key=lambda x: x.int_location)
    
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def find_offsetting_tensor(self, x, normalized_prototypes):
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """
        # TODO - This should really only be done once on initialization.
        # This is a major waste of time
        arange1 = torch.arange(normalized_prototypes.shape[0] // self.num_classes).view((normalized_prototypes.shape[0]  // self.num_classes, 1)).repeat((1, normalized_prototypes.shape[0]  // self.num_classes))
        indices = torch.LongTensor(torch.arange(normalized_prototypes.shape[0]  // self.num_classes))
        arange2 = (arange1 - indices) % (normalized_prototypes.shape[0]  // self.num_classes)
        arange3 = torch.arange(normalized_prototypes.shape[0]  // self.num_classes, x.shape[3])
        arange3 = arange3.view((1, x.shape[3] - normalized_prototypes.shape[0]  // self.num_classes))
        arange3 = arange3.repeat((normalized_prototypes.shape[0]  // self.num_classes, 1))
        
        arange4 = torch.concatenate((arange2, arange3), dim=1)
        arange4 = arange4.unsqueeze(1).unsqueeze(1)
        arange4 = arange4.repeat((1, x.shape[1], x.shape[2], 1))

        arange4 = arange4.repeat((self.num_classes,1,1,1))
        arange4 = arange4.to(x.device)

        return arange4
    
    def find_offsetting_tensor_for_similarity(self, similarities):
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """
        eye = torch.eye(similarities.shape[2])
        eye = 1 - eye
        eye = eye.unsqueeze(0).repeat((similarities.shape[0], 1,1))
        eye = eye.to(torch.int64)

        return eye.to(similarities.device)

    def cosine_similarity(self, x, with_width_dim=False):
        sqrt_dims = (self.full_prototype_shape[2] * self.full_prototype_shape[3]) ** .5
        x_norm = F.normalize(x, dim=1) / sqrt_dims
        normalized_prototypes = F.normalize(self.prototype_vectors, dim=1) / sqrt_dims

        if self.fix_prototypes:
            offsetting_tensor = self.find_offsetting_tensor(x, normalized_prototypes)
            normalized_prototypes = F.pad(normalized_prototypes, (0, x.shape[3] - normalized_prototypes.shape[3], 0, 0))
            normalized_prototypes = torch.gather(normalized_prototypes, 3, offsetting_tensor)
            
            if with_width_dim:
                similarities = F.conv2d(x_norm, normalized_prototypes)
                
                # Take similarities from [80, 1600, 1, 1] to [80, 40, 40, 1]
                similarities = similarities.reshape((similarities.shape[0], self.num_classes, similarities.shape[1] // self.num_classes, 1))
                # Take similarities to [3200, 40, 1]
                similarities = similarities.reshape((similarities.shape[0] * similarities.shape[1], similarities.shape[2], similarities.shape[3]))
                # Take similarities to [3200, 40, 40]
                similarities = F.pad(similarities, (0, x.shape[3] - similarities.shape[2], 0, 0), value=-1)
                similarity_offsetting_tensor = self.find_offsetting_tensor_for_similarity(similarities)

                # print(similarities.shape, similarity_offsetting_tensor.shape)
                similarities = torch.gather(similarities, 2, similarity_offsetting_tensor)
                # Take similarities to [80, 40, 40, 40]
                similarities = similarities.reshape((similarities.shape[0] // self.num_classes, self.num_classes, similarities.shape[1], similarities.shape[2]))

                # Take similarities to [80, 1600, 40]
                similarities = similarities.reshape((similarities.shape[0], similarities.shape[1] * similarities.shape[2], similarities.shape[3]))
                similarities = similarities.unsqueeze(2)

                return similarities

        return F.conv2d(x_norm, normalized_prototypes)

    def distance_2_similarity(self, distances):
        return -1 * distances

    def get_logits(self, conv_features):
        # NOTE: x is conv_features
        similarity = self.cosine_similarity(conv_features)
        max_similarities = F.max_pool2d(similarity,
                        kernel_size=(similarity.size()[2],
                                    similarity.size()[3]))
        # Set pruned prototype similarities to 0.
        max_similarities = self.prototype_mask * max_similarities
        min_distances = -1 * max_similarities

        # for each prototype, finds the spatial location that's closest to the prototype.
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)

        # logits are class logits

        return logits, min_distances
    
    def push_get_dist(self, conv_features):
        similarities = self.cosine_similarity(conv_features)
        distances = -1 * similarities

        return distances

    def forward(self, conv_features):
        """
        This recursively forwards through the network
        """
        logits, min_distances = self.get_logits(conv_features)
        return logits, min_distances
    
    def recursive_forward(self,conv_features):
        return (*self.forward(conv_features), [
            child(conv_features) for child in self.child_nodes
        ])

    def cuda(self, device = None):
        for child in self.child_nodes:
            child.cuda(device)
        return super().cuda(device)

    def to(self, *args, **kwargs):
        for child in self.child_nodes:
            child.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def push_forward(self, conv_features):
        """
        This one is not recursive, because I realized doing it recursive was a bad idea.
        """
        return conv_features, self.push_get_dist(conv_features)

    def __repr__(self):
        return f"TreeNode: [{'>'.join(self.named_location)}]"

class Hierarchical_PPNet(nn.Module):
    def __init__(self, 
                 features,  # resnet/CNN backbone
                 img_size, 
                 prototype_shape,
                 num_prototypes_per_class,
                 class_specification, # the directionary {tree: , levels: }
                 proto_layer_rf_info, # tuple or smth
                 mode: Mode, 
                 max_num_prototypes_per_class=None, 
        ):
        """
        Rearrange logit map maps the genetic class index to the image class index, which will be considered the true class index.
        
        NOTE: Prototype shape is a 3-tuple, it does not include the number of prototypes. This is non-standard.
        """
        
        super().__init__()

        if mode == Mode.MULTIMODAL:
            raise ValueError("Use MultiHierarchical_PPNet for joint mode")
        
        self.tree_specification = class_specification["tree"]
        self.levels = class_specification["levels"]
                
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.genetics_mode = True if mode == Mode.GENETIC else False 
        self.mode = mode

        # always linear since we use cosine similarity 
        self.prototype_activation_function = "linear" # NOTE: This implementation doesn't support l2 loss...

        self.max_num_prototypes_per_class = max_num_prototypes_per_class

        self.num_prototypes_per_class = num_prototypes_per_class
            
        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        # Construct PPNet Tree
        self.root = self.construct_last_layer_tree()
        
        # TODO - Handle different image ppnet sizes you know what I mean
        self.add_on_layers = nn.Sequential()

        self.nodes_with_children = self.get_nodes_with_children()

    def cuda(self, device = None): 
        self.root.cuda(device)
        return super().cuda(device)

    def to(self, *args, **kwargs): 
        self.root.to(*args, **kwargs)
        self.features = self.features.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_nodes_with_children(self):
        nodes_with_children = []
        def get_nodes_with_children_recursive(node):
            if len(node.all_child_nodes) > 0:
                nodes_with_children.append(node)
                for child in node.child_nodes:
                    get_nodes_with_children_recursive(child)
        get_nodes_with_children_recursive(self.root)
        return nodes_with_children

    def construct_last_layer_tree(self):
        """
        This constructs the tree that will be used to determine the last layer.
        """
        root = TreeNode(
            int_location=[],
            named_location=[],
            num_prototypes_per_class=self.num_prototypes_per_class,
            prototype_shape=self.prototype_shape,
            tree_specification=self.tree_specification,
            fix_prototypes=self.genetics_mode,
            max_num_prototypes_per_class=self.max_num_prototypes_per_class
        )
        return root

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)

        return x
            
    def forward(self, x):
        conv_features = self.conv_features(x)
        logit_tree = self.root.recursive_forward(conv_features)

        return logit_tree
    
    def __repr__(self):
        return '<Hierarchical_PPNet>'

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # NOTE - Last Layer Intialization Occurs in TreeNode 
                
    def get_joint_distribution(self):
        batch_size = self.root.logits.size(0)

        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = self.root.logits
        bottom_level = self.root.distribution_over_furthest_descendents(batch_size)    

        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)

        bottom_level = bottom_level[:,idx]        
        
        return top_level, bottom_level
    
    def get_last_layer_parameters(self):
        return self.root.get_last_layer_parameters()
    
    def get_prototype_parameters(self):
        return self.root.get_prototype_parameters()
    
class CombinerTreeNode(nn.Module):
    """
    This implements the NN that connects the genetic and image PPNet
    """
    def __init__(self, genetic_tree_node, image_tree_node):
        super().__init__()
        self.genetic_tree_node = genetic_tree_node
        self.image_tree_node = image_tree_node
        self.int_location = genetic_tree_node.int_location
        self.named_location = genetic_tree_node.named_location
        self.mode = Mode.MULTIMODAL
        self.max_tracker = ([], [])
        self.correlation_count = 0
        self.correlation_table = torch.zeros((40,10)).cuda()

        self.child_nodes = nn.ModuleList() # Note this will be initialized by Multi_Hierarchical_PPNet

        self.multi_last_layer = nn.Linear(2 * self.genetic_tree_node.num_classes, self.genetic_tree_node.num_classes,
                                    bias=False)
        self.logit_class_identity = torch.zeros(2*self.genetic_tree_node.num_classes, self.genetic_tree_node.num_classes)
        for i in range(2*self.genetic_tree_node.num_classes):
            self.logit_class_identity[i, i % self.genetic_tree_node.num_classes] = 1
        # self.last_layer = nn.Linear(self.genetic_tree_node.num_prototypes + self.image_tree_node.num_prototypes, self.genetic_tree_node.num_classes,
        #                             bias=False)
        
        # Create the correspondence map
        self.prototype_ratio = self.genetic_tree_node.num_prototypes / self.image_tree_node.num_prototypes
        assert self.prototype_ratio == int(self.prototype_ratio)
        self.prototype_ratio = int(self.prototype_ratio)

        self.init_last_layer()

    def init_last_layer(self):
        """
        Init corresponding weights with 1, and the rest with -.5 
        """
        positive_one_weights_locations = torch.t(self.logit_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = -0.5
        self.multi_last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def get_logits(self, genetic_conv_features, image_conv_features, get_middle_logits=False):
        genetic_logits, genetic_distances = self.genetic_tree_node.get_logits(genetic_conv_features)
        image_logits, image_distances = self.image_tree_node.get_logits(image_conv_features)

        if get_middle_logits:
            return (genetic_logits, image_logits), (genetic_distances, image_distances)

        logits = self.multi_last_layer(torch.cat((genetic_logits, image_logits), dim=1))

        return logits, (genetic_distances, image_distances)
    
    def forward(self, x, get_middle_logits=False):
        genetic_conv_features, image_conv_features = x

        return self.get_logits(genetic_conv_features, image_conv_features, get_middle_logits=get_middle_logits)

class Multi_Hierarchical_PPNet(nn.Module):

    def __init__(self, genetic_hierarchical_ppnet, image_hierarchical_ppnet):
        super().__init__()
        self.genetic_hierarchical_ppnet = genetic_hierarchical_ppnet
        self.image_hierarchical_ppnet = image_hierarchical_ppnet

        if self.genetic_hierarchical_ppnet.mode == Mode.MULTIMODAL or self.genetic_hierarchical_ppnet.mode == Mode.MULTIMODAL.value:
            self.genetic_hierarchical_ppnet = self.genetic_hierarchical_ppnet.genetic_hierarchical_ppnet
        
        if self.image_hierarchical_ppnet.mode == Mode.MULTIMODAL or self.image_hierarchical_ppnet.mode == Mode.MULTIMODAL.value:
            self.image_hierarchical_ppnet = self.image_hierarchical_ppnet.image_hierarchical_ppnet

        if self.genetic_hierarchical_ppnet.mode != Mode.GENETIC and self.genetic_hierarchical_ppnet.mode != Mode.GENETIC.value:
            raise ValueError("Genetic Hierarchical PPNet must be in genetics mode")
        
        if self.image_hierarchical_ppnet.mode != Mode.IMAGE and self.image_hierarchical_ppnet.mode != Mode.IMAGE.value:
            print(self.image_hierarchical_ppnet.mode)
            raise ValueError("Image Hierarchical PPNet must be in image mode")

        if self.genetic_hierarchical_ppnet.tree_specification != self.image_hierarchical_ppnet.tree_specification:
            raise ValueError("Tree specifications must be the same")

        self.tree_specification = self.genetic_hierarchical_ppnet.tree_specification
        self.levels = self.genetic_hierarchical_ppnet.levels

        self.mode = Mode.MULTIMODAL

        self.root = self.construct_multi_node_tree()

        self.features = nn.ModuleList([self.genetic_hierarchical_ppnet.features, self.image_hierarchical_ppnet.features])
        self.add_on_layers = nn.ModuleList([self.genetic_hierarchical_ppnet.add_on_layers, self.image_hierarchical_ppnet.add_on_layers])

        self.nodes_with_children = self.get_nodes_with_children()

    def get_nodes_with_children(self):
        nodes_with_children = nn.ModuleList()
        def get_nodes_with_children_recursive(node):
            nodes_with_children.append(node)
            for child in node.child_nodes:
                get_nodes_with_children_recursive(child)
        get_nodes_with_children_recursive(self.root)
        return nodes_with_children

    def construct_multi_node_tree(self):
        """
        Makes a tree, mirroring the genetic and image trees, but with a combiner node instead of a tree node.
        """
        def construct_multi_node_tree_recursive(genetic_node, image_node):
            # Check if genetic_node is an instance of LeafNode
            if not genetic_node.parent:
                return genetic_node
            
            if len(genetic_node.child_nodes) != len(image_node.child_nodes):
                raise ValueError("Genetic and Image nodes must have the same number of children")
            if len(genetic_node.child_nodes) == 0:
                return CombinerTreeNode(genetic_node, image_node)
            else:
                node = CombinerTreeNode(genetic_node, image_node)
                node.child_nodes = nn.ModuleList([construct_multi_node_tree_recursive(genetic_child, image_child) for genetic_child, image_child in zip(genetic_node.child_nodes, image_node.child_nodes)])

                return node

        return construct_multi_node_tree_recursive(self.genetic_hierarchical_ppnet.root, self.image_hierarchical_ppnet.root)

    def cuda(self, device = None):
        self.genetic_hierarchical_ppnet.cuda(device)
        self.image_hierarchical_ppnet.cuda(device)
        return super().cuda()

    def forward(self, x, get_middle_logits=False):
        genetic_conv_features = self.genetic_hierarchical_ppnet.conv_features(x[0])
        image_conv_features = self.image_hierarchical_ppnet.conv_features(x[1])

        return self.root((genetic_conv_features, image_conv_features), get_middle_logits=get_middle_logits)
    
    def get_last_layer_parameters(self):
        return nn.ParameterList([*self.genetic_hierarchical_ppnet.get_last_layer_parameters(), *self.image_hierarchical_ppnet.get_last_layer_parameters()])
    
    def get_prototype_parameters(self):
        return nn.ParameterList(
            [*self.genetic_hierarchical_ppnet.get_prototype_parameters(), *self.image_hierarchical_ppnet.get_prototype_parameters()]
        )
    
    def get_last_layer_multi_parameters(self):
        return nn.ParameterList([child.multi_last_layer.weight for child in self.root.child_nodes])

    def conv_features(self, x):
        return (self.genetic_hierarchical_ppnet.conv_features(x[0]), self.image_hierarchical_ppnet.conv_features(x[1]))

def construct_genetic_tree_ppnet(cfg: CfgNode) -> Hierarchical_PPNet: 
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

        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=720,
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=cfg.DATASET.GENETIC.PROTOTYPE_SHAPE[1]
        )

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

def construct_image_tree_ppnet(cfg: CfgNode) -> Hierarchical_PPNet: 
    class_specification = json.load(open(cfg.DATASET.TREE_SPECIFICATION_FILE, "r")) 

    if cfg.DATASET.IMAGE.PPNET_PATH != "NA":
        image_ppnet = torch.load(cfg.DATASET.IMAGE.PPNET_PATH)
        if image_ppnet.mode == 3 or image_ppnet.mode == Mode.MULTIMODAL:
            image_ppnet = image_ppnet.image_hierarchical_ppnet
    else:
        # Image Mode
        features = base_architecture_to_features["resnetbioscan"](pretrained=True)
        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=cfg.DATASET.IMAGE.SIZE,
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=cfg.DATASET.IMAGE.PROTOTYPE_SHAPE[1]
        )

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

def construct_tree_ppnet(cfg: CfgNode, log=print) -> Union[Hierarchical_PPNet, Multi_Hierarchical_PPNet]:
    # This is a gross fix to handle the renaming of the model file
    sys.modules['model.hierarchical_ppnet'] = sys.modules['model.model']

    mode = Mode(cfg.DATASET.MODE) 
    match mode: 
        case Mode.GENETIC: 
            return construct_genetic_tree_ppnet(cfg)
        case Mode.IMAGE: 
            return construct_image_tree_ppnet(cfg)
        case Mode.MULTIMODAL: 
            # if cfg.MODEL.MULTI.MULTI_PPNET_PATH != "NA":
            #     if cfg.DATASET.GENETIC.PPNET_PATH != "NA" or cfg.DATASET.IMAGE.PPNET_PATH != "NA":
            #         log("Warning: Loading MultiModalNetwork, other pretrained networks are ignored...")
            #     # Load from file, not from state_dict
            #     multi = torch.load(cfg.MODEL.MULTI.MULTI_PPNET_PATH)
            #     return multi
            multi = Multi_Hierarchical_PPNet(
                construct_genetic_tree_ppnet(cfg), 
                construct_image_tree_ppnet(cfg) 
            )
            return multi
