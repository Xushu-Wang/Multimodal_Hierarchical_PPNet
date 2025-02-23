import torch
from yacs.config import CfgNode
import torch.nn as nn
from dataio.dataset import Mode 
from model.hierarchical import *

class CombinerProtoNode(nn.Module):
    """
    A wrapper node containing a genetic and image ProtoNode for the same TaxNode
    """
    def __init__(self, gen_node: ProtoNode, img_node: ProtoNode): 
        super().__init__()

        # it is already checked that both nodes have same indexing
        self.gen_node = gen_node
        self.img_node = img_node  
        self.taxonomy = gen_node.taxonomy 
        self.idx = gen_node.idx 
        self.flat_idx = gen_node.flat_idx 
        self.depth = gen_node.depth
        self.childs = nn.ModuleList() # Note this will be initialized by Multi_Hierarchical_PPNet
        self.mode = Mode.MULTIMODAL 
        self.correlation_count = 0
        self.correlation_table = torch.zeros((40,10)).cuda()

        if self.gen_node.childs: 
            self.nclass = self.gen_node.nclass
            self.match = self.init_match()
            
            # Create the correspondence map 
            if self.gen_node.nprotos % self.img_node.nprotos == 0: 
                self.prototype_ratio = self.gen_node.nprotos // self.img_node.nprotos
            else: 
                raise ValueError("Number of genetic prototypes must be an integral multiple of image prototypes.")

    def init_match(self): 
        """
        A 0/1 block of maps between img/gen class and classes
        """
        match = torch.zeros(2 * self.gen_node.nclass, self.gen_node.nclass)
        for i in range(2 * self.gen_node.nclass):
            match[i, i % self.gen_node.nclass] = 1 
        return match

    def get_logits(self, gen_conv_features, img_conv_features): 
        genetic_logit, genetic_dist = self.gen_node.get_logits(gen_conv_features)
        image_logit, image_dist = self.img_node.get_logits(img_conv_features)

        return (genetic_logit, image_logit), (genetic_dist, image_dist)
    
    def forward(self, gen_conv_features, img_conv_features): 
        return self.get_logits(gen_conv_features, img_conv_features) 

class MultiHierProtoPNet(nn.Module):
    """
    A wrapper class around a Genetic and Image Hierarchical ProtoPNet 
    Adds another tree of CombinerProtoNodes 
    """
    def __init__(self, gen_net: HierProtoPNet, img_net: HierProtoPNet):
        super().__init__()
        if (gen_net.mode != Mode.GENETIC) or (img_net.mode != Mode.IMAGE): 
            raise ValueError(f"Incorrect Modes: GenNet [{gen_net.mode.value}], ImgNet [{img_net.mode.value}]")
        if gen_net.hierarchy != img_net.hierarchy:
            raise ValueError("Hierarchies between gen and img nets must be the same.")

        self.hierarchy = gen_net.hierarchy
        self.features = nn.ModuleList([gen_net.features, img_net.features])
        self.gen_net = gen_net
        self.img_net = img_net

        self.mode = Mode.MULTIMODAL
        self.classifier_nodes = [] 
        self.root = self.build_combiner_proto_tree(self.gen_net.root, self.img_net.root)
        self.add_on_layers = nn.ModuleList([self.gen_net.add_on_layers, self.img_net.add_on_layers])
        self.nodes_with_children = self.get_nodes_with_children()

    def get_nodes_with_children(self):
        nodes_with_children = nn.ModuleList()
        def get_nodes_with_children_recursive(node):
            nodes_with_children.append(node)
            for child in node.childs:
                get_nodes_with_children_recursive(child)
        get_nodes_with_children_recursive(self.root)
        return nodes_with_children

    def build_combiner_proto_tree(self, gen_node: ProtoNode, img_node: ProtoNode):
        """
        Makes a tree, mirroring the genetic and image trees, but with a combiner node instead of a tree node.
        """
        # if gen_node.prototype is None: 
        #     return gen_node
        
        if len(gen_node.childs) != len(img_node.childs):
            raise ValueError("Genetic and Image nodes must have the same number of children")
        if len(gen_node.childs) == 0:
            node = CombinerProtoNode(gen_node, img_node)
        else:
            node = CombinerProtoNode(gen_node, img_node)
            self.classifier_nodes.append(node)
            childs = [] 
            for name in gen_node.childs: 
                gen_child_node = gen_node.childs[name] 
                img_child_node = img_node.childs[name] 
                childs.append(self.build_combiner_proto_tree(gen_child_node, img_child_node))
            node.childs = nn.ModuleList(childs)

        return node

    def cuda(self, device = None):
        self.gen_net.cuda(device)
        self.img_net.cuda(device)
        return super().cuda()

    def conv_features(self, genetic, image):
        return self.gen_net.conv_features(genetic), self.img_net.conv_features(image)

    def forward(self, genetic, image):
        genetic_conv_features, image_conv_features = self.conv_features(genetic, image) 
        return self.root(genetic_conv_features, image_conv_features)
    
    def get_last_layer_parameters(self):
        return nn.ParameterList([
            *self.gen_net.get_last_layer_parameters(), 
            *self.img_net.get_last_layer_parameters()
        ])
    
    def get_prototype_parameters(self):
        return nn.ParameterList([
            *self.gen_net.get_prototype_parameters(), 
            *self.img_net.get_prototype_parameters()
        ])
    
def construct_ppnet(cfg: CfgNode):
    mode = Mode(cfg.DATASET.MODE) 
    match mode: 
        case Mode.GENETIC: 
            return construct_genetic_ppnet(cfg)
        case Mode.IMAGE: 
            return construct_image_ppnet(cfg)
        case Mode.MULTIMODAL: 
            multi = MultiHierProtoPNet(
                construct_genetic_ppnet(cfg), 
                construct_image_ppnet(cfg) 
            )
            return multi
