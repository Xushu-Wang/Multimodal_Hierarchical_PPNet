import torch
from torch import Tensor
from yacs.config import CfgNode
import torch.nn as nn
import torch.nn.functional as F
from model.features.genetic_features import GeneticCNN2D
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.backbones import base_architecture_to_features 
from dataio.dataset import Mode, TaxNode, Hierarchy

class ProtoNode(nn.Module): 
    """
    Node class that contains the prototypes for each TaxNode in Hierarchy. 
    Equivalent to the prototype layer in ProtoPNet. It is a function that predicts 
    the classification of a label in each children.  
    Attributes: 
      - childs  - pointers to the children ProtoNode 
      - taxnode - pointer to the corresponding TaxNode
      - nprotos - number of prototypes per class
      - pshape  - pshape, e.g. [64, 1, 1] for genetics or [2048, 1, 1] for image
    """
    def __init__(
        self, 
        taxnode: TaxNode, 
        nprotos: int, 
        pshape: tuple, 
        mode: Mode
    ): 
        """
        Every ProtoNode has a pointer to a TaxNode and its children 
        "Copies" structure of the TaxNode tree with additional prototypes
        """
        super().__init__()
        self.taxnode = taxnode 
        self.childs = dict() 
        self.prototype = None
        self.mode = mode 

        if taxnode.childs: 
            # only protonodes corresponding to genus need prototypes to classify species
            self.nclass = len(self.taxnode.childs)
            self.nprotos = nprotos # prototypes PER CLASS
            self.pshape = pshape 
            self.nprotos_total = self.nprotos * self.nclass
            self.match = self.init_match()

            self.prototype = nn.Parameter(
                torch.rand((self.nclass * self.nprotos, *pshape)),
                requires_grad=True
            )

            self.last_layer = self.init_last_layer()

            if self.mode == Mode.GENETIC: 
                  self.offset_tensor = self.init_offset_tensor()

    def init_match(self): 
        """
        A 0/1 block of maps between prototype and its corresponding class label
        """
        match = torch.zeros(self.nprotos_total, self.nclass)
        for j in range(self.nprotos_total):
            match [j, j // self.nprotos] = 1 
        return match

    def init_last_layer(self, inc_str = -0.5): 
        """
        initializes a matrix mapping activations to next classes in the hierarchy
        inc_str - incorrect strength initialized usually to -0.5 in paper
        """
        last_layer = nn.Linear(self.nprotos_total, self.nclass, bias=False) 
        param = self.match.clone().T
        param[param == 0] = inc_str
        last_layer.weight.data.copy_(param)
        return last_layer  

    
    def init_offset_tensor(self): 
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """
        arange1 = torch.arange(self.nprotos).view((self.nprotos, 1)).repeat((1, self.nprotos))
        indices = torch.LongTensor(torch.arange(self.nprotos))
        arange2 = (arange1 - indices) % self.nprotos
        arange3 = torch.arange(self.nprotos, 40)
        arange3 = arange3.view((1, 40 - self.nprotos))
        arange3 = arange3.repeat((self.nprotos, 1))
        
        arange4 = torch.concatenate((arange2, arange3), dim=1)
        arange4 = arange4.view(40, 1, 1, 40)
        arange4 = arange4.repeat((self.nclass, 64, 1, 1))
        return arange4
    
    def get_prototype_parameters(self):
        """
        Puts all prototype tensors of this ProtoNode and all child ProtoNodes into a list
        """ 
        if self.prototype is None: 
            # if it's a leaf node there are no params
            return []

        params = [self.prototype] 
        for _, child in self.childs.items(): 
            for param in child.get_prototype_parameters(): 
                params.append(param)
        return params

    def get_last_layer_parameters(self):
        """
        Puts all linear last-layer weights of this ProtoNode and all child ProtoNodes into a list
        """
        if self.prototype is None: 
            # if it's a leaf node there are no params
            return []

        params = [p for p in self.last_layer.parameters()]
        for _, child in self.childs.items(): 
            for param in child.get_last_layer_parameters(): 
                params.append(param)
        return params


    def find_offsetting_tensor_for_similarity(self, similarities):
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """
        eye = torch.eye(similarities.shape[2])
        eye = 1 - eye
        eye = eye.unsqueeze(0).repeat((similarities.shape[0], 1,1))
        eye = eye.to(torch.int64)

        return eye.to(similarities.device)


    def cos_sim(self, x, with_width_dim = False): 
        """
        x - convolutional output features: img=(80, 2048, 8, 8) gen=(80, 64, 1, 40) 
        """
        sqrt_D = (self.pshape[1] * self.pshape[2]) ** 0.5 
        x = F.normalize(x, dim=1) / sqrt_D 
        normalized_prototypes = F.normalize(self.prototypes, dim=1) / sqrt_D 

        if self.mode == Mode.GENETIC: 
            normalized_prototypes = F.pad(normalized_prototypes, (0, x.shape[3] - normalized_prototypes.shape[3], 0, 0))
            normalized_prototypes = torch.gather(normalized_prototypes, 3, self.offset_tensor.to(x.device))
            
            if with_width_dim:
                sim = F.conv2d(x, normalized_prototypes)
                
                # Take similarities from [80, 1600, 1, 1] to [80, 40, 40, 1]
                sim = sim.reshape((sim.shape[0], self.nclass, sim.shape[1] // self.nclass, 1))
                # Take simto [3200, 40, 1]
                sim = sim.reshape((sim.shape[0] * sim.shape[1], sim.shape[2], sim.shape[3]))
                # Take simto [3200, 40, 40]
                sim = F.pad(sim, (0, x.shape[3] - sim.shape[2], 0, 0), value=-1)
                similarity_offsetting_tensor = self.find_offsetting_tensor_for_similarity(sim)

                sim = torch.gather(sim, 2, similarity_offsetting_tensor)
                # Take similarities to [80, 40, 40, 40]
                sim = sim.reshape((sim.shape[0] // self.nclass, self.nclass, sim.shape[1], sim.shape[2]))

                # Take similarities to [80, 1600, 40]
                sim = sim.reshape((sim.shape[0], sim.shape[1] * sim.shape[2], sim.shape[3]))
                sim = sim.unsqueeze(2)

                return sim 

        return F.conv2d(x, normalized_prototypes)

    def get_logits(self, conv_features):
        sim = self.cosine_similarity(conv_features)
        max_sim = F.max_pool2d(
            sim,
            kernel_size = (sim.size()[2], sim.size()[3])
        )
        # Set pruned prototype similarities to 0.
        max_sim = self.prototype_mask * max_sim
        min_distances = -1 * max_sim

        # for each prototype, finds the spatial location that's closest to the prototype.
        min_distances = min_distances.view(-1, self.nclass * self.nprotos)

        # convert distance to similarity
        prototype_activations = -min_distances
        logits = self.last_layer(prototype_activations)

        return logits, min_distances

    def push_get_dist(self, conv_features):
        similarities = self.cos_sim(conv_features)
        distances = -1 * similarities

        return distances

    def forward(self, conv_features):
        logits, min_distances = self.get_logits(conv_features)
        return logits, min_distances
    
    def recursive_forward(self,conv_features):
        return (*self.forward(conv_features), [
            child(conv_features) for child in self.childs
        ])

    def cuda(self, device = None):
        for _, child_node in self.childs.items():
            child_node.cuda(device)
        return super().cuda(device)

    def to(self, *args, **kwargs):
        for _, child_node in self.childs.items():
            child_node.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def push_forward(self, conv_features):
        """
        This one is not recursive, because I realized doing it recursive was a bad idea.
        """
        return conv_features, self.push_get_dist(conv_features)

class HierProtoPNet(nn.Module): 
    def __init__(
        self, 
        hierarchy: Hierarchy, 
        features: nn.Module, 
        nprotos: int,
        pshape: tuple, 
        mode: Mode
    ): 
        """
        Base ProtoPnet architecture for either/or Image or Genetics. 
        Attributes: 
            features - CNN backbone 
            prototype_shape - [64, 1, 1] for genetics or [2048, 1, 1] for image
            mode - Mode 
            pshape - 3-tensor of prototype shapes 
            nproto - num prototypes per class
        """
        super().__init__()
        if mode == Mode.MULTIMODAL:
            raise ValueError("Use MultiHierarchical_PPNet for joint mode")
        if len(pshape) != 3: 
            raise ValueError("Prototype shape must be D x H x W. Leave out number of prototypes.")
        self.hierarchy = hierarchy 
        self.features = features 
        self.nprotos = nprotos
        self.pshape = pshape
        self.mode = mode
        self.root = self.build_proto_tree(self.hierarchy.root) 
        self.add_on_layers = nn.Sequential() 

    def build_proto_tree(self, taxnode: TaxNode) -> ProtoNode: 
        """
        Construct the hierarchy tree with identical structure as Hierarchy. 
        But this new tree has prototypes and linear layers for class prediction.
        """ 
        node = ProtoNode(taxnode, self.nprotos, self.pshape, self.mode) 
        node.childs = dict() 
        for k, v in taxnode.childs.items(): 
            node.childs[k] = self.build_proto_tree(v)

        return node

    def cuda(self, device = None):
        self.root.cuda(device) 
        return super().cuda(device)

    def to(self, *args, **kwargs):
        self.root.to(*args, **kwargs) 
        self.features = self.features.to(*args, **kwargs) 
        return super().to(*args, **kwargs) 

    def get_prototype_parameters(self): 
        return self.root.get_prototype_parameters() 

    def get_last_layer_parameters(self): 
        return self.root.get_last_layer_parameters()

    def conv_features(self, x): 
        x = self.features(x) 
        x = self.add_on_layers(x) 
        return x

    def forward(self, x): 
        conv_features = self.conv_features(x) 
        logit_tree = self.root.recursive_forward(conv_features) 
        return logit_tree

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

def construct_genetic_ppnet(cfg: CfgNode) -> HierProtoPNet: 
    hierarchy = Hierarchy(cfg.DATASET.TREE_SPECIFICATION_FILE)

    if cfg.DATASET.GENETIC.PPNET_PATH != "NA":
        # retrieve cached_ppnet 
        genetic_ppnet = torch.load(cfg.DATASET.GENETIC.PPNET_PATH)
    else:
        backbone = GeneticCNN2D(720, 1, include_connected_layer=False)

        # Remove the fully connected layer
        weights = torch.load(cfg.MODEL.GENETIC_BACKBONE_PATH)
        for k in list(weights.keys()):
            if "conv" not in k:
                del weights[k]
        backbone.load_state_dict(weights)

        # NOTE - Layer_paddings is different from the padding in the image models
        layer_filter_sizes, layer_strides, layer_paddings = backbone.conv_info()

        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=720,
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=cfg.DATASET.GENETIC.PROTOTYPE_SHAPE[1]
        )

        genetic_ppnet = HierProtoPNet(
            hierarchy = hierarchy, 
            features = backbone, 
            nprotos = cfg.DATASET.GENETIC.NUM_PROTOTYPES_PER_CLASS, 
            pshape = cfg.DATASET.GENETIC.PROTOTYPE_SHAPE, 
            mode = Mode.GENETIC
        ) 

    return genetic_ppnet

def construct_image_ppnet(cfg: CfgNode) -> HierProtoPNet: 
    hierarchy = Hierarchy(cfg.DATASET.TREE_SPECIFICATION_FILE)

    if cfg.DATASET.IMAGE.PPNET_PATH != "NA":
        # retrieve cached_ppnet 
        image_ppnet = torch.load(cfg.DATASET.IMAGE.PPNET_PATH)
        # if image_ppnet.mode == 3 or image_ppnet.mode == Mode.MULTIMODAL:
        #     image_ppnet = image_ppnet.img_net
    else:
        backbone = base_architecture_to_features["resnetbioscan"](pretrained=True)
        layer_filter_sizes, layer_strides, layer_paddings = backbone.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=cfg.DATASET.IMAGE.SIZE,
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=cfg.DATASET.IMAGE.PROTOTYPE_SHAPE[1]
        )

        image_ppnet = HierProtoPNet(
            hierarchy = hierarchy, 
            features = backbone, 
            nprotos = cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS, 
            pshape = cfg.DATASET.IMAGE.PROTOTYPE_SHAPE,
            mode=Mode.IMAGE
        )

    return image_ppnet

class CombinerProtoNode(nn.Module):
    """
    A wrapper node containing a genetic and image ProtoNode for the same TaxNode
    """
    def __init__(self, gen_node: ProtoNode, img_node: ProtoNode): 
        super().__init__()
        self.gen_node = gen_node
        self.img_node = img_node
        # self.int_location = gen_node.int_location
        # self.named_location = gen_node.named_location
        self.mode = Mode.MULTIMODAL
        self.max_tracker = ([], [])
        self.correlation_count = 0
        self.correlation_table = torch.zeros((40,10)).cuda()

        self.childs = nn.ModuleList() # Note this will be initialized by Multi_Hierarchical_PPNet

        self.match = self.init_match()
        
        # Create the correspondence map 
        if self.gen_node.nprotos % self.img_node.nprotos == 0: 
            self.prototype_ratio = self.gen_node.nprotos // self.img_node.nprotos
        else: 
            raise ValueError("Number of genetic prototypes must be an integral multiple of image prototypes.")

        self.multi_last_layer = self.init_multi_last_layer()

    def init_match(self): 
        """
        A 0/1 block of maps between img/gen class and classes
        """
        match = torch.zeros(2 * self.gen_node.nclass, self.gen_node.nclass)
        for i in range(2 * self.gen_node.nclass):
            match[i, i % self.gen_node.nclass] = 1 
        return match

    def init_multi_last_layer(self, inc_str = -0.5):
        """
        initializes a matrix mapping activations to next classes in the hierarchy
        inc_str - incorrect strength initialized usually to -0.5 in paper
        """
        last_layer = nn.Linear(2 * self.gen_node.nclass, self.gen_node.nclass, bias=False) 
        param = self.match.clone().T
        param[param == 0] = inc_str
        last_layer.weight.data.copy_(param)
        return last_layer  

    def get_logits(self, gen_conv_features, img_conv_features, intermediate=False):
        genetic_logit, genetic_dist = self.gen_node.get_logits(gen_conv_features)
        image_logit, image_dist = self.img_node.get_logits(img_conv_features)

        if intermediate:
            return (genetic_logit, image_logit), (genetic_dist, image_dist)

        logits = self.multi_last_layer(torch.cat((genetic_logit, image_logit), dim=1))
        return logits, (genetic_dist, image_dist)
    
    def forward(self, x, intermediate=False):
        gen_conv_features, img_conv_features = x
        return self.get_logits(gen_conv_features, img_conv_features, intermediate=intermediate)

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

        # self.levels = self.gen_net.levels
        self.mode = Mode.MULTIMODAL
        self.root = self.build_combiner_proto_tree()
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

    def build_combiner_proto_tree(self):
        """
        Makes a tree, mirroring the genetic and image trees, but with a combiner node instead of a tree node.
        """
        def build_combiner_proto_tree_rec(gen_node: ProtoNode, img_node: ProtoNode):
            # Check if genetic_node is an instance of LeafNode 
            if gen_node.prototype is None: 
                return gen_node
            
            if len(gen_node.childs) != len(img_node.childs):
                raise ValueError("Genetic and Image nodes must have the same number of children")
            if len(gen_node.childs) == 0:
                return CombinerProtoNode(gen_node, img_node)
            else:
                node = CombinerProtoNode(gen_node, img_node)
                childs = [] 
                for name in gen_node.childs: 
                    gen_child_node = gen_node.childs[name] 
                    img_child_node = img_node.childs[name] 
                    childs.append(build_combiner_proto_tree_rec(gen_child_node, img_child_node))
                node.childs = nn.ModuleList(childs)

                return node

        return build_combiner_proto_tree_rec(self.gen_net.root, self.img_net.root)

    def cuda(self, device = None):
        self.gen_net.cuda(device)
        self.img_net.cuda(device)
        return super().cuda()

    def conv_features(self, x):
        return (self.gen_net.conv_features(x[0]), self.img_net.conv_features(x[1]))

    def forward(self, x, intermediate=False):
        genetic_conv_features, image_conv_features = self.conv_features(x) 
        return self.root((genetic_conv_features, image_conv_features), intermediate=intermediate)
    
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
    
    def get_last_layer_multi_parameters(self):
        return nn.ParameterList([child.multi_last_layer.weight for child in self.root.childs])

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
