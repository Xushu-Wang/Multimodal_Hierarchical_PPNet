import torch
import numpy as np
from yacs.config import CfgNode
import torch.nn as nn
import torch.nn.functional as F
from model.features.genetic_features import GeneticCNN2D
from prototype.receptive_field import compute_proto_layer_rf_info_v2
from model.backbones import base_architecture_to_features 
from dataio.dataset import Mode, TaxNode, Hierarchy

# Decorator that returns immediately if self.vestigial is True
def classifier_only(func):
    def wrapper(self, *args, **kwargs):
        if self.vestigial:
            return
        return func(self, *args, **kwargs)
    return wrapper

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
        self.taxonomy = taxnode.taxonomy
        self.idx = taxnode.idx 
        self.flat_idx = taxnode.flat_idx 
        self.depth = taxnode.depth
        self.childs = []
        self.prototype = None
        self.mode = mode 

        self.vestigial = not taxnode.childs or len(taxnode.childs) == 1
        # self.vestigial = False

        if taxnode.childs: 
            # only protonodes corresponding to genus need prototypes to classify species
            self.nclass = len(self.taxnode.childs)
            self.min_species_idx = self.taxnode.min_species_idx
            self.max_species_idx = self.taxnode.max_species_idx
            self.nprotos = nprotos # prototypes PER CLASS
            self.pshape = pshape 
            self.nprotos_total = self.nprotos * self.nclass

            if self.vestigial:
                # Vestigial nodes only have one child and thus don't have to perform any classification
                # outputs of forward pass will be stored here 
                self.logits = torch.tensor([0]).cuda()
                self.probs = None
                self.max_sim = None
            else:
                self.match = self.init_match().cuda()

                self.prototype = nn.Parameter(
                    torch.rand((self.nprotos_total, *pshape), device="cuda"),
                    requires_grad=True
                )

                self.last_layer = self.init_last_layer()

                # outputs of forward pass will be stored here 
                self.logits = None
                self.probs = None
                self.max_sim = None

                if self.mode == Mode.GENETIC: 
                    self.register_buffer('offset_tensor', self.init_offset_tensor())

                self.npredictions = 0 
                self.n_next_correct = 0 
                self.n_species_correct = 0

                self.init_push()

    @classifier_only
    def init_match(self): 
        """
        A 0/1 block of maps between prototype and its corresponding class label
        """
        match = torch.zeros(self.nprotos_total, self.nclass)
        for j in range(self.nprotos_total):
            match[j, j // self.nprotos] = 1 
        return match

    @classifier_only
    def init_push(self):
        # all attributes regarding to push stage 
        # saves the closest distance seen so far 
        # add a new attribute and initialize it to be infinity
        self.global_max_proto_sim = np.full(self.nprotos_total, np.inf)

        # saves the patch representation that gives the current smallest distance
        self.global_max_fmap_patches = np.zeros([self.nprotos_total, *self.pshape])

        # 

        # We assume save_prototype_class_identity is true
        '''
        proto_rf_boxes and proto_bound_boxes column:
        0: image index in the entire dataset
        1: height start index
        2: height end index
        3: width start index
        4: width end index
        5: (optional) class identity
        '''
        self.proto_rf_boxes = np.full([self.nprotos, 6], fill_value=-1)
        self.proto_bound_boxes = np.full([self.nprotos, 6], fill_value=-1)

    @classifier_only
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
    
    @classifier_only
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
        arange4 = arange4.reshape(40, 1, 1, 40)
        arange4 = arange4.repeat((self.nclass, 64, 1, 1))
        return arange4
    
    def cuda(self, device = None):
        for child in self.childs:
            child.cuda(device)
        return super().cuda(device)

    def to(self, *args, **kwargs):
        for child in self.childs:
            child.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_prototype_parameters(self):
        """
        Puts all prototype tensors of this ProtoNode and all child ProtoNodes into a list
        """ 
        if self.prototype is None: 
            # if it's a leaf node there are no params
            return []

        params = [] if self.vestigial else [self.prototype]
        for child in self.childs:
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

        params = [] if self.vestigial else [p for p in self.last_layer.parameters()]
        for child in self.childs:
            for param in child.get_last_layer_parameters(): 
                params.append(param)
        return params

    @classifier_only
    def find_offsetting_tensor_for_similarity(self, similarities):
        """
        This finds the tensor used to offset each prototype to a different spatial location.
        """
        eye = torch.eye(similarities.shape[2])
        eye = 1 - eye
        eye = eye.unsqueeze(0).repeat((similarities.shape[0], 1,1))
        eye = eye.to(torch.int64)

        return eye.to(similarities.device)

    @classifier_only
    def cos_sim(self, x): 
        """
        x - convolutional output features: img=(80, 2048, 8, 8) gen=(80, 64, 1, 40)  
        prototype - full prototypes in node: img=(10, 2048, 1, 1), gen=(40, 64, 1, 40)
        """
        # keep sqrt_D here 
        sqrt_D = (self.pshape[1] * self.pshape[2]) ** 0.5 
        x = F.normalize(x, dim=1) / sqrt_D
        prototype = self.prototype.to(x.device) 
        normalized_prototypes = F.normalize(prototype, dim=1) / sqrt_D # type:ignore

        if self.mode == Mode.GENETIC: 
            normalized_prototypes = F.pad(normalized_prototypes, (0, x.shape[3] - normalized_prototypes.shape[3], 0, 0))
            normalized_prototypes = torch.gather(normalized_prototypes, 3, self.offset_tensor)
            
        # IMG: (80, 2048, 8, 8) * (10 * nclass, 2048, 1, 1) -> (80, 10 * nclass, 8, 8) 
        # GEN: (80, 64, 1, 40)  * (40 * nclass, 64, 1, 40)  -> (80, 40 * nclass, 1, 1)
        similarities = F.conv2d(x, normalized_prototypes)  
        return similarities

    @classifier_only
    def forward(self, conv_features):
        """
        Forward pass on this node. Used for training when conv_features 
        are masked. 
        """
        # IMG: (80, 10 * nclass, 8, 8), GEN: (80, 40 * nclass, 1, 1), values in [-1, 1]
        sim = self.cos_sim(conv_features) 
        # IMG: (80, 10 * nclass, 1, 1), GEN: (80, 40 * nclass, 1, 1), in [-1, 1]
        max_sim = F.max_pool2d(sim, kernel_size = (sim.size(2), sim.size(3)))   

        # for each prototype, finds the spatial location that's closest to the prototype. 
        # IMG: (80, 10 * nclass), GEN: (80, 40 * nclass)
        max_sim = max_sim.view(-1, self.nclass * self.nprotos)  

        # convert distance to similarity
        logits = self.last_layer(max_sim)
        self.logits = logits
        self.max_sim = max_sim

        return logits, max_sim

    @classifier_only
    def softmax(self): 
        if self.logits is None: 
            raise ValueError("You must do a forward pass so that logits are set.") 
        self.probs = F.softmax(self.logits, dim=1)

    @classifier_only
    def clear_cache(self): 
        if self.logits != None:
            del self.logits
            self.logits = None
        if self.probs != None:
            del self.probs
            self.probs = None
        if self.max_sim != None:
            del self.max_sim
            self.max_sim = None


class HierProtoPNet(nn.Module): 
    def __init__(
        self, 
        hierarchy: Hierarchy, 
        features: nn.Module, 
        nprotos: int,
        pshape: tuple, 
        mode: Mode,
        proto_layer_rf_info=None
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
        self.classifier_nodes = []
        self.all_classifier_nodes = []
        self.root = self.build_proto_tree(self.hierarchy.root) 
        self.add_on_layers = nn.Sequential() 
        self.proto_layer_rf_info = proto_layer_rf_info

    def build_proto_tree(self, taxnode: TaxNode) -> ProtoNode: 
        """
        Construct the hierarchy tree with identical structure as Hierarchy. 
        But this new tree has prototypes and linear layers for class prediction. 
        We also add references to all non-leaf nodes into list
        """ 
        node = ProtoNode(taxnode, self.nprotos, self.pshape, self.mode) 
        if taxnode.childs: 
            self.all_classifier_nodes.append(node)

            if not node.vestigial: 
                self.classifier_nodes.append(node)

        node.childs = []
        for c_node in taxnode.childs:
            node.childs.append(self.build_proto_tree(c_node))

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

    def conditional_normalize(self, node: ProtoNode, scale = torch.ones(1).cuda()): 
        """
        Once node.probs is instantiated, scale the children's probabilities 
        by the conditional probability of the parent's prediction
        """ 
        # base case when nod ehas no childs
        if not hasattr(node, "probs"): 
            # leaf node
            return 
        elif node.probs is None and not node.vestigial: 
            raise ValueError("The probabilities for the nodes should be instantiated.")
        else: 
            # node.probs = (B, nclass) 
            # scale = (B) -> unsqueeze it to (B, 1) to broadcast properly
            if node.vestigial:
                node.probs = torch.ones_like(scale.unsqueeze(1))

            node.probs *= scale.unsqueeze(1)
            for child in node.childs:
                child_idx = child.idx[-1]
                self.conditional_normalize(child, node.probs[:,child_idx]) 

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_pred(self): 
        """
        Wipe out the logits, probs, max_sim, and prediction statistics. 
        Should be called at the end of every epoch. 
        """
        self.logits = None 
        self.probs = None 
        self.max_sim = None 
        self.npredictions = 0 
        self.n_next_correct = 0 
        self.n_species_correct = 0
        torch.cuda.empty_cache()

def construct_genetic_ppnet(cfg: CfgNode) -> HierProtoPNet: 
    hierarchy = Hierarchy(cfg.DATASET.TREE_SPECIFICATION_FILE)

    if cfg.DATASET.GENETIC.PPNET_PATH != "NA":
        # retrieve cached_ppnet 
        genetic_ppnet = torch.load(cfg.DATASET.GENETIC.PPNET_PATH) 
    else:
        backbone = GeneticCNN2D(720, 1, include_connected_layer=False)

        # Remove the fully connected layer
        weights = torch.load(cfg.MODEL.GENETIC_BACKBONE_PATH, weights_only=True)
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
            mode = Mode.GENETIC,
            proto_layer_rf_info=proto_layer_rf_info
        ) 

    return genetic_ppnet

def construct_image_ppnet(cfg: CfgNode) -> HierProtoPNet: 
    hierarchy = Hierarchy(cfg.DATASET.TREE_SPECIFICATION_FILE)

    if cfg.DATASET.IMAGE.PPNET_PATH != "NA":
        image_ppnet = torch.load(cfg.DATASET.IMAGE.PPNET_PATH)
    else:
        # should not remove the final ReLU layer before the avgpool 
        backbone = base_architecture_to_features["resnetbioscan"](pretrained=True) 
        layer_filter_sizes, layer_strides, layer_paddings = backbone.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=cfg.DATASET.IMAGE.SIZE,
            layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides,
            layer_paddings=layer_paddings,
            prototype_kernel_size=cfg.DATASET.IMAGE.PROTOTYPE_SHAPE[1],
        )

        image_ppnet = HierProtoPNet(
            hierarchy = hierarchy, 
            features = backbone, 
            nprotos = cfg.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS, 
            pshape = cfg.DATASET.IMAGE.PROTOTYPE_SHAPE,
            mode = Mode.IMAGE,
            proto_layer_rf_info=proto_layer_rf_info
        )

    return image_ppnet

