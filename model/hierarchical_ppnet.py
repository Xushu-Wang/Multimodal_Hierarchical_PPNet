import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TreeNode(nn.Module):
    def __init__(self,
                 int_location,
                 named_location,
                 num_prototypes_per_class,
                 prototype_shape,
                 tree_specification,
                 fix_prototypes
    ):
        """
        int_location: the 2nd child of the 1st class will look like (1,2) (1 indexing)
        named_location: the 2nd child of the 1st class will look like ("class_1", "class_2")
        num_prototypes_per_class: the number of prototypes per class
        prototype_shape: the shape of the prototype (minus the number of prototypes) length 3
        tree_specification: the tree specification
        fix_prototypes: whether to fix the prototypes or not
        """
        super().__init__()

        if len(prototype_shape) != 3:
            raise ValueError("Prototype_shape must be of length 3, leave out the number of prototypes man, come on.")

        self.int_location = int_location
        self.named_location = named_location
        self.num_classes = len(tree_specification)
        self.num_prototypes = self.num_classes * num_prototypes_per_class
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototype_shape = prototype_shape
        self.full_prototype_shape = (self.num_prototypes, prototype_shape[0], prototype_shape[1], prototype_shape[2]) 
        self.fix_prototypes = fix_prototypes
        self.tree_specification = tree_specification
        
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.child_nodes = []

        self.prototype_vectors = nn.Parameter(
            torch.randn(self.full_prototype_shape),
            requires_grad=True
        )
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False)
        
        self.set_last_layer_incorrect_connection(-0.5)
        self.create_children()

    def get_prototype_parameters(self):
        # Forgive me for this confusing black magic. It just puts all the parameters into a list
        return [self.prototype_vectors] + [param for child in self.child_nodes for param in child.get_prototype_parameters()]

    def get_last_layer_parameters(self):
        return [p for p in self.last_layer.parameters()] + [param for child in self.child_nodes for param in child.get_last_layer_parameters()]

    def create_children(self):
        i = 1 # 0 Is reserved for not_classified
        for name, child in self.tree_specification.items():
            if child is None:
                continue
            self.child_nodes.append(TreeNode(int_location=self.int_location + [i],
                                          named_location=self.named_location + [name],
                                          num_prototypes_per_class=self.num_prototypes_per_class,
                                          prototype_shape=self.prototype_shape,
                                          tree_specification=child,
                                          fix_prototypes=self.fix_prototypes
                                          ))
            i += 1
    
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
        min_distances = -1 * max_similarities

        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)

        return logits, min_distances
    
    def push_get_dist(self, conv_features):
        similarities = self.cosine_similarity(conv_features)
        distances = -1 * similarities

        return distances

    def forward(self, conv_features):
        logits, min_distances = self.get_logits(conv_features)
        return (logits, min_distances, [child(conv_features) for child in self.child_nodes])
        # return {
        #     "int_location": self.int_location,
        #     "named_location": self.named_location,
        #     "logits": logits,
        #     "min_distances": min_distances,
        #     "children": [child(conv_features) for child in self.child_nodes]
        # }
    
    def cuda(self):
        for child in self.child_nodes:
            child.cuda()
        return super().cuda()

    def push_forward(self, conv_features):
        return {
            "int_location": self.int_location,
            "named_location": self.named_location,
            "distances": self.push_get_dist(conv_features),
            "children": [child.push_forward(conv_features) for child in self.child_nodes]
        }

    def __repr__(self):
        return f"TreeNode: [{'>'.join(self.named_location)}]"


class Hierarchical_PPNet(nn.Module):
    def __init__(self, 
                 features,
                 img_size, 
                 prototype_shape,
                 num_prototypes_per_class,
                 class_specification,
                 proto_layer_rf_info, 
                 genetics_mode=False,
        ):
        """
        Rearrange logit map maps the genetic class index to the image class index, which will be considered the true class index.
        
        NOTE: Prototype shape is a 3-tuple, it does not include the number of prototypes. This is non-standard.
        """
        
        super().__init__()
        
        self.tree_specification = class_specification["tree"]
        self.levels = class_specification["levels"]
                
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.genetics_mode = genetics_mode

        self.num_prototypes_per_class = num_prototypes_per_class
            
        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        # Construct PPNet Tree
        self.root = self.construct_last_layer_tree()
        
        # TODO - Handle different image ppnet sizes you know what I mean
        self.add_on_layers = nn.Sequential()

        self.nodes_with_children = self.get_nodes_with_children()

    def cuda(self):
        self.root.cuda()
        return super().cuda()


    def get_nodes_with_children(self):
        nodes_with_children = []
        def get_nodes_with_children_recursive(node):
            if len(node.child_nodes) > 0:
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
            fix_prototypes=self.genetics_mode)
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
        logit_tree = self.root(conv_features)

        return logit_tree
    
    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distance_tree = self.root.push_forward(conv_output)
        
        return conv_output, distance_tree

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