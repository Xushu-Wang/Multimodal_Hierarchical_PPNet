import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Hierarchical_PPNet(nn.Module):
    def __init__(self, features, img_size, 
                 prototype_shape,
                 num_prototypes_per_class,
                 root,
                 proto_layer_rf_info, 
                 init_weights=True,
                 prototype_distance_function = 'cosine',
                 prototype_activation_function='log',
                 genetics_mode=False,
        ):
        """
        Rearrange logit map maps the genetic class index to the image class index, which will be considered the true class index.
        """
                
        super().__init__()
        
        self.root = root
        
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes_per_class = num_prototypes_per_class
        self.epsilon = 1e-4
            
        self.prototype_distance_function = prototype_distance_function
        self.prototype_activation_function = prototype_activation_function
        
        
        for name, num_children in root.class_to_num_children().items():
            setattr(self,"num_" + name, num_children)
            
        for name,shape in root.class_to_proto_shape(x_per_child=num_prototypes_per_class, dimension=prototype_shape).items():
            setattr(self,name + "_prototype_shape",shape)
            setattr(self,"num_" + name + "_prototypes",shape[0])
            setattr(self,name + "_prototype_vectors", nn.Parameter(torch.rand(shape), requires_grad=True))
            setattr(self,name + "_layer", nn.Linear(shape[0], getattr(self,"num_" + name), bias = False))
            setattr(self,"ones_" + name, nn.Parameter(torch.ones(shape), requires_grad=False))
            
            root.set_node_attr(name,"num_prototypes_per_class",num_prototypes_per_class)
            root.set_node_attr(name,"prototype_shape",shape)  
                    

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

    
        if self.prototype_distance_function == 'cosine':
            self.add_on_layers = nn.Sequential()
            
            self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                requires_grad=True)
            
        elif self.prototype_distance_function == 'l2':
            proto_depth = self.prototype_shape[1]
            
            features_name = str(self.features).upper()
            if features_name.startswith('VGG') or features_name.startswith('RES'):
                first_add_on_layer_in_channels = \
                    [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
            elif features_name.startswith('DENSE'):
                first_add_on_layer_in_channels = \
                    [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
            elif genetics_mode:
                first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
            else:
                raise Exception('other base base_architecture NOT implemented')

            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=proto_depth, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=proto_depth, out_channels=proto_depth, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                requires_grad=True)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)

        return x
    
    
    def classifier(self, conv_features, node):        
                
        if self.prototype_distance_function == 'cosine':
            similarity = self.cosine_similarity(conv_features, node.name)
            max_similarities = F.max_pool2d(similarity,
                            kernel_size=(similarity.size()[2],
                                        similarity.size()[3]))
            min_distances = -1 * max_similarities
        elif self.prototype_distance_function == 'l2':
            distances = self.l2_distance(conv_features, node.name)
            
            # global min pooling
            min_distances = -F.max_pool2d(-distances,
                                        kernel_size=(distances.size()[2],
                                                    distances.size()[3]))
            
                    
        min_distances = min_distances.view(-1, getattr(self,"num_" + node.name + "_prototypes"))
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = getattr(self, node.name + "_layer")(prototype_activations)
        
        setattr(node, "logits", logits)
        setattr(node, "min_distances", min_distances)
        
    def forward(self, x):

        conv_features = self.conv_features(x)

        for node in self.root.nodes_with_children():
            self.classifier(conv_features,node)
    

    def cosine_similarity(self, x, name, with_width_dim=False):
        sqrt_dims = (self.prototype_shape * self.prototype_shape) ** .5
        
        x_norm = F.normalize(x, dim=1) / sqrt_dims
        normalized_prototypes = F.normalize(getattr(self,"ones_" + name), dim=1) / sqrt_dims

        return F.conv2d(x_norm, normalized_prototypes)
    
    def l2_distance(self, x, name):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2_patch_sum = F.conv2d(input=x**2, weight=getattr(self,"ones_" + name))

        p2 = torch.sum(self.prototype_vectors ** 2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise NotImplementedError
        
    
    # Prototype Pruning Starts from Here

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

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        # Possibly better to go through and change push with this similarity metric
        conv_output = self.conv_features(x)
        if self.prototype_distance_function == 'cosine':
            similarities = self.cosine_similarity(conv_output)
            distances = -1 * similarities
        elif self.prototype_distance_function == 'l2':
            distances = self.l2_distance(conv_output)
        return conv_output, distances

    def push_forward_fixed(self,x):
        conv_output = self.conv_features(x)
        similarities = self.cosine_similarity(conv_output)
        distances = -1 * similarities

        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

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

    def _initialize_weights(self):
        for m in self.modules(): # Returns an iterator over all modules in the network.   
            if len(m._modules) > 0: # skip anything that's not a single layer
                continue
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)            
            elif isinstance(m, nn.Linear):
                identity = torch.eye(m.out_features)
                repeated_identity = identity.unsqueeze(2).repeat(1,1,self.num_prototypes_per_class).\
                                            view(m.out_features, -1)
                m.weight.data.copy_(1.5 * repeated_identity - 0.5)
                
    def get_joint_distribution(self):
           

        batch_size = self.root.logits.size(0)

        #top_level = torch.nn.functional.softmax(self.root.logits,1)            
        top_level = self.root.logits
        bottom_level = self.root.distribution_over_furthest_descendents(batch_size)    

        names = self.root.unwrap_names_of_joint(self.root.names_of_joint_distribution())
        idx = np.argsort(names)

        bottom_level = bottom_level[:,idx]        
        
        return top_level, bottom_level


