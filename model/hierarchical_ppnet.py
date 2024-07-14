import torch
import torch.nn as nn
import torch.nn.functional as F


class Hierarchical_PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 root,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_distance_function = 'cosine',
                 prototype_activation_function='log',
                 genetics_mode=False,
                 fix_prototypes=False,
                 rearrange_logit_map=None
        ):
        """
        Rearrange logit map maps the genetic class index to the image class index, which will be considered the true class index.
        """
                
        super().__init__()
        
        self.root = root
        
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.fix_prototypes = fix_prototypes

        if self.fix_prototypes:
            if self.prototype_shape[3] != 1:
                raise NotImplementedError("Fix_prototypes only supported for 1x1 prototypes")
            
        self.prototype_distance_function = prototype_distance_function
        self.prototype_activation_function = prototype_activation_function # 'log' or 'linear'

        # Ensure that we're using linear with cosine similarity
        assert(not (prototype_distance_function == 'cosine' and prototype_activation_function != "linear"))

        assert(self.num_prototypes % self.num_classes == 0)
        
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

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


        if self.prototype_distance_function == 'cosine':
            self.add_on_layers = nn.Sequential()
            
            self.prototype_vectors = nn.Parameter(torch.randn(self.prototype_shape),
                                requires_grad=True)
            
        elif self.prototype_distance_function == 'l2':
            proto_depth = self.prototype_shape[1]

            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=proto_depth, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=proto_depth, out_channels=proto_depth, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                requires_grad=True)
            

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

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
            similarity = self.cosine_similarity(conv_features)
            max_similarities = F.max_pool2d(similarity,
                            kernel_size=(similarity.size()[2],
                                        similarity.size()[3]))
            min_distances = -1 * max_similarities
        elif self.prototype_distance_function == 'l2':
            distances = self.l2_distance(conv_features)
            
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
    

    def cosine_similarity(self, x, with_width_dim=False):
        sqrt_dims = (self.prototype_shape[2] * self.prototype_shape[3]) ** .5
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
    
    def l2_distance(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2_patch_sum = F.conv2d(input=x**2, weight=self.ones)

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
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

