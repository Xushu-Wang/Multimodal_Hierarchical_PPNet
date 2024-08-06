import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import pandas as pd

from model.utils import decode_onehot
from prototype.receptive_field import compute_rf_prototype
from utils.util import makedir, find_high_activation_crop

def init_nodal_push_prototypes(node, root_dir_for_saving_prototypes, epoch_number, log):
    """
    Initializes all these variables that are normally global on each node
    """
    # saves the closest distance seen so far
    node.global_min_proto_dist = np.full(node.num_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    node.global_min_fmap_patches = np.zeros(
        [
            node.num_prototypes,
            node.prototype_shape[0],
            node.prototype_shape[1],
            node.prototype_shape[2]
        ]
    )

    # We assume save_prototype_class_identiy is true
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    node.proto_rf_boxes = np.full(
        shape=[node.num_prototypes, 6],
        fill_value=-1
    )
    node.proto_bound_boxes = np.full(
        shape=[node.num_prototypes, 6],
        fill_value=-1
    )

    if root_dir_for_saving_prototypes != None:
        # This is gross, no?
        proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                        f'epoch-{epoch_number}',*node.named_location, "prototypes")
        makedir(proto_epoch_dir)
        node.proto_epoch_dir = proto_epoch_dir
        # Clear this directory
        log(f"Clearing {proto_epoch_dir}")
        for file in os.listdir(proto_epoch_dir):
            os.remove(os.path.join(proto_epoch_dir, file))
    else:
        proto_epoch_dir = None

def nodal_update_prototypes_on_batch(
    node,
    search_batch_input,
    conv_features,
    start_index_of_search_batch,
    model,
    search_y,
    preprocess_input_function,
    prototype_layer_stride,
    prototype_img_filename_prefix,
    prototype_self_act_filename_prefix,
    prototype_activation_function_in_numpy,
    no_save
):
    model.eval()
    mode = model.mode
    dir_for_saving_prototypes = node.proto_epoch_dir

    level = node.level

    num_classes = node.num_classes

    with torch.no_grad():
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = node.push_forward(conv_features)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    # Assume class specific
    class_to_img_index_dict = {key+1: [] for key in range(num_classes)}
    # img_y is the image's integer label
    for img_index, img_y in enumerate(search_y):
        img_label = int(img_y[level].item())
        if img_label > 0:
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = node.full_prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    patch_df_list = None

    if mode == 1 or mode == 3:
        patch_df_list = []

    for j in range(n_prototypes):
        # We assume class_specifc is true
        # target_class is the class of the class_specific prototype
        target_class = torch.argmax(
            node.prototype_class_identity[j]
        ).item() + 1
        # if there is not images of the target_class from this batch
        # we go on to the next prototype
        if len(class_to_img_index_dict[target_class]) == 0:
            continue
        proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < node.global_min_proto_dist[j]:
            if mode == 1:
                batch_argmin_proto_dist_j = [np.argmin(proto_dist_j[:,0,0]),0,(j % (n_prototypes // num_classes))]
            else:
                batch_argmin_proto_dist_j = \
                    list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                        proto_dist_j.shape))
            '''
            change the argmin index from the index among
            images of the target class to the index in the entire search
            batch
            '''
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                :,
                                                fmap_height_start_index:fmap_height_end_index,
                                                fmap_width_start_index:fmap_width_end_index]

            node.global_min_proto_dist[j] = batch_min_proto_dist_j
            node.global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            if no_save:
                continue

            if mode == 1:
                assert search_batch_input.shape[2] == 1
                protoL_rf_info = model.proto_layer_rf_info

                # get the whole image
                original_img_j = search_batch_input[batch_argmin_proto_dist_j[0]]
                original_img_j = original_img_j.numpy()
                original_img_size = original_img_j.shape[0]

                # crop out the receptive field
                # rf_img_j = original_img_j[:, 0, (j % (n_prototypes // num_classes)) * protoL_rf_info[1]: (j % (n_prototypes // num_classes) + 1) * protoL_rf_info[1]]
                rf_img_j = original_img_j[:, 0, (j % (n_prototypes // num_classes)) * protoL_rf_info[1]: (j % (n_prototypes // num_classes) + 1) * protoL_rf_info[1]]

                string_prototype = decode_onehot(rf_img_j, False)

                patch_df_list.append(
                    {
                        "key": j,
                        "class_index": j // (n_prototypes // num_classes),
                        "prototype_index": j % (n_prototypes // num_classes),
                        "patch": string_prototype
                    }
                )
            else:
                # Get the receptive field boundary of the image patch
                # that generates the representation
                protoL_rf_info = model.proto_layer_rf_info
                rf_prototype_j = compute_rf_prototype(search_batch_input.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

                # Get the whole image
                original_img_j = search_batch_input[rf_prototype_j[0]]
                original_img_j = original_img_j.numpy()
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]

                # Crop out the receptive field
                rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                        rf_prototype_j[3]:rf_prototype_j[4], :]
                
                # Save the prototype receptive field information
                node.proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
                node.proto_rf_boxes[j, 1] = rf_prototype_j[1]
                node.proto_rf_boxes[j, 2] = rf_prototype_j[2]
                node.proto_rf_boxes[j, 3] = rf_prototype_j[3]
                node.proto_rf_boxes[j, 4] = rf_prototype_j[4]
                if node.proto_rf_boxes.shape[1] == 6 and search_y is not None:
                    node.proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0],level].item()

                # Find the highly activated region of the original image
                proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
                if model.prototype_activation_function == 'log':
                    proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.epsilon))
                elif model.prototype_activation_function == 'linear':
                    proto_act_img_j = max_dist - proto_dist_img_j
                else:
                    proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
                
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                interpolation=cv2.INTER_CUBIC)

                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                # Crop out the image patch with high activation as prototype image
                proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                            proto_bound_j[2]:proto_bound_j[3], :]
                
                # Save the prototype boundary (rectangular boundary of highly activated region)
                node.proto_bound_boxes[j, 0] = node.proto_rf_boxes[j, 0]
                node.proto_bound_boxes[j, 1] = proto_bound_j[0]
                node.proto_bound_boxes[j, 2] = proto_bound_j[1]
                node.proto_bound_boxes[j, 3] = proto_bound_j[2]
                node.proto_bound_boxes[j, 4] = proto_bound_j[3]
                if node.proto_bound_boxes.shape[1] == 6 and search_y is not None:
                    node.proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0],level].item()
                
                if dir_for_saving_prototypes is not None:
                    # save the numpy array of the prototype self activation
                    if prototype_self_act_filename_prefix is not None:
                        np.save(os.path.join(dir_for_saving_prototypes,
                                            prototype_self_act_filename_prefix + str(j) + '.npy'),
                                proto_act_img_j)
                    if prototype_img_filename_prefix is not None:
                        # save the image of the prototype
                        cv2.imwrite(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '.png'),
                                    cv2.cvtColor(proto_img_j, cv2.COLOR_RGB2BGR))
                        # overlay (upsampled) self activation on original image and save the result
                        rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                        heatmap = np.float32(heatmap) / 255
                        heatmap = heatmap[...,::-1]
                        # Clamp heatmap to [0,1]
                        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                                overlayed_original_img_j,
                                vmin=0.0,
                                vmax=1.0)
                        
                        # if different from the original (whole) image, save the prototype receptive field as png
                        if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                    rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                            overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                        rf_prototype_j[3]:rf_prototype_j[4]]
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                    overlayed_rf_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                        
                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '.png'),
                                proto_img_j,
                                vmin=0.0,
                                vmax=1.0)
    # If we're saving genetic patches. Save 'em here.
    if mode == 1 and patch_df_list is not None and len(patch_df_list):
        patch_df = pd.DataFrame(patch_df_list, columns=["key", "class_index", "prototype_index", "patch"])
        if os.path.isfile(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + ".csv")):
            # Update old file
            existing_df = pd.read_csv(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + ".csv"))
            existing_df = existing_df.set_index("key")
            patch_df = patch_df.set_index("key")

            # Combine the two
            existing_df = existing_df.drop(patch_df.index, errors="ignore")
            existing_df = pd.concat([existing_df,patch_df])

            patch_df = existing_df.reset_index()
        patch_df.to_csv(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + ".csv"), index=False)

    del class_to_img_index_dict

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    no_save=False,
):
    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()

    search_batch_size = dataloader.batch_size

    if prototype_network_parallel.module.mode == 3:
        for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
            init_nodal_push_prototypes(node, root_dir_for_saving_prototypes, epoch_number, log)
        for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
            init_nodal_push_prototypes(node, root_dir_for_saving_prototypes, epoch_number, log)
    else:
        for node in prototype_network_parallel.module.nodes_with_children:
            init_nodal_push_prototypes(node, root_dir_for_saving_prototypes, epoch_number, log)

    for push_iter, ((genetics, image), search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        if prototype_network_parallel.module.mode == 1:
            search_batch_input = genetics
        elif prototype_network_parallel.module.mode == 2:
            search_batch_input = image
        else:
            search_batch_input = (genetics, image)

        """
        search_batch_input is the raw input image, this is used for generating output images.
        search_batch is the input to the network, this is used for generating the feature maps. This may be normalzied
        """
        if preprocess_input_function is not None and prototype_network_parallel.module.mode == 2:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        elif preprocess_input_function is not None and prototype_network_parallel.module.mode == 3:
            search_batch = (search_batch_input[0], preprocess_input_function(search_batch_input[1]))
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            if prototype_network_parallel.module.mode == 3:
                search_batch = (search_batch[0].cuda(), search_batch[1].cuda())
            else:
                search_batch = search_batch.cuda()

            conv_features = prototype_network_parallel.module.conv_features(search_batch)

        if prototype_network_parallel.module.mode == 3:
            for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
                nodal_update_prototypes_on_batch(
                    node=node,
                    search_batch_input=search_batch_input[0],
                    conv_features=conv_features[0],
                    start_index_of_search_batch=start_index_of_search_batch,
                    model=prototype_network_parallel.module.genetic_hierarchical_ppnet,
                    search_y=search_y,
                    preprocess_input_function=preprocess_input_function,
                    prototype_layer_stride=prototype_layer_stride,
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                    no_save=no_save
                )
            for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
                nodal_update_prototypes_on_batch(
                    node=node,
                    search_batch_input=search_batch_input[1],
                    conv_features=conv_features[1],
                    start_index_of_search_batch=start_index_of_search_batch,
                    model=prototype_network_parallel.module.image_hierarchical_ppnet,
                    search_y=search_y,
                    preprocess_input_function=preprocess_input_function,
                    prototype_layer_stride=prototype_layer_stride,
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                    no_save=no_save
                )
        else:
            for node in prototype_network_parallel.module.nodes_with_children:
                nodal_update_prototypes_on_batch(
                    node=node,
                    search_batch_input=search_batch_input,
                    conv_features=conv_features,
                    start_index_of_search_batch=start_index_of_search_batch,
                    prototype_network_parallel=prototype_network_parallel,
                    search_y=search_y,
                    preprocess_input_function=preprocess_input_function,
                    prototype_layer_stride=prototype_layer_stride,
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                    no_save=no_save
                )

    # TODO - Implement bounding box saving
    # if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
    #             proto_rf_boxes)
    #     np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
    #             proto_bound_boxes)

    log('\tExecuting push ...')
    if prototype_network_parallel.module.mode == 3:
        for node in prototype_network_parallel.module.genetic_hierarchical_ppnet.nodes_with_children:
            prototype_update = np.reshape(node.global_min_fmap_patches,
                                        tuple(node.full_prototype_shape))
            node.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
        for node in prototype_network_parallel.module.image_hierarchical_ppnet.nodes_with_children:
            prototype_update = np.reshape(node.global_min_fmap_patches,
                                        tuple(node.full_prototype_shape))
            node.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    else:
        for node in prototype_network_parallel.module.nodes_with_children:
            prototype_update = np.reshape(node.global_min_fmap_patches,
                                        tuple(node.full_prototype_shape))
            node.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))