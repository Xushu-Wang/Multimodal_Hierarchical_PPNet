import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger, makedir
from os import mkdir

from  configs.cfg import get_cfg_defaults

# from dataio.dataset import get_dataset

from model.node import Node
from model.hierarchical_ppnet import Hierarchical_PPNet
from model.utils import get_optimizers, construct_tree, print_tree

import train_and_test as tnt

import prototype.push as push       


def main():
    
    # Step 1: Add training/data parameters and create Logger
    cfg = get_cfg_defaults()


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='cub.yaml')
    args = parser.parse_args()


    cfg.merge_from_file(args.configs)
    
    args = parser.parse_args()

    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))
    
    
    
    # Step 2: Initialize Dataset

    train_loader, train_push_loader, val_loader = get_dataset(cfg, log)
    
    # Step 3: Fix Tree Structure
    
    root = Node("Diptera")
    
    # construct_tree(json_data, root)
    
    root.assign_all_descendents()
    
    log(print_tree(root))
    
    
    # Remember to specify prototypes directory
    
    model_dir = cfg.OUTPUT.MODEL_DIR
    img_dirs = [os.path.join(model_dir, name + "_prototypes") for name in root.classes_with_children()]
    
    for img_dir in img_dirs:
        makedir(img_dir)
    prototype_img_filename_prefix = 'prototype-img'
    prototype_original_img_filename_prefix = 'prototype-original-img'
    proto_bound_boxes_filename_prefix = 'bb'
    
    

    # Construct and parallel the model
    hierarchical_ppnet = Hierarchical_PPNet()
    ppnet_multi = torch.nn.DataParallel(hierarchical_ppnet) 
    class_specific = True
    
    
    
    joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(cfg, hierarchical_ppnet)

    log('start training')
    
    # # Prepare loss function
    # coefs = {
    #     'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
    #     'clst': cfg.OPTIM.COEFS.CLST,
    #     'sep': cfg.OPTIM.COEFS.SEP,
    #     'l1': cfg.OPTIM.COEFS.L1
    # }

    # if cfg.DATASET.NAME == 'multimodal':
    #     for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
    #         log('epoch: \t{0}'.format(epoch))

    #         last_only_multimodal(model=ppnet_multi, log=log)
            
    #         _ = train_multimodal(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, 
    #                         class_specific=class_specific, coefs=coefs, log=log)
            
    #         accu = test_multimodal(model=ppnet_multi, dataloader=val_loader,
    #                         class_specific=class_specific, log=log)
             
                     
    #     if cfg.OPTIM.JOINT:
    #         for i in range(5):
    #             joint_multimodal(model=ppnet_multi, log=log)
                
    #             _ = train_multimodal(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, 
    #                         class_specific=class_specific, coefs=coefs, log=log)
            
    #             accu = test_multimodal(model=ppnet_multi, dataloader=val_loader,
    #                             class_specific=class_specific, log=log)
                
    #         push.push_prototypes(
    #             train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
    #             prototype_network_parallel=ppnet_multi.image_net, # pytorch network with prototype_vectors
    #             class_specific=class_specific,
    #             preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
    #             prototype_layer_stride=1,
    #             root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
    #             epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
    #             prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
    #             prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
    #             proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
    #             save_prototype_class_identity=True,
    #             log=log,
    #             no_save=cfg.OUTPUT.NO_SAVE,
    #             fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
            
    #         push.push_prototypes(
    #             train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
    #             prototype_network_parallel=ppnet_multi.genetic_net, # pytorch network with prototype_vectors
    #             class_specific=class_specific,
    #             preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
    #             prototype_layer_stride=1,
    #             root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
    #             epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
    #             prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
    #             prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
    #             proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
    #             save_prototype_class_identity=True,
    #             log=log,
    #             no_save=cfg.OUTPUT.NO_SAVE,
    #             fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
            
    #     logclose()

        
    # else:
    #     for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
    #         log('epoch: \t{0}'.format(epoch))
            
    #         # Warm up and Training Epochs
    #         if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
    #             tnt.warm_only(model=ppnet_multi, log=log)
    #             _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
    #                         class_specific=class_specific, coefs=coefs, log=log)
    #         else:
    #             tnt.joint(model=ppnet_multi, log=log)
    #             _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
    #                         class_specific=class_specific, coefs=coefs, log=log)
    #             joint_lr_scheduler.step()

    #         # Testing Epochs
    #         accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
    #                         class_specific=class_specific, log=log)
    #         save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'nopush', accu=accu,
    #                                     target_accu=0.70, log=log)

    #         # Pushing Epochs
    #         print(os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_' + 'push_weights.pth'))
    #         if epoch >= cfg.OPTIM.PUSH_START and epoch in cfg.OPTIM.PUSH_EPOCHS:
    #             push.push_prototypes(
    #                 train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
    #                 prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
    #                 class_specific=class_specific,
    #                 preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
    #                 prototype_layer_stride=1,
    #                 root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
    #                 epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
    #                 prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
    #                 prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
    #                 proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
    #                 save_prototype_class_identity=True,
    #                 log=log,
    #                 no_save=cfg.OUTPUT.NO_SAVE,
    #                 fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
                
    #             accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
    #                             class_specific=class_specific, log=log)
    #             save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push', accu=accu,
    #                                         target_accu=0.70, log=log)

    #             # Optimize last layer
    #             tnt.last_only(model=ppnet_multi, log=log)
    #             for i in range(20):
    #                 log('iteration: \t{0}'.format(i))
    #                 _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
    #                             class_specific=class_specific, coefs=coefs, log=log)
    #                 accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
    #                                 class_specific=class_specific, log=log)
    #                 save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=accu, target_accu=0.70, log=log)

    #             # Print the weights of the last layer
    #             # Save the weigts of the last layer
    #             torch.save(ppnet, os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_push_weights.pth'))


    #     logclose()
        


if __name__ == '__main__':
    main()
    
