import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger, makedir
from os import mkdir
import json

from configs.cfg import get_cfg_defaults
from prototype.push import push_prototypes
from dataio.tree import get_dataloaders

from model.node import Node
from model.hierarchical_ppnet import Hierarchical_PPNet
from model.model import construct_ppnet

from model.utils import get_optimizers, construct_tree, print_tree, adjust_learning_rate

from utils.util import handle_run_name_weirdness

import train_and_test as tnt

def main():
    # Step 1: Add training/data parameters and create Logger
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image.yaml')
    args = parser.parse_args()

    cfg.merge_from_file(args.configs)
    
    args = parser.parse_args()

    handle_run_name_weirdness(cfg)
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))

    try:
        # Step 2: Initialize Dataset
        # NOTE: Use val_loader. We're not using test_loader until we're almost done with the paper.
        train_loader, train_push_loader, val_loader, test_loader = get_dataloaders(cfg, log)

        with open("class_trees/example.json", 'r') as file:
            data = json.load(file)
                        
        # Step 3: Fix Tree Structure
        root = Node('Diptera Order')
        
        construct_tree(data['tree'], root)        
        
        print_tree(root)
        
        root.assign_all_descendents()    
        
        # Remember to specify prototypes directory
        
        model_dir = cfg.OUTPUT.MODEL_DIR
        img_dirs = [os.path.join(model_dir, name + "_prototypes") for name in root.classes_with_children()]
        
        for img_dir in img_dirs:
            makedir(img_dir)
        prototype_img_filename_prefix = 'prototype-img'
        prototype_original_img_filename_prefix = 'prototype-original-img'
        proto_bound_boxes_filename_prefix = 'bb'
        
        # Construct and parallel the model
        hierarchical_ppnet = construct_ppnet(cfg, root)
        ppnet_multi = torch.nn.DataParallel(hierarchical_ppnet) 
        class_specific = True
        
        # Prepare optimizer
        through_protos_optimizer, warm_optimizer, joint_optimizer, last_layer_optimizer = get_optimizers(cfg, hierarchical_ppnet, root)
        
        optimizers = [through_protos_optimizer,warm_optimizer,joint_optimizer]


        log('start training')
        
        # Prepare loss function
        coefs = {
            'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
            'clst': cfg.OPTIM.COEFS.CLST,
            'sep': cfg.OPTIM.COEFS.SEP,
            'l1': cfg.OPTIM.COEFS.L1
        }    
        
        
        # dictionaries
        class_names = os.listdir("datasets/full_bioscan_images")
        class_names.sort()
        label2name = {i : name for (i,name) in enumerate(class_names)}
        IDcoarse_names = root.get_children_names()

        # train the model
        log('start training')
    
        # warm epoch
        for epoch in range(cfg.OPTIM.NUM_WARM_EPOCHS):
            log('epoch: \t{0}'.format(epoch))    	
            
            tnt.coarse_warm(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, label2name=label2name, optimizer=warm_optimizer, coefs = coefs, class_specific=class_specific, log=log, warm_up = True)	

        # proto epoch

        for epoch in range(cfg.OPTIM.NUM_PROTO_EPOCHS):
            log('epoch: \t{0}'.format(epoch))

            # train
            tnt.up_to_protos(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, label2name=label2name, optimizer=through_protos_optimizer, coefs = coefs, class_specific=class_specific, log=log)
            
        
            if epoch > 0 and epoch % args.push_every == 0 or epoch == cfg.OPTIM.NUM_PROTO_EPOCHS-1:
                push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    label2name=label2name,
                    class_specific=class_specific,
                    preprocess_input_function=cfg.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                    prototype_layer_stride=1,           
                    root_dir_for_saving_prototypes=model_dir, # if not None, prototypes will be saved here
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_original_img_filename_prefix=prototype_original_img_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log)

                acc, _ = tnt.valid(model=ppnet_multi, dataloader=val_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

                save_model_w_condition(model=hierarchical_ppnet, model_dir=model_dir, model_name='best_model_protos_opt', accu=acc,
                            target_accu=0.75, log=log)


                tnt.last_layers(model=ppnet_multi, log=log)
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, label2name=label2name, optimizer=last_layer_optimizer, args = args, class_specific=class_specific, log=log)

                acc, _ = tnt.valid(model=ppnet_multi, dataloader=val_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

                save_model_w_condition(model=hierarchical_ppnet, model_dir=model_dir, model_name='best_model_protos_opt', accu=acc,
                            target_accu=0.75, log=log)
            

            if (epoch+1) % args.decay == 0:
                log('lowered lrs by factor of 10')
                adjust_learning_rate(optimizers)


            log("optimize joint")
            for epoch in range(cfg.OPTIM.NUM_JOINT_EPOCHS):

                log('epoch: \t{0}'.format(epoch))

                tnt.joint(model=ppnet_multi, log=log)
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, label2name=label2name, optimizer=joint_optimizer, args = args, class_specific=class_specific, log=log)				
                
                if epoch > 0 and epoch % args.push_every == 0 or epoch == cfg.OPTIM.NUM_JOINT_EPOCHS - 1:

                    push_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                        label2name=label2name,
                        class_specific=class_specific,
                        preprocess_input_function=cfg.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                        prototype_layer_stride=1,           
                        root_dir_for_saving_prototypes=model_dir, # if not None, prototypes will be saved here
                        prototype_img_filename_prefix=prototype_img_filename_prefix,
                        prototype_original_img_filename_prefix=prototype_original_img_filename_prefix,
                        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                        save_prototype_class_identity=True,
                        log=log)

                    acc, _ = tnt.valid(model=ppnet_multi, dataloader=val_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

                    save_model_w_condition(model=hierarchical_ppnet, model_dir=model_dir, model_name='best_model_joint_opt', accu=acc,
                                target_accu=0.75, log=log)
            
                    for i in range(2):		
                        tnt.last_layers(model=ppnet_multi, log=log)
                        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, label2name=label2name, optimizer=last_layer_optimizer, args = args, class_specific=class_specific, log=log)			

                        acc, _ = tnt.valid(model=ppnet_multi, dataloader=val_loader, label2name=label2name, args=args, class_specific=class_specific, log=log)

                        save_model_w_condition(model=hierarchical_ppnet, model_dir=model_dir, model_name='best_model_joint_opt', accu=acc,
                                    target_accu=0.75, log=log)
                        
                        
                if (epoch+1) % args.decay == 0:
                    log('lowered lrs by factor of 10')
                    adjust_learning_rate(optimizers)
                    
    except Exception as e:
        # Print e with the traceback
        log(f"ERROR: {e}")
        logclose()
        raise(e)

    logclose()

if __name__ == '__main__':
    main()
    
