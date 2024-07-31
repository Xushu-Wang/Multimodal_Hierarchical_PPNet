import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger
from os import mkdir

from  configs.cfg import get_cfg_defaults
from dataio.tree import get_dataloaders

from model.model import construct_ppnet
from model.utils import get_optimizers

import train.train_and_test as tnt
from train.train_multimodal import train_multimodal, test_multimodal, last_only_multimodal, joint_multimodal

import prototype.push as push       
from utils.util import handle_run_name_weirdness


def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image.yaml')
    cfg.merge_from_file(args.configs)
    
    args = parser.parse_args()

    handle_run_name_weirdness(cfg)
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        mkdir(cfg.OUTPUT.IMG_DIR)


    try:
        train_loader, train_push_loader, val_loader, test_loader = get_dataloaders(cfg, log)
        tree_ppnet = construct_tree_ppnet(cfg)
    except Exception as e:
        # Print e with the traceback
        log(f"ERROR: {e}")
        logclose()
        raise(e)

    # Construct and parallel the model
    
    if cfg.DATASET.NAME == 'multimodal':
        ppnet.load_state_dict_genetic(cfg.DATASET.GENETIC.MODEL_PATH)
        ppnet.load_state_dict_img(cfg.DATASET.IMAGE.MODEL_PATH)
    
    ppnet_multi = torch.nn.DataParallel(ppnet) 
    class_specific = True
    
    if cfg.DATASET.NAME == 'multimodal':
        last_layer_optimizer = get_optimizers(cfg, ppnet)
    else: 
        joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(cfg, ppnet)

    log('start training')
    
    # Prepare loss function
    coefs = {
        'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
        'clst': cfg.OPTIM.COEFS.CLST,
        'sep': cfg.OPTIM.COEFS.SEP,
        'l1': cfg.OPTIM.COEFS.L1
    }

    if cfg.DATASET.NAME == 'multimodal':
        for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
            log('epoch: \t{0}'.format(epoch))

            last_only_multimodal(model=ppnet_multi, log=log)
            
            _ = train_multimodal(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, 
                            class_specific=class_specific, coefs=coefs, log=log)
            
            accu = test_multimodal(model=ppnet_multi, dataloader=val_loader,
                            class_specific=class_specific, log=log)
             
                     
        if cfg.OPTIM.JOINT:
            for i in range(5):
                joint_multimodal(model=ppnet_multi, log=log)
                
                _ = train_multimodal(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, 
                            class_specific=class_specific, coefs=coefs, log=log)
            
                accu = test_multimodal(model=ppnet_multi, dataloader=val_loader,
                                class_specific=class_specific, log=log)
                
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi.image_net, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                save_prototype_class_identity=True,
                log=log,
                no_save=cfg.OUTPUT.NO_SAVE,
                fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
            
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi.genetic_net, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                save_prototype_class_identity=True,
                log=log,
                no_save=cfg.OUTPUT.NO_SAVE,
                fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
            
        logclose()

        
    else:
        for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
            log('epoch: \t{0}'.format(epoch))
            
            # Warm up and Training Epochs
            if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
                tnt.warm_only(model=ppnet_multi, log=log)
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
            else:
                tnt.joint(model=ppnet_multi, log=log)
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
                joint_lr_scheduler.step()

            # Testing Epochs
            accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                            class_specific=class_specific, log=log)
            save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'nopush', accu=accu,
                                        target_accu=0.70, log=log)

            # Pushing Epochs
            print(os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_' + 'push_weights.pth'))
            if epoch >= cfg.OPTIM.PUSH_START and epoch in cfg.OPTIM.PUSH_EPOCHS:
                push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=cfg.OUTPUT.PREPROCESS_INPUT_FUNCTION, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                    prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                    proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                    save_prototype_class_identity=True,
                    log=log,
                    no_save=cfg.OUTPUT.NO_SAVE,
                    fix_prototypes=cfg.DATASET.GENETIC.FIX_PROTOTYPES)
                
                accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                                class_specific=class_specific, log=log)
                save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push', accu=accu,
                                            target_accu=0.70, log=log)

                # Optimize last layer
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=val_loader,
                                    class_specific=class_specific, log=log)
                    save_model_w_condition(model=ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=accu, target_accu=0.70, log=log)

                # Print the weights of the last layer
                # Save the weigts of the last layer
                torch.save(ppnet, os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_push_weights.pth'))


        logclose()
        


if __name__ == '__main__':
    main()
    
