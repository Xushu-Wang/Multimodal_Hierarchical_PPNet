import argparse, os
import torch
from utils.util import save_model_w_condition, create_logger
from os import mkdir

from  configs.cfg import get_cfg_defaults
from dataio.tree import get_dataloaders

from model.model import construct_tree_ppnet
from model.utils import get_optimizers

import train_and_test as tnt
# from train.train_multimodal import train_multimodal, test_multimodal, last_only_multimodal, joint_multimodal

import prototype.push as push       
from utils.util import handle_run_name_weirdness


def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image.yaml')
    args = parser.parse_args()
    cfg.merge_from_file(args.configs)

    handle_run_name_weirdness(cfg)
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        mkdir(cfg.OUTPUT.IMG_DIR)

    try:
        train_loader, train_push_loader, val_loader, test_loader, image_normalizer = get_dataloaders(cfg, log)
        tree_ppnet = construct_tree_ppnet(cfg).cuda()

        tree_ppnet_multi = torch.nn.DataParallel(tree_ppnet)
        # TODO - Handle multi models

        tree_ppnet_multi = tree_ppnet_multi

        class_specific = True

        joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(cfg, tree_ppnet)

        # Construct and parallel the model
        log('start training')
        
        # Prepare loss function
        coefs = {
            'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
            'clst': cfg.OPTIM.COEFS.CLST,
            'sep': cfg.OPTIM.COEFS.SEP,
            'l1': cfg.OPTIM.COEFS.L1,
            'CEDA': False
        }

        for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
            log('epoch: \t{0}'.format(epoch))
            
            # Warm up and Training Epochs
            if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
                tnt.warm_only(model=tree_ppnet_multi, log=log)
                _ = tnt.train(model=tree_ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
            else:
                tnt.joint(model=tree_ppnet_multi, log=log)
                _ = tnt.train(model=tree_ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)
                joint_lr_scheduler.step()
            
            if tree_ppnet.mode == 3:
                tnt.multi_last_layer(model=tree_ppnet_multi, log=log)
                _ = tnt.train(model=tree_ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log)

            # Testing Epochs
            accus = tnt.test(model=tree_ppnet_multi, dataloader=val_loader,
                            class_specific=class_specific, log=log)
            save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'nopush', accu=accus.min(),
                                        target_accu=0.70, log=log)

            # Pushing Epochs
            if epoch >= cfg.OPTIM.PUSH_START and epoch in cfg.OPTIM.PUSH_EPOCHS:
                push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=tree_ppnet_multi, # pytorch network with prototype_vectors
                    preprocess_input_function=image_normalizer, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=cfg.OUTPUT.IMG_DIR, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=cfg.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX,
                    prototype_self_act_filename_prefix=cfg.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX,
                    proto_bound_boxes_filename_prefix=cfg.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX,
                    log=log,
                    no_save=cfg.OUTPUT.NO_SAVE
                )
                
                accus = tnt.test(model=tree_ppnet_multi, dataloader=val_loader,
                                class_specific=class_specific, log=log)
                save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push',
                                            target_accu=0.70, log=log, accu=accus.min())

                # Optimize last layer
                tnt.last_only(model=tree_ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=tree_ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accus = tnt.test(model=tree_ppnet_multi, dataloader=val_loader,
                                    class_specific=class_specific, log=log)
                    save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=accus.min(), target_accu=0.70, log=log)

                # Print the weights of the last layer
                # Save the weigts of the last layer
                torch.save(tree_ppnet, os.path.join(cfg.OUTPUT.IMG_DIR, str(epoch) + '_push_weights.pth'))

    except Exception as e:
        # Print e with the traceback
        log(f"ERROR: {e}")
        logclose()
        raise(e)

    logclose()
        


if __name__ == '__main__':
    main()
    
