import argparse, os
import torch
from prototype.prune import prune_prototypes
from os import mkdir

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, save_model_w_condition
from model.model import Mode
from dataio.dataloader import get_dataloaders
from model.model import construct_tree_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
import prototype.push as push       

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/genetics.yaml')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.configs)
    run_id_accumulator(cfg)

    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        mkdir(cfg.OUTPUT.IMG_DIR)

    try:
        train_loader, train_push_loader, val_loader, _, image_normalizer = get_dataloaders(cfg, log)
        
        tree_ppnet = construct_tree_ppnet(cfg).to("cuda")

        tree_ppnet_multi = torch.nn.DataParallel(tree_ppnet)
        tree_ppnet_multi = tree_ppnet_multi

        joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(tree_ppnet)

        # Construct and parallel the model
        log('start training')
        
        # Prepare loss function
        coefs = {
            'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
            'clst': cfg.OPTIM.COEFS.CLST,
            'sep': cfg.OPTIM.COEFS.SEP,
            'l1': cfg.OPTIM.COEFS.L1,
            "correspondence": cfg.OPTIM.COEFS.CORRESPONDENCE,
            'CEDA': False
        }

        for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS):
            log('epoch: \t{0}'.format(epoch))
            
            # Warm up and Training Epochs
            if not args.validate:
                if epoch < cfg.OPTIM.NUM_WARM_EPOCHS:
                    tnt.warm_only(model=tree_ppnet_multi, log=log)
                    tnt.train(
                        model=tree_ppnet_multi, 
                        dataloader=train_loader, 
                        optimizer=warm_optimizer,
                        parallel_mode=cfg.DATASET.PARALLEL_MODE,
                        global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                        coefs=coefs,
                        log=log
                    )
                else:
                    if tree_ppnet.mode == Mode.MULTIMODAL and not cfg.DATASET.PARALLEL_MODE:
                        tnt.multi_last_layer(model=tree_ppnet_multi, log=log)
                        tnt.train(
                            model=tree_ppnet_multi,
                            global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY, 
                            parallel_mode=cfg.DATASET.PARALLEL_MODE,
                            dataloader=train_loader, 
                            optimizer=last_layer_optimizer,
                            coefs=coefs,
                            log=log
                        )
                    
                    tnt.joint(model=tree_ppnet_multi, log=log)
                    tnt.train( 
                        model=tree_ppnet_multi,
                        global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                        parallel_mode=cfg.DATASET.PARALLEL_MODE,
                        dataloader=train_loader,
                        optimizer=joint_optimizer,
                        coefs=coefs,
                        log=log
                    )
                    joint_lr_scheduler.step()
            
            # Testing Epochs
            prob_accu = tnt.test(
                model=tree_ppnet_multi,
                dataloader=val_loader,
                log=log,
                global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                parallel_mode=cfg.DATASET.PARALLEL_MODE
            )
            save_model_w_condition(
                model=tree_ppnet, 
                model_dir=cfg.OUTPUT.MODEL_DIR, 
                model_name=str(epoch) + 'nopush', 
                accu=prob_accu,
                target_accu=0, 
                log=log
            )

            if args.validate:
                break
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
                prob_accu = tnt.test(model=tree_ppnet_multi, dataloader=val_loader,
                                     log=log,
                                global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,parallel_mode=cfg.DATASET.PARALLEL_MODE)
                save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + 'push',
                                            target_accu=0, log=log, accu=prob_accu)

                # Optimize last layer
                if cfg.MODEL.PRUNE and epoch >= cfg.OPTIM.PRUNE_START:
                    if cfg.MODEL.PRUNING_TYPE == "weights":
                        tnt.last_only(model=tree_ppnet_multi, log=log)
                        for i in range(10):
                            log('[weights pruning] iteration: \t{0}'.format(i))
                            _ = tnt.train(
                                model=tree_ppnet_multi,
                                global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                                parallel_mode=cfg.DATASET.PARALLEL_MODE,
                                dataloader=train_loader,
                                optimizer=last_layer_optimizer,
                                coefs=coefs,
                                log=log
                            )
                            prob_accu = tnt.test(
                                model=tree_ppnet_multi, 
                                parallel_mode=cfg.DATASET.PARALLEL_MODE, 
                                global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY, 
                                dataloader=val_loader,
                                log=log
                            )

                            if tree_ppnet.mode == Mode.MULTIMODAL and not cfg.DATASET.PARALLEL_MODE:
                                tnt.multi_last_layer(model=tree_ppnet_multi, log=log)

                                tnt.train(
                                    model=tree_ppnet_multi, 
                                    dataloader=train_loader, 
                                    optimizer=last_layer_optimizer,
                                    coefs=coefs, 
                                    parallel_mode=cfg.DATASET.PARALLEL_MODE, 
                                    global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY, 
                                    log=log
                                )

                                prob_accu = tnt.test(
                                    model=tree_ppnet_multi, 
                                    parallel_mode=cfg.DATASET.PARALLEL_MODE, 
                                    global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY, 
                                    dataloader=val_loader,
                                    log=log)

                    prune_prototypes(
                        tree_ppnet_multi,
                        train_push_loader,
                        preprocess_input_function=image_normalizer,
                        pruning_type=cfg.MODEL.PRUNING_TYPE,
                        k=cfg.MODEL.PRUNING_K,
                        tau=cfg.MODEL.PRUNING_TAU,
                        log=log
                    )

                # Optimize last layer again
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(
                        model=tree_ppnet_multi,
                        dataloader=train_loader,
                        optimizer=last_layer_optimizer,
                        parallel_mode=cfg.DATASET.PARALLEL_MODE,
                        global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                        coefs=coefs,
                        log=log
                    )
                    prob_accu = tnt.test(
                        model=tree_ppnet_multi,
                        dataloader=val_loader,
                        parallel_mode=cfg.DATASET.PARALLEL_MODE,
                        global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                        log=log
                    )
                    save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=prob_accu, target_accu=0, log=log)
                    if tree_ppnet.mode == Mode.MULTIMODAL and not cfg.DATASET.PARALLEL_MODE:
                        tnt.multi_last_layer(model=tree_ppnet_multi, log=log)
                        _ = tnt.train(
                            model=tree_ppnet_multi,
                            global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                            parallel_mode=cfg.DATASET.PARALLEL_MODE,
                            dataloader=train_loader,
                            optimizer=last_layer_optimizer,
                            coefs=coefs,
                            log=log
                        )
                        prob_accu = tnt.test(
                            model=tree_ppnet_multi,
                            global_ce=cfg.OPTIM.GLOBAL_CROSSENTROPY,
                            dataloader=val_loader,
                            parallel_mode=cfg.DATASET.PARALLEL_MODE,
                            log=log
                        )
                        save_model_w_condition(model=tree_ppnet, model_dir=cfg.OUTPUT.MODEL_DIR, model_name=str(epoch) + '_' + 'push', accu=prob_accu, target_accu=0, log=log)

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
    
