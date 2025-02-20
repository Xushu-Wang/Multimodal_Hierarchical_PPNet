import torch
import argparse, os
from os import mkdir

import wandb

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator
from dataio.dataloader import get_dataloaders
from model.model import construct_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt

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
    run = wandb.init(
        project="Multimodal Hierarchical Protopnet",
        name=cfg.RUN_NAME,
        config=cfg,
        mode=cfg.WANDB_MODE
    )
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        mkdir(cfg.OUTPUT.IMG_DIR)

    try:
        train_loader, train_push_loader, val_loader, _, image_normalizer = get_dataloaders(cfg, log, validate=args.validate)
        print("Dataloaders Got")

        tree_ppnet = construct_ppnet(cfg).to("cuda")
        print("PPNET constructed")

        joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(tree_ppnet)

        for epoch in range(cfg.OPTIM.NUM_WARM_EPOCHS): 
            log(f'Warm Epoch: {epoch + 1}/{cfg.OPTIM.NUM_WARM_EPOCHS}')
            tnt.warm_only(model=tree_ppnet) 
            tnt.train(
                model=tree_ppnet,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                cfg=cfg,
                run=run,
                log=log
            )

        for epoch in range(cfg.OPTIM.NUM_WARM_EPOCHS, cfg.OPTIM.NUM_TRAIN_EPOCHS): 
            log(f'Train Epoch: {epoch + 1}/{cfg.OPTIM.NUM_TRAIN_EPOCHS}')
            tnt.joint(model=tree_ppnet) 
            tnt.train(
                model=tree_ppnet,
                dataloader=train_loader,
                optimizer=warm_optimizer,
                cfg=cfg,
                run=run,
                log=log
            )

    except Exception as e:
        # Print e with the traceback
        run.finish()
        logclose()
        raise(e)

    run.finish()
    logclose()

if __name__ == '__main__':
    main()
    
