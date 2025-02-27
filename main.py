import torch
import argparse, os, wandb
from yacs.config import CfgNode

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator
from dataio.dataloader import get_dataloaders
from model.multimodal import construct_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
from train.train_and_test import OptimMode
import prototype.push as push 

from typing import Callable

run_mode = {1 : "Genetic", 2 : "Image", 3 : "Multimodal"} 

# bottleneck in push for image
# add orthogonality for gen/img ppnet loss?

def main(cfg: CfgNode, log: Callable):
    
    if not os.path.exists(cfg.OUTPUT.MODEL_DIR):
        os.mkdir(cfg.OUTPUT.MODEL_DIR)
    if not os.path.exists(cfg.OUTPUT.IMG_DIR):
        os.mkdir(cfg.OUTPUT.IMG_DIR)

    train_loader, train_push_loader, val_loader, _, image_normalizer = get_dataloaders(cfg, log, validate=args.validate)
    log("Dataloaders Constructed")

    model = construct_ppnet(cfg).cuda()
    log("ProtoPNet constructed")

    warm_optim, joint_optim, last_layer_optim = get_optimizers(model, cfg)

    for epoch in range(cfg.OPTIM.NUM_TRAIN_EPOCHS): 

        # run an epoch of training
        if epoch < cfg.OPTIM.NUM_WARM_EPOCHS: 
            log(f'Warm Epoch: {epoch + 1}/{cfg.OPTIM.NUM_WARM_EPOCHS}')
            tnt.train(model, train_loader, warm_optim, cfg, OptimMode.WARM, log) 
            tnt.test(model, val_loader, cfg, log)

        elif epoch in cfg.OPTIM.PUSH_EPOCHS: 
            log(f'Push Epoch: {epoch + 1}/{cfg.OPTIM.NUM_TRAIN_EPOCHS}') 
            tnt.train(model, train_loader, joint_optim, cfg, OptimMode.JOINT, log) 
            tnt.test(model, val_loader, cfg, log)

            push.push(model, train_push_loader, cfg, epoch, image_normalizer, stride = 1)

            # need to implement pruning here

            for _ in range(20):
                tnt.train(model, train_loader, last_layer_optim, cfg, OptimMode.LAST, log)  
                tnt.test(model, val_loader, cfg, log)

            if cfg.OUTPUT.SAVE:
                torch.save(model, os.path.join(cfg.OUTPUT.MODEL_DIR, f"{epoch}_push_full.pth"))
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"{epoch}_push_weights.pth"))
        else: 
            log(f'Train Epoch: {epoch + 1}/{cfg.OPTIM.NUM_TRAIN_EPOCHS}') 
            tnt.train(model, train_loader, joint_optim, cfg, OptimMode.JOINT, log) 
            tnt.test(model, val_loader, cfg, log)

            # need to implement saving models
            if epoch % 5 == 0 and cfg.OUTPUT.SAVE:
                torch.save(model, os.path.join(cfg.OUTPUT.MODEL_DIR, f"{epoch}_full.pth"))
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"{epoch}_weights.pth"))

    wandb.finish()
    logclose()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/multi.yaml')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0') 
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.configs)
    run_id_accumulator(cfg)

    log, logclose = create_logger(os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))

    wandb.init(
        project=f"{run_mode[cfg.DATASET.MODE]} Hierarchical Protopnet",
        name=cfg.RUN_NAME,
        config=cfg,
        mode=cfg.WANDB_MODE
    )
    try:
        main(cfg, log)
    except Exception as e: 
        wandb.finish() 
        logclose() 
        raise(e)
    
