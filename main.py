import torch
import numpy as np
import argparse, os, wandb
from yacs.config import CfgNode

from configs.cfg import get_cfg_defaults 
from configs.io import create_logger, run_id_accumulator, remove_irrelevant_hyperparams
from dataio.dataloader import get_dataloaders
from model.multimodal import construct_ppnet
from train.optimizer import get_optimizers
import train.train_and_test as tnt
from prototype import push, prune 
from typing import Callable
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

run_mode = {1 : "genetic", 2 : "image", 3 : "multimodal"} 

def main(cfg: CfgNode, log: Callable):
    train_loader, push_loader, val_loader, _ = get_dataloaders(cfg, log)
    log("Dataloaders Constructed")

    model = construct_ppnet(cfg, log).cuda()
    log("ProtoPNet constructed")
    wandb.watch(model, log="all", log_freq=2) 
    # wandb.watch(model.gen_net.features, log="all", log_freq=2) 
    # wandb.watch(model.img_net.features, log="all", log_freq=2) 
    # wandb.watch(model.gen_net.root, log="all", log_freq=2) 
    # wandb.watch(model.img_net.root, log="all", log_freq=2) 

    warm_optim, joint_optim, last_optim, test_optim = get_optimizers(model, cfg) 

    # initial logging before any training 
    train_loss = tnt.traintest(model, train_loader, test_optim, cfg)  
    test_loss = tnt.traintest(model, val_loader  , test_optim, cfg)
    log(str(train_loss))
    log(str(test_loss))
    wandb.log(train_loss.to_dict() | test_loss.to_dict()) 

    for epoch in range(cfg.OPTIM.NUM_EPOCHS): 
        if epoch in cfg.OPTIM.PRUNE.EPOCHS:
            log(f"Pruning")
            prune.prune(model, train_loader, cfg)

        if epoch < cfg.OPTIM.NUM_WARM_EPOCHS: 
            log(f'Warm Epoch: {epoch + 1}/{cfg.OPTIM.NUM_WARM_EPOCHS}')
            train_loss = tnt.traintest(model, train_loader, warm_optim, cfg)  
            test_loss = tnt.traintest(model, val_loader  , test_optim, cfg)

        elif epoch in cfg.OPTIM.PUSH.EPOCHS: 
            log(f'Push Epoch: {epoch + 1}/{cfg.OPTIM.NUM_EPOCHS}') 
            push.push(model, push_loader, cfg, stride=1, epoch=epoch)
            
            # need to implement pruning here
            
            for _ in range(cfg.OPTIM.PUSH.N_LAST_LAYER):
                tnt.traintest(model, train_loader, last_optim, cfg)

            train_loss = tnt.traintest(model, train_loader, last_optim, cfg)
            test_loss = tnt.traintest(model, val_loader, test_optim, cfg)
            
        else: 
            log(f'Train Epoch: {epoch + 1}/{cfg.OPTIM.NUM_EPOCHS}') 
            train_loss = tnt.traintest(model, train_loader, joint_optim, cfg)
            test_loss = tnt.traintest(model, val_loader, test_optim, cfg)

        log(str(train_loss))
        log(str(test_loss))
        wandb.log(train_loss.to_dict() | test_loss.to_dict()) 

        if cfg.OUTPUT.SAVE_MODEL: 
            if (epoch in cfg.OUTPUT.SAVE_MODEL_EPOCHS) or (epoch in cfg.OPTIM.PUSH.EPOCHS): 
                status = "push" if (epoch in cfg.OPTIM.PUSH.EPOCHS) else "train"
                match cfg.MODE: 
                    case 1:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"gen_{status}_ep{epoch}.pth"))
                    case 2:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"img_{status}_ep{epoch}.pth"))
                    case 3:
                        torch.save(model.gen_net.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"gen_{status}_ep{epoch}.pth"))
                        torch.save(model.img_net.state_dict(), os.path.join(cfg.OUTPUT.MODEL_DIR, f"img_{status}_ep{epoch}.pth")) 
                log("Saved Models")

    wandb.finish()
    logclose()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='multi.yaml')

    parser.add_argument('--gcrs_ent', type=float, default=20.0) 
    parser.add_argument('--gclst', type=float, default=-100.0) 
    parser.add_argument('--gsep', type=float, default=10.0) 
    parser.add_argument('--gl1', type=float, default=0.0) 
    parser.add_argument('--gortho', type=float, default=0.0) 
    
    parser.add_argument('--icrs_ent', type=float, default=20.0) 
    parser.add_argument('--iclst', type=float, default=-100.0) 
    parser.add_argument('--isep', type=float, default=10.0) 
    parser.add_argument('--il1', type=float, default=0.0) 
    parser.add_argument('--iortho', type=float, default=0.0)
    
    parser.add_argument('--corr', type=float, default=0.0)
    parser.add_argument('--run-name', type=str, default=None)
    
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("configs", args.config))

    if args.run_name is not None:
        cfg.RUN_NAME = args.run_name

    run_id_accumulator(cfg) 
    remove_irrelevant_hyperparams(cfg) 

    if cfg.MODE == 3:
        cfg.OPTIM.TRAIN.COEFS.CORRESPONDENCE = args.corr 
    if cfg.MODE == 1 or cfg.MODE == 3: 
        cfg.OPTIM.TRAIN.COEFS.GENETIC.CRS_ENT = args.gcrs_ent
        cfg.OPTIM.TRAIN.COEFS.GENETIC.CLST = args.gclst
        cfg.OPTIM.TRAIN.COEFS.GENETIC.SEP = args.gsep
        cfg.OPTIM.TRAIN.COEFS.GENETIC.L1 = args.gl1
        cfg.OPTIM.TRAIN.COEFS.GENETIC.ORTHO = args.gortho
    if cfg.MODE == 2 or cfg.MODE == 3: 
        cfg.OPTIM.TRAIN.COEFS.IMAGE.CRS_ENT = args.icrs_ent
        cfg.OPTIM.TRAIN.COEFS.IMAGE.CLST = args.iclst
        cfg.OPTIM.TRAIN.COEFS.IMAGE.SEP = args.isep
        cfg.OPTIM.TRAIN.COEFS.IMAGE.L1 = args.il1
        cfg.OPTIM.TRAIN.COEFS.IMAGE.ORTHO = args.iortho

    log, logclose = create_logger(os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))

    wandb.init(
        project=f"{run_mode[cfg.MODE]} Hierarchical Protopnet",
        name=cfg.RUN_NAME,
        config=cfg,
        mode=cfg.WANDB_MODE,
        entity="charlieberens-duke-university"
    )
    wandb.log({})
    log(str(cfg))
    try: 
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        main(cfg, log)  
    except Exception as e: 
        wandb.finish() 
        logclose() 
        raise(e)
