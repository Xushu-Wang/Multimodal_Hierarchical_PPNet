"""
This trains a ResNet backbone or CNN backbone for genetic or image classification.
"""

import argparse, os, wandb
import torch
import numpy as np
from yacs.config import CfgNode
from typing import Callable
from configs.cfg import get_cfg_defaults

from dataio.dataset import Mode
from model.features.genetic_features import GeneticCNN2D
from configs.io import create_logger, run_id_accumulator 
from torchvision.models import resnet50
from dataio.dataloader import get_dataloaders
import torch.optim as optim

run_mode = {1 : "Genetic", 2 : "Image", 3 : "Multimodal"} 

def main(cfg: CfgNode, log: Callable):
    device = torch.device(cfg.DEVICE)
    train_loader, _, val_loader, _ = get_dataloaders(cfg, log)  

    class_count = train_loader.dataset.hierarchy.levels.counts[-1] # type: ignore  

    if cfg.MODE == Mode.GENETIC.value:
        model = GeneticCNN2D(720, class_count, include_connected_layer=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        model = resnet50(weights='DEFAULT') 
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, class_count)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            momentum=.9
        )
    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=5, threshold=1e-6)
    criterion = torch.nn.CrossEntropyLoss().to(device) 

    max_accuracy = max_accuracy_epoch = 0

    for epoch in range(150): 
        print(f"===== EPOCH: {epoch + 1} =====")
        total_loss = 0.0
        correct_guesses = 0

        model.train() 
        for (genetics, image), (_, flat_label) in train_loader:  
            inputs = genetics.to(device) if cfg.MODE == Mode.GENETIC.value else image.to(device)
            labels = flat_label[:,-1].to(device) 

            optimizer.zero_grad() 

            outputs = model(inputs) 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss

            y_pred = torch.argmax(outputs, dim=1)
            correct_guesses += torch.sum(y_pred == labels)

        scheduler.step(total_loss / len(train_loader.dataset)) # type: ignore
        train_acc = correct_guesses / len(train_loader.dataset) # type: ignore
        log(f"Train Loss/Acc: {total_loss:.5f}, {train_acc:.5f}")
        
        # Evaluate on test set with balanced accuracy
        model.eval()
        val_accuracy = torch.zeros(4)
        hier = train_loader.dataset.hierarchy # type: ignore 
        species_to_genus = hier.species_to_genus.to(device)
        species_to_family = hier.species_to_family.to(device)
        species_to_order = hier.species_to_order.to(device)

        with torch.no_grad():
            for (genetics, image), (_, flat_label) in val_loader:
                inputs = genetics.to(device) if cfg.MODE == Mode.GENETIC.value else image.to(device)
                labels = flat_label[:,-1].to(device)
                flat_label = flat_label.to(device)
                logits = model(inputs) 

                species_pred = torch.argmax(logits, dim=1)
                order_pred = species_to_order[species_pred]
                family_pred = species_to_family[species_pred]
                genus_pred = species_to_genus[species_pred] 

                val_accuracy[0] += torch.sum((flat_label[:,0] == order_pred)).item()
                val_accuracy[1] += torch.sum((flat_label[:,1] == family_pred)).item()
                val_accuracy[2] += torch.sum((flat_label[:,2] == genus_pred)).item()
                val_accuracy[3] += torch.sum((flat_label[:,3] == species_pred)).item() 

            val_accuracy /= len(val_loader.dataset) # type: ignore

        log(f"Valid Acc: {val_accuracy}")

        if val_accuracy[-1] > max_accuracy:
            max_accuracy = val_accuracy[-1]
            max_accuracy_epoch = epoch 
            torch.save(model.state_dict(), os.path.join("backbones", f"{cfg.RUN_NAME}_best.pth")) 

        wandb.log({
            "train-loss" : total_loss, 
            "train-acc" : train_acc, 
            "val-acc-order" : val_accuracy[0], 
            "val-acc-family" : val_accuracy[1], 
            "val-acc-genus" : val_accuracy[2], 
            "val-acc-species" : val_accuracy[3]
        })
        
    log(f"Best Accuracy: {max_accuracy:.4f} at epoch {max_accuracy_epoch+1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='image_backbone.yaml')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("configs", args.config))

    run_id_accumulator(cfg) 

    if os.path.exists(os.path.join("backbones", f"{cfg.RUN_NAME}_best.pth")): 
        raise Exception("You are overriding a previously saved weights. Use another name.")

    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    wandb.init(
        project=f"{run_mode[cfg.MODE]} Blackbox Backbone",
        name=cfg.RUN_NAME,
        config=cfg,
        mode=cfg.WANDB_MODE,
        entity="charlieberens-duke-university"
    )
    log(str(cfg))
    try: 
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        main(cfg, log) 
    except Exception as e: 
        wandb.finish() 
        logclose() 
        raise(e)


