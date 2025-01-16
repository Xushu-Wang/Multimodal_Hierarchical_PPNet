from typing import Tuple, Callable
import torch 
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import transforms
from yacs.config import CfgNode
from .dataset import TreeDataset, get_datasets

def create_dataloaders(
    train_dataset: TreeDataset, 
    train_push_dataset: TreeDataset, 
    val_dataset: TreeDataset, 
    test_dataset: TreeDataset, 
    normalize: transforms.Normalize, 
    train_batch_size:int,
    train_push_batch_size:int,
    test_batch_size:int,
    seed: int = 2024
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, transforms.Normalize]:
    np.random.seed(seed) 

    def collate_fn(batch):
        genetics = []
        images = []
        labels = []

        for item in batch:
            if item[0][0] != None:
                genetics.append(item[0][0])
            if item[0][1] != None:
                images.append(item[0][1])
            labels.append(item[1])

        if genetics:
            genetics = torch.stack(genetics)
        if images:
            images = torch.stack(images)
        labels = torch.stack(labels)

        if len(genetics) == 0:
            genetics = None
        if len(images) == 0:
            images = None

        return (genetics, images), labels


    train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=1, pin_memory=False, collate_fn=collate_fn,       
    )
    train_push_loader = DataLoader(
            train_push_dataset, batch_size=train_push_batch_size, shuffle=True,
            num_workers=1, pin_memory=False, collate_fn=collate_fn)
    
    val_loader = DataLoader(
            val_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=1, pin_memory=False, collate_fn=collate_fn)
    
    test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=1, pin_memory=False, collate_fn=collate_fn)  

    return train_loader, train_push_loader, val_loader, test_loader, normalize

def get_dataloaders(cfg: CfgNode, log: Callable, flat_class=False
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, transforms.Normalize]: 
    train, train_push, val, test, normalize = get_datasets(cfg, log, flat_class) 
    log("Getting Dataloaders")

    return create_dataloaders(
        train, 
        train_push, 
        val, 
        test,
        normalize, 
        train_batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
        train_push_batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE,
        test_batch_size=cfg.DATASET.TEST_BATCH_SIZE,
        seed=cfg.SEED
    )

