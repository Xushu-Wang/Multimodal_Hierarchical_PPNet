import torch 
from typing import Callable
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from .dataset import get_datasets, TreeDataset
from typing import Any, Tuple 

TreeDL = DataLoader[TreeDataset]

def get_dataloaders(cfg: CfgNode, log: Callable) \
-> Tuple[TreeDL, TreeDL, TreeDL, TreeDL]:
    train_ds, push_ds, val_ds, test_ds = get_datasets(cfg, log) 

    def collate_fn(batch): 
        '''
        postprocessing collate function
        '''
        genetics = []
        images = []
        labels = []
        flat_labels = []

        for item in batch:
            if item[0][0] != None:
                genetics.append(item[0][0])
            if item[0][1] != None:
                images.append(item[0][1])
            labels.append(item[1][0])
            flat_labels.append(item[1][1])

        if genetics:
            genetics = torch.stack(genetics)
        if images:
            images = torch.stack(images)
        labels = torch.stack(labels)
        flat_labels = torch.stack(flat_labels)

        if len(genetics) == 0:
            genetics = None
        if len(images) == 0:
            images = None

        return (genetics, images), (labels, flat_labels)

    train_loader = DataLoader(
            train_ds, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=1, pin_memory=False, collate_fn=collate_fn,       
    )
    push_loader = DataLoader(
        push_ds, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=False, collate_fn=collate_fn
    )  

    return train_loader, push_loader, val_loader, test_loader
