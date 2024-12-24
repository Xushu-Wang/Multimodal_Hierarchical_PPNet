"""All functions regarding logging and saving models is defined here""" 
import datetime
import os
from model.model import Mode
import torch
from typing import Callable, Tuple 
from yacs.config import CfgNode

def makedir(path: str) -> None:
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def create_logger(log_filename: str, display: bool = True) -> Tuple[Callable, Callable]:
    f = open(log_filename, 'a+')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
        # Question: do we need to flush()
    return logger, f.close

def save_model_w_condition(
    model: torch.nn.Module, 
    model_dir: str, 
    model_name: str, 
    accu, 
    target_accu, 
    log: Callable = print
) -> None:
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def run_id_accumulator(cfg: CfgNode) -> None:
    """
    All of this prevents overwriting of existing runs.
    """
    if cfg.RUN_NAME == '':
        # Generate a run name from the current time
        cfg.RUN_NAME = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')

    mode_name = "genetic_only" if cfg.DATASET.MODE == Mode.GENETIC else ("image_only" if cfg.DATASET.MODE == Mode.IMAGE else "joint")
    # Check if RUN_NAME already exists in output, change it if it doesn't
    i = 0
    print(os.path.join("../output", cfg.RUN_NAME))
    root_run_name = cfg.RUN_NAME
    while os.path.exists(os.path.join("../output", mode_name, cfg.RUN_NAME)):
        i += 1
        cfg.RUN_NAME = f"{root_run_name}_{i:03d}"

    if cfg.OUTPUT.MODEL_DIR == '':
        cfg.OUTPUT.MODEL_DIR = os.path.join("../output", mode_name, cfg.RUN_NAME)
        # If the model directory doesn't exist, create it
        makedir(cfg.OUTPUT.MODEL_DIR)
    cfg.OUTPUT.IMG_DIR = os.path.join(cfg.OUTPUT.MODEL_DIR, "images")

