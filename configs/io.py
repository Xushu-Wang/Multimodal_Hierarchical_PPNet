"""
All functions regarding logging and saving models is defined here
""" 
import datetime
import os
from model.hierarchical import Mode
from typing import Callable, Tuple 
from yacs.config import CfgNode

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
    return logger, f.close

def run_id_accumulator(cfg: CfgNode):
    """
    Add a numeric padding of form *_001 to prevent overwriting of existing runs
    """

    output_dir = os.path.join(
        "..", 
        "output", 
        "genetic_only" if cfg.MODE == Mode.GENETIC else ("image_only" if cfg.MODE == Mode.IMAGE else "joint")
    )

    if cfg.RUN_NAME == '':
        # Generate a run name from the current time
        cfg.RUN_NAME = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')  
    else: 
        # Check if RUN_NAME already exists in output, change it if it doesn't
        i = 0
        while os.path.exists(os.path.join(output_dir, f"{cfg.RUN_NAME}_{i:03d}")):
            i += 1
        cfg.RUN_NAME += f"_{i:03d}"  

    cfg.OUTPUT.MODEL_DIR = os.path.join(output_dir, cfg.RUN_NAME)
    cfg.OUTPUT.IMG_DIR = os.path.join(output_dir, cfg.RUN_NAME, "images")  

    os.makedirs(cfg.OUTPUT.IMG_DIR)

def remove_irrelevant_hyperparams(cfg: CfgNode):  

    if cfg.MODE == 1: 
        # Genetics only run, delete all image-related params . 
        cfg.MODEL.IMAGE.clear()
        cfg.DATASET.IMAGE_PATH.clear()
        cfg.IMAGE_SIZE.clear()
        cfg.TRANSFORM_MEAN.clear()
        cfg.TRANSFORM_STD .clear()
        cfg.OPTIM.TRAIN.COEFS.IMAGE.clear()
    elif cfg.MODE == 2: 
        # image only
        cfg.MODEL.GENETIC.clear()
        cfg.OPTIM.TRAIN.COEFS.GENETIC.clear()
    else: 
        pass 

    # filter dataset args based on whether it's cached 
    if cfg.DATASET.CACHED_DATASET_FOLDER: 
        cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS.clear()
        cfg.DATASET.TRAIN_VAL_TEST_SPLIT.clear()  


    # Make sure you push on the last epoch
    cfg.OPTIM.PUSH.EPOCHS.append(cfg.OPTIM.NUM_EPOCHS - 1)

