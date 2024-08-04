import datetime
import os
import torch
import numpy as np


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1



def create_logger(log_filename, display=True):
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

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def handle_run_name_weirdness(cfg):
    """
    All of this prevents overwriting of existing runs.
    """
    if cfg.RUN_NAME == '':
        # Generate a run name from the current time
        cfg.RUN_NAME = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')

    mode_name = "genetic_only" if cfg.DATASET.MODE == 1 else ("image_only" if cfg.DATASET.MODE == 2 else "joint")
    # Check if RUN_NAME already exists in output, change it if it doesn't
    i = 0
    print(os.path.join("../output", cfg.RUN_NAME))
    root_run_name = cfg.RUN_NAME
    while os.path.exists(os.path.join("../output", mode_name, cfg.RUN_NAME)):
        i += 1
        cfg.RUN_NAME = f"{root_run_name}_{i}"

    if cfg.OUTPUT.MODEL_DIR == '':
        cfg.OUTPUT.MODEL_DIR = os.path.join("../output", mode_name, cfg.RUN_NAME)
        # If the model directory doesn't exist, create it
        makedir(cfg.OUTPUT.MODEL_DIR)
    cfg.OUTPUT.IMG_DIR = os.path.join(cfg.OUTPUT.MODEL_DIR, "images")