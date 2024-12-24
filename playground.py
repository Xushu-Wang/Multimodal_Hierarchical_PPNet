from enum import Enum
from yacs.config import CfgNode as CN

class Mode(Enum): 
    GENETIC = 1
    IMAGE = 2 
    MULTIMODAL = 3 

C = CN()

C.RUN_NAME = "" 
C.SEED = 202

print(dir(type(C)))
