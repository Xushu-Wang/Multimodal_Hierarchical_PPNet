from yacs.config import CfgNode as CN

_C = CN()

_C.RUN_NAME = "" # Name of the run. If "", will be set to the current time.
_C.SEED = 2024
_C.WANDB_MODE = "online"

# Model
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda" 
_C.MODEL.IMAGE_BACKBONE = 'resnetbioscan'
_C.MODEL.GENETIC_BACKBONE_PATH = "NA"
_C.MODEL.PRUNE = False
_C.MODEL.PRUNING_TYPE = "quality"
_C.MODEL.PRUNING_K = 6
_C.MODEL.PRUNING_TAU = 3

_C.MODEL.MULTI = CN()
_C.MODEL.MULTI.MULTI_PPNET_PATH = "NA"

# _C.MODEL.PROTOTYPE_DISTANCE_FUNCTION = 'cosine'
# _C.MODEL.PROTOTYPE_ACTIVATION_FUNCTION = 'linear'
_C.MODEL.GENETIC_MODE = False

# Dataset
_C.DATASET = CN()
_C.DATASET.DATA_FILE = "../datasets/source_files/metadata_cleaned_permissive.tsv" # Path to CSV from which data can be selected
_C.DATASET.IMAGE_PATH = "../datasets/full_bioscan_images/" # Path to image directory
_C.DATASET.TREE_SPECIFICATION_FILE = "NA" # Path to JSON file that specifies tree structure
_C.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS = [0,0,.25,.5] # Proportions of samples at each level that are unclassified [order, family, genus, species]. Note: Lower levels counts do not consider higher level counts, so for this default, > 50% of species are unclassified (50% + 25% of genus)
_C.DATASET.FLAT_CLASS = True # If true, the dataset will be flattened to only include species. If false, the dataset will include all levels.

_C.DATASET.MODE = 3 # 0 is illegal, don't use. 1 is genetic only, 2 is image only, 3 is joint. This will only affect what the dataloader/dataset returns. Not the augmentation.
_C.DATASET.PARALLEL_MODE = False # If true, the image and genetic models will train separately with a shared correspondence loss. If false, the models will train jointly. Only relevant if MODE = 3.

_C.DATASET.CACHED_DATASET_FOLDER = "" # Path to folder with pre-existing datasets. Most other dataset parameters will be ignored if this is set. "" for no cache.  

_C.DATASET.TRAIN_VAL_TEST_SPLIT = (120, 40, 40) # For each leaf node, the number of samples in the training, validation, and test sets.

_C.DATASET.PREEXISTING = False # Whether the dataset has already preprocessed (augmentaiton has occured)

_C.DATASET.GENETIC_AUGMENTATION = CN()
_C.DATASET.GENETIC_AUGMENTATION.SUBSTITUTION_RATE = 0.05 # Probability of substitution for each base pair
_C.DATASET.GENETIC_AUGMENTATION.DELETION_COUNT = 4 # Maximum number of deletions to perform
_C.DATASET.GENETIC_AUGMENTATION.INSERTION_COUNT = 4 # Maximum number of insertions to perform

_C.DATASET.TRAIN_BATCH_SIZE = 60
_C.DATASET.TEST_BATCH_SIZE = 40
_C.DATASET.TRAIN_PUSH_BATCH_SIZE = 40

# Image Dataset
_C.DATASET.IMAGE = CN()

_C.DATASET.IMAGE.SIZE = 256
_C.DATASET.IMAGE.PROTOTYPE_SHAPE = (0,0,0)
_C.DATASET.IMAGE.NUM_PROTOTYPES_PER_CLASS = 8
_C.DATASET.IMAGE.NUM_PROTOTYPE = 8
_C.DATASET.IMAGE.PPNET_PATH = "NA"

_C.DATASET.IMAGE.TRAIN_BATCH_SIZE = 0
_C.DATASET.IMAGE.TRANSFORM_MEAN = ()
_C.DATASET.IMAGE.TRANSFORM_STD = ()

# Genetic Dataset 
_C.DATASET.GENETIC = CN()
_C.DATASET.GENETIC.PROTOTYPE_SHAPE = (0, 0, 0)
_C.DATASET.GENETIC.FIX_PROTOTYPES = True
_C.DATASET.GENETIC.NUM_PROTOTYPES_PER_CLASS = 40
_C.DATASET.GENETIC.MAX_NUM_PROTOTYPES_PER_CLASS = 8
_C.DATASET.GENETIC.PPNET_PATH = "NA"

# OPTIMIZER
_C.OPTIM = CN()

# Epochs Data 
_C.OPTIM.NUM_WARM_EPOCHS = 1
_C.OPTIM.NUM_TRAIN_EPOCHS = 30
_C.OPTIM.PUSH_EPOCHS = []

# Coefficients for Loss Term
_C.OPTIM.COEFS = CN()
_C.OPTIM.COEFS.GENETIC = CN()
_C.OPTIM.COEFS.GENETIC.CRS_ENT = 0.
_C.OPTIM.COEFS.GENETIC.CLST = 0.
_C.OPTIM.COEFS.GENETIC.SEP = 0.
_C.OPTIM.COEFS.GENETIC.L1 = 0.
_C.OPTIM.COEFS.GENETIC.ORTHOGONALITY = 0.

_C.OPTIM.COEFS.IMAGE = CN()
_C.OPTIM.COEFS.IMAGE.CRS_ENT = 0.
_C.OPTIM.COEFS.IMAGE.CLST = 0.
_C.OPTIM.COEFS.IMAGE.SEP = 0.
_C.OPTIM.COEFS.IMAGE.L1 = 0.
_C.OPTIM.COEFS.IMAGE.ORTHOGONALITY = 0.

_C.OPTIM.COEFS.CORRESPONDENCE = -0.

# Warm optimizer
_C.OPTIM.WARM = CN()
_C.OPTIM.WARM.ADD_ON_LAYERS_LR = -1.
_C.OPTIM.WARM.ADD_ON_LAYERS_WD = -1.
_C.OPTIM.WARM.PROTOTYPE_LR = -1.
_C.OPTIM.WARM.PROTOTYPE_WD = -1.

# Joint optimizer
_C.OPTIM.JOINT = CN() 
_C.OPTIM.JOINT.FEATURES_LR = -1.
_C.OPTIM.JOINT.FEATURES_WD = -1.
_C.OPTIM.JOINT.ADD_ON_LAYERS_LR = -1.
_C.OPTIM.JOINT.ADD_ON_LAYERS_WD = -1.
_C.OPTIM.JOINT.LAST_LAYER_LR = -1.
_C.OPTIM.JOINT.LAST_LAYER_WD = -1.
_C.OPTIM.JOINT.PROTOTYPE_LR = -1.
_C.OPTIM.JOINT.PROTOTYPE_WD = -1.

# Last layer optimizer
_C.OPTIM.LAST_LAYER= CN()
_C.OPTIM.LAST_LAYER.LAST_LAYER_LR = -1.
_C.OPTIM.LAST_LAYER.LAST_LAYER_MOM = -1.


# Deprecated (?)
# If true, the crossentropy term will be applied to a large vector corresponding to conditional probabilities of each species (or higher level if not classified).
# Otherwise, the crossentropy term will be applied to each classification task and summed.
_C.OPTIM.GLOBAL_CROSSENTROPY = True
_C.OPTIM.CORRESPONDENCE_TYPE = "Batched" # One of ("Batched", "Single"). Batched evaluates the correspondence accross an entire batch.
_C.OPTIM.PRUNE_START = 10

# Output 
_C.OUTPUT = CN()
_C.OUTPUT.MODEL_DIR = ""
_C.OUTPUT.WEIGHT_MATRIX_FILENAME = "NA" 
_C.OUTPUT.PROTOTYPE_IMG_FILENAME_PREFIX = "NA" 
_C.OUTPUT.PROTOTYPE_SELF_ACT_FILENAME_PREFIX = "NA" 
_C.OUTPUT.PROTO_BOUND_BOXES_FILENAME_PREFIX = "NA" 
_C.OUTPUT.SAVE = False
_C.OUTPUT.PREPROCESS_INPUT_FUNCTION = None

def get_cfg_defaults(): 
    return _C.clone()
