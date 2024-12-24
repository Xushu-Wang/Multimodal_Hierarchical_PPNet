import os, json 
from typing import Optional, Dict, Tuple, List, Callable 
import torch 
from torch import Tensor
from torch.utils.data import Dataset 
import pandas as pd
import numpy as np
from skimage import io
from torchvision.transforms import transforms
from model.hierarchical_ppnet import Mode
from yacs.config import CfgNode
from .custom_transforms import GeneticOneHot, GeneticMutationTransform, create_transforms
import math 

class Split(): 
    def __init__(self, train: int, val: int, test: int): 
        assert(train >= 0 and val >= 0 and test >= 0) 
        self.train = train 
        self.val = val 
        self.test = test 
        self._sum = train + val + test

    def sum(self): 
        return self._sum  

    def __sub__(self, other: "Split"): 
        return Shortage(self.val - other.val, self.test - other.test) 

    def __repr__(self): 
        return f"Split(train={self.train}, val={self.val}, test={self.test})"

def scaled_split(split: Split, total: int) -> Split: 
    """Scales all split counts equally so that they sum approx to total."""
    ratio = (total - 3) / split.sum() 
    new_split = Split(
        math.ceil(ratio * split.train) + 1,
        math.ceil(ratio * split.val) + 1,
        math.ceil(ratio * split.test) + 1
    )

    while new_split.sum() > total: 
        i: int = np.random.choice(3) 
        match i: 
            case 0: new_split.train -= 1 
            case 1: new_split.val -= 1 
            case 2: new_split.test -= 1 
        new_split._sum -= 1 
    
    if new_split.train < 0 or new_split.val < 0 or new_split.test < 0: 
        raise Exception("We decremented to negative counts. ")

    return new_split 

class Shortage(): 
    def __init__(self, val: int = 0, test = 0): 
        self.val = val      # validation shortage
        self.test = test    # test shortage  

    def __iadd__(self, other: "Shortage"): 
        self.val += other.val 
        self.test += other.test 
        return self

    def __sub__(self, other: "Shortage"): 
        return Shortage(self.val - other.val, self.test - other.test)

    def sum(self) -> int: 
        return self.val + self.test 

    def shortage_exists(self): 
        return self.val > 0 and self.test > 0 

    def __repr__ (self): 
        return f"Shortage(val={self.val}, test={self.test})"

class TreeDataset(Dataset):
    """
    This is a heirarchichal dataset for genetics and images.
    """

    def __init__(
        self, 
        source_df:pd.DataFrame, 
        image_root_dir: str, 
        class_specification: Dict, 
        image_transforms: transforms.Compose, 
        genetic_transforms: Optional[GeneticMutationTransform], 
        mode: Mode, 
        flat_class: bool = False
        ):
        self.df = source_df
        self.image_root_dir = image_root_dir

        self.tree, self.levels = class_specification["tree"], class_specification["levels"]

        def generate_tree_indicies(tree:dict, idx=0):
            """
            This is deeply gross and I appologize for it.
            """
            tree = {k: v for k,v in tree.items()}
            tree["idx"] = idx

            idx = 0

            for k,v in tree.items():
                if k == "idx":
                    continue
                if isinstance(v, dict):
                    tree[k] = generate_tree_indicies(v, idx)
                else:
                    tree[k] = {
                        "idx": idx
                    }
                idx += 1

            return tree
        
        def generate_leaf_indicies(tree, idx):
            tree = {k: v for k,v in tree.items()}
            for k, v in tree.items():
                if v == None:
                    tree[k] = {"idx": idx["val"]}
                    idx["val"] += 1
                else:
                    tree[k] = generate_leaf_indicies(v, idx)[0]
            
            return tree, idx["val"]

        self.indexed_tree = generate_tree_indicies(self.tree)

        self.mode = mode
        self.one_hot_encoder = GeneticOneHot(length=720, zero_encode_unknown=True, include_height_channel=True)
        self.image_transforms = image_transforms
        self.genetic_transforms = genetic_transforms

        # If flat_class is true, the label will be an integer, with each species having a unique integer.
        self.flat_class = flat_class

        if flat_class:
            self.leaf_indicies, self.class_count = generate_leaf_indicies(self.tree, {"val": 0}) 

    def get_label(self, row:pd.Series) -> Optional[Tensor]:
        """
        Label is a tensor of indicies for each level of the tree.
        0 represents not_classified (or ignored, like in cases with only one class)
        """
        if self.flat_class:
            tree = self.leaf_indicies

            for i, level in enumerate(self.levels):
                if row[level] == "not_classified" or row[level] not in tree:
                    raise ValueError("Somehow you got not classified's up in here. That's not supported in this mode.")
                
                if i == len(self.levels)-1:
                    return torch.tensor(tree[row[level]]["idx"]).long()
                
                tree = tree[row[level]]
        else:
            tensor = torch.zeros(len(self.levels))
            tree = self.indexed_tree

            for i, level in enumerate(self.levels):
                if row[level] == "not_classified" or row[level] not in tree:
                    tensor[i:] = 0
                    break
                else:
                    tensor[i] = tree[row[level]]["idx"] + 1
                    tree = tree[row[level]]
            
            # Convert float tensor to int tensor
            tensor = tensor.long()

            return tensor

    def __getitem__(
        self, 
        idx: int) -> Tuple[Tuple[Optional[Tensor], Optional[np.ndarray]], Optional[Tensor]]:
        row = self.df.iloc[idx]
        image_path = os.path.join(
            self.image_root_dir, 
            row["order"], 
            row["family"], 
            row["genus"], 
            row["species"], 
            row["image_file"]
        ) 
        
        if self.mode.value & 1:
            genetics_string: str = row["nucraw"] 
            if self.genetic_transforms:
                genetics_string = self.genetic_transforms(genetics_string)
            genetics = self.one_hot_encoder(genetics_string)
        else:
            genetics = None
       
        image = self.image_transforms(io.imread(image_path)) \
          if self.mode.value >> 1 else None

        label = self.get_label(row)

        return (genetics, image), label

    def __len__(self):
        return len(self.df)

def oversample(
    source_df: pd.DataFrame, 
    count: int, 
    seed:int = 2024, 
    log: Callable = print
    ) -> pd.DataFrame:
    log("""Oh no, we oversampled. This will break data augmentation... 
        Must improve data augmentation.""")
    # Fairly oversample the source_df to count
    shortage = count - len(source_df)
    output = pd.concat(
        [source_df] + [source_df] * (shortage // len(source_df)) + 
        [source_df.sample(shortage % len(source_df), replace=False, random_state=seed)])

    return output.sample(frac=1, random_state=seed)

def balanced_sample(
    source_df:pd.DataFrame, 
    class_specification:dict, 
    count_per_leaf:Split, 
    train_not_classified_proportions: List[float],
    seed:int = 2024, 
    log = print
):
    """
    Returns a balanced sample of the source_df based on the class_specification and count_per_leaf.
    """
    source_df = source_df.sample(frac=1, random_state=seed)
    tree, levels = class_specification["tree"], class_specification["levels"]  

    # convert to dict
    train_not_classified_proportion = {l : p for l, p in zip(levels, train_not_classified_proportions)}

    def recursive_balanced_sample(
        tree:dict, 
        levels: List[str], 
        source_df:pd.DataFrame, 
        count_per_leaf:Split, 
        train_not_classified_proportion, 
        seed:int = 2024, 
        parent_name: Optional[str] = None, 
        log=print) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Shortage, dict[str, Tuple[int, int, int]]]:
        train_output = []
        val_output = []
        test_output = []
        count_tree = {}

        if len(levels) == 1:
            shortages = Shortage()
            for k, v in tree.items():
                if k == "not_classified":
                    continue
                class_size = len(source_df[source_df[levels[0]] == k])

                if class_size < 3:
                    raise ValueError(f"Less than 3 samples for {k} of {parent_name} at level {levels[0]}. Unable to proceed.")

                if class_size < (s := count_per_leaf.sum()):
                    log(f"Only {class_size} (of needed {s}) samples for {k} ({levels[0]}) of parent {parent_name}. Dividing proportionally")
                    
                    temp_count_per_leaf = scaled_split(count_per_leaf, class_size)

                    log(f"New counts for {k}")
                    log(f"Train:\t\t{temp_count_per_leaf.train}")
                    log(f"Validation:\t{temp_count_per_leaf.val}")
                    log(f"Test:\t\t{temp_count_per_leaf.test}")

                    shortages += count_per_leaf - temp_count_per_leaf
                else:
                    temp_count_per_leaf = count_per_leaf

                test_sample = source_df[source_df[levels[0]] == k].sample(temp_count_per_leaf.test, random_state=seed)
                validation_sample = source_df[source_df[levels[0]] == k].drop(index=list(test_sample.index)).sample(temp_count_per_leaf.val, random_state=seed)
                train_sample = pd.DataFrame(source_df[source_df[levels[0]] == k].drop(index=list(test_sample.index)).drop(index=list(validation_sample.index)).sample(temp_count_per_leaf.train, random_state=seed))

                if len(train_sample) < count_per_leaf.train:
                    train_sample = oversample(train_sample, count_per_leaf.train, seed)

                train_output.append(train_sample)
                val_output.append(validation_sample)
                test_output.append(test_sample)

                count_tree[k] = (len(train_sample), len(validation_sample), len(test_sample))            
        else:
            shortages = Shortage()

            for k, v in tree.items():
                if v is None:
                    not_classified = source_df[source_df[levels[0]] == k]

                    temp_count_per_leaf = count_per_leaf
                    child_shortages = Shortage()

                    if len(not_classified) < (s := count_per_leaf.sum()):
                        log(f"Only {len(not_classified)} (of needed {s}) samples for {k} ({levels[0]}) of parent {parent_name}. Dividing proportionally")
                        
                        temp_count_per_leaf = scaled_split(count_per_leaf, len(not_classified))

                        log(f"New counts for {k}")
                        log(f"Train:\t\t{temp_count_per_leaf.train}")
                        log(f"Validation:\t{temp_count_per_leaf.val}")
                        log(f"Test:\t\t{temp_count_per_leaf.test}")

                        child_shortages += count_per_leaf - temp_count_per_leaf

                    child_test = not_classified.sample(temp_count_per_leaf.test, random_state=seed)
                    child_val = not_classified.drop(index=list(child_test.index)).sample(temp_count_per_leaf.val, random_state=seed)
                    child_train = not_classified.drop(index=list(child_test.index)).drop(index=list(child_val.index)).sample(temp_count_per_leaf.train, random_state=seed)
                    child_count_tree = {"not_classified": (len(child_train), len(child_val), len(child_test))}
                else:
                    child_train, child_val, child_test, child_shortages, child_count_tree = recursive_balanced_sample(
                        v,
                        levels[1:],
                        pd.DataFrame(source_df[source_df[levels[0]] == k]),
                        count_per_leaf,
                        train_not_classified_proportion,
                        seed,
                        k,
                    )
                shortages += child_shortages

                train_output.append(child_train)
                val_output.append(child_val)
                test_output.append(child_test)

                count_tree[k] = child_count_tree

        # Handle not_classified
        not_classified = source_df[(source_df[levels[0]] == "not_classified") | (~source_df[levels[0]].isin([c for c in tree.keys()]))]
        train_not_classified_count = train_not_classified_proportion[levels[0]] * (len(tree.keys()) - 1) * count_per_leaf.train
        train_not_classified_count = int(train_not_classified_count)

        if len(not_classified) < shortages.sum() + train_not_classified_count:
            log(f"Unable to counterbalance with not_classified for {parent_name} at {levels[0]}. Sorry.")
            # TODO - This could be handled by going up one level and adding more not classified.
            not_classified_sample_amounts = Shortage()
            s = scaled_split(count_per_leaf, len(not_classified))
            train_not_classified_count = s.train
            not_classified_sample_amounts.val = s.val
            not_classified_sample_amounts.test = s.test
        else:
            not_classified_sample_amounts = shortages

        test_sample = not_classified.sample(not_classified_sample_amounts.test, random_state=seed)
        validation_sample = not_classified.drop(index=list(test_sample.index)).sample(not_classified_sample_amounts.val, random_state=seed)

        train_sample = not_classified.drop(index=list(test_sample.index)).drop(index=list(validation_sample.index)).sample(train_not_classified_count, random_state=seed)

        train_output.append(train_sample)
        val_output.append(validation_sample)
        test_output.append(test_sample)

        count_tree["not_classified"] = (len(train_sample), len(validation_sample), len(test_sample))

        return pd.concat(train_output), pd.concat(val_output), pd.concat(test_output), shortages - not_classified_sample_amounts, count_tree

    train_df, val_df, test_df, shortages, count_tree = recursive_balanced_sample(tree, levels, source_df, count_per_leaf, train_not_classified_proportion, seed)

    if shortages.shortage_exists(): 
        raise ValueError(f"Unable to balance dataset. Shortages: {shortages}. This should not happen, I am very confused.")

    log("---- Overall Results ----")
    log(f"Train:\t\t{len(train_df)}")
    log(f"Validation:\t{len(val_df)}")
    log(f"Test:\t\t{len(test_df)}")

    return train_df, val_df, test_df, count_tree

def create_datasets(
    source: str,                        # path to dataset tsv file
    image_root_dir: str,
    gen_aug_params:CfgNode,
    train_val_test_split:Split,
    oversampling_rate:int,
    train_not_classified_proportions: List[float],
    tree_specification_file:str,
    cached_dataset_folder:str,
    cached_dataset_root:str,
    run_name:str,
    transform_mean:tuple,
    transform_std:tuple,
    mode:Mode,
    seed: int = 2024,
    flat_class: bool = False,
    log: Callable = print
) -> Tuple[TreeDataset, TreeDataset, TreeDataset, TreeDataset, transforms.Normalize]:
    """
    Creates train, train_push, validation, and test dataloaders for the tree dataset.
    
    source_file - tsv that contains all needed data (ex. metadata_cleaned_permissive.tsv). Note: all entires in source_file must have images in image_root
    image_root_dir - root directory for the images.
    gen_aug_params - object genetic augmentation parameters to apply to the genetic data. (found in cfg.py)
    image_augmentations - list of image augmentations to apply to the image data.
    train_val_test_split - 3-tuple of integers representing the number of true train, validation, and test samples for each leaf node of the tree. In most cases, it's the desired # of samples per species. NOTE: This does not include oversampling of train samples.
    train_end_count - The number of train_end_countsamples to end with in the training set.
    train_not_classified_proportion - An object specifying the porportion of samples at each level that should be not classified.
    tree_specification_file - path to json file tree of valid classes.
    mode - Mode enumerate object
    seed - random seed for splitting, shuffling, transformations, etc.
    oversampling_rate - how much we argument the train dataloader, but for images is deprecated since we always agument on the fly. Should always be 1 (and removed later). On genetics, this is not online. Stored in the dataframe and in the cached dataset folder. 
    """
    np.random.seed(seed) 

    # load the tree 
    class_specification = json.load(open(tree_specification_file, "r"))

    if cached_dataset_folder.strip():
        train_df = pd.read_csv(os.path.join(cached_dataset_folder, "train.tsv"), sep="\t")
        train_push_df = pd.read_csv(os.path.join(cached_dataset_folder, "train_push.tsv"), sep="\t")
        val_df = pd.read_csv(os.path.join(cached_dataset_folder, "val.tsv"), sep="\t")
        test_df = pd.read_csv(os.path.join(cached_dataset_folder, "test.tsv"), sep="\t")
    else:
        # generate the dataframes
        train_df, val_df, test_df, _ = balanced_sample(
            pd.read_csv(source, sep="\t"),
            class_specification, 
            train_val_test_split, 
            train_not_classified_proportions, 
            seed
        )
        
        # save the train_push dataframe
        train_push_df = train_df.copy()

        # oversample from train dataframe if needed
        old_train_size = len(train_df)
        if oversampling_rate != 1:
            train_df = oversample(train_df, len(train_df) * oversampling_rate, seed)

        log(f"Oversampled train from {old_train_size:,} samples to {len(train_df):,} samples")

        # Save all three datasets
        output_dir = os.path.join(cached_dataset_root, run_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        train_df.to_csv(train_path := os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
        log(f"Saved train dataset to {train_path}")

        train_push_df.to_csv(train_push_path := os.path.join(output_dir, "train_push.tsv"), sep="\t", index=False)
        log(f"Saved train_push dataset to {train_push_path}")

        val_df.to_csv(val_path := os.path.join(output_dir, "val.tsv"), sep="\t", index=False)
        log(f"Saved validation dataset to {val_path}")

        test_df.to_csv(test_path := os.path.join(output_dir, "test.tsv"), sep="\t", index=False)
        log(f"Saved test dataset to {test_path}")

    # generate the transforms for image and genetics datasets
    augmented_img_transforms, \
    augmented_genetic_transforms, \
    push_img_transforms, \
    img_transforms, \
    normalize = create_transforms(transform_mean, transform_std, gen_aug_params)
    
    # create the datasets
    train_dataset = TreeDataset(
        source_df=train_df,
        # image_cache_dir if oversampling_rate != 1 else image_root_dir,
        image_root_dir=image_root_dir,
        class_specification=class_specification,
        image_transforms=augmented_img_transforms,
        genetic_transforms=augmented_genetic_transforms,
        mode=mode,
        flat_class=flat_class
    )
    train_push_dataset = TreeDataset(
        source_df=train_push_df,
        image_root_dir=image_root_dir,
        class_specification=class_specification,
        image_transforms=push_img_transforms,
        genetic_transforms=None,
        mode=mode,
        flat_class=flat_class
    )

    val_dataset = TreeDataset(
        source_df=val_df,
        image_root_dir=image_root_dir,
        class_specification=class_specification,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
        flat_class=flat_class
    )
    test_dataset = TreeDataset(
        source_df=test_df,
        image_root_dir=image_root_dir,
        class_specification=class_specification,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
        flat_class=flat_class
    )

    return train_dataset, train_push_dataset, val_dataset, test_dataset, normalize

def get_datasets(cfg: CfgNode, log: Callable, flat_class=False
) -> Tuple[TreeDataset, TreeDataset, TreeDataset, TreeDataset, transforms.Normalize]:
    log("Getting Datasets")

    return create_datasets(
        source=cfg.DATASET.DATA_FILE,
        image_root_dir=cfg.DATASET.IMAGE_PATH,
        gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
        train_val_test_split = Split(*cfg.DATASET.TRAIN_VAL_TEST_SPLIT),
        oversampling_rate=cfg.DATASET.OVERSAMPLING_RATE,
        train_not_classified_proportions=cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS,
        tree_specification_file=cfg.DATASET.TREE_SPECIFICATION_FILE,
        cached_dataset_folder=cfg.DATASET.CACHED_DATASET_FOLDER,
        cached_dataset_root=cfg.DATASET.CACHED_DATASET_ROOT,
        run_name=cfg.RUN_NAME,
        transform_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
        transform_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
        mode=Mode(cfg.DATASET.MODE),
        seed=cfg.SEED,
        log=log,
        flat_class=flat_class
    )

