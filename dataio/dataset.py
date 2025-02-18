import os, json 
from typing import Optional, Dict, Tuple, List, Callable 
import torch 
from torch import Tensor
from torch.utils.data import Dataset 
import pandas as pd
import numpy as np
from skimage import io
from torchvision.transforms import transforms
from model.model import Mode
from yacs.config import CfgNode
from .custom_transforms import GeneticOneHot, GeneticMutationTransform, create_transforms
import math 

class TaxNode(): 
    def __init__(self, taxonomy, idx: List): 
        self.taxonomy = taxonomy 
        self.children = set()
        self.idx = idx

    def __repr__(self): 
        return f"TaxNode({self.taxonomy}, {self.idx})"

class Hierarchy(): 
    """
    Stores the hierarchy of a dataset that we will work with. 
    Attributes: 
        tree   - the actual tree with the different classes
        levels - the metadata describing each level, e.g. ["order", "family", "genus"]
    """

    def __init__(self, json_file: str): 
        """
        All json files should be created by build_class_tree.py. 
        Simply input in the path to the json file to instantiate an Hierarchy object. 
        """
        meta = json.load(open(json_file, "r")) 
        self.levels = meta["levels"] 
        self.tree_dict = meta["tree"]
        self.root = self.dict_to_tree(TaxNode("Insect", []), self.tree_dict)

    def dict_to_tree(self, node, d): 
        """
        Convert dictionary loaded from json file to actual tree of TaxNodes. 
        Traversed using DFS. 
        """
        for i, (k, v) in enumerate(d.items()): 
            if isinstance(v, dict): 
                child = self.dict_to_tree(TaxNode(k, node.idx + [i]), v) 
                node.children.add(child)

        return node

class Split():  
    """
    A class that holds train/val/test splits 
    """
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

    def scale(self, total: int) -> "Split": 
        """Scales all split counts equally so that they sum approx to total."""
        ratio = total / self.sum()
        scaled_split = Split(
            math.floor(ratio * self.train), 
            math.floor(ratio * self.val), 
            math.floor(ratio * self.test)
        )
        scaled_split.train += (total - scaled_split.sum())
        return scaled_split 

class Shortage(): 
    """
    Class to hold the validation and test shortage inputs.  
    """
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
    Hierarchical dataset for genetics and images. 
    """

    def __init__(
        self, 
        source_df:pd.DataFrame, 
        image_root_dir: str, 
        hierarchy: Dict, 
        image_transforms: transforms.Compose, 
        genetic_transforms: Optional[GeneticMutationTransform], 
        mode: Mode
    ):
        self.df = source_df
        self.image_root_dir = image_root_dir

        self.tree, self.levels = hierarchy["tree"], hierarchy["levels"]

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
        
        def generate_leaf_indicies(tree, idx, level=0, prior_vals=[]):
            tree = {k: v for k,v in tree.items()}
            for k, v in tree.items():
                if v == None:
                    tree[k] = {"idx": idx[level]}
                else:
                    min_idx = idx[3]
                    tree[k] = generate_leaf_indicies(v, idx, level + 1, prior_vals + [idx[level]])[0]
                    tree[k]["idx"] = idx[level]
                    max_idx = idx[3] - 1

                    self.level_species_map[level][idx[level]] = (min_idx, max_idx)
                idx[level] += 1
            
            return tree, idx

        self.indexed_tree = generate_tree_indicies(self.tree)

        self.mode = mode
        self.one_hot_encoder = GeneticOneHot(length=720, zero_encode_unknown=True, include_height_channel=True)
        self.image_transforms = image_transforms
        self.genetic_transforms = genetic_transforms

        self.level_species_map = [{}, {}, {}]
        self.leaf_indicies, idx = generate_leaf_indicies(self.tree, [0,0,0,0]) 
        self.class_count = idx[3]

    def get_species_mask(self, level_indicies:Tensor, level:int) -> Tensor:
        """
        Pass an nx1 tensor of level_indicies (and the corresponding level).

        This returns an nx(num_species) mask of the species indicies corresponding to the level.
        """
        if level == 3:
            mins = level_indicies
            maxs = level_indicies
        else:
            tuples = [self.level_species_map[level][i.item()] for i in level_indicies]
            mins = torch.tensor([t[0] for t in tuples]).long()
            maxs = torch.tensor([t[1] for t in tuples]).long()

        mask = torch.zeros(len(level_indicies), self.class_count).bool()

        for i, (min_, max_) in enumerate(zip(mins, maxs)):
            mask[i, min_:max_+1] = True

        return mask

    def get_label_flat(self, row:pd.Series) -> Optional[Tensor]:
        tree = self.leaf_indicies
        out = [0,0,0,0]

        for i, level in enumerate(self.levels):
            if row[level] == "not_classified" or row[level] not in tree:
                raise ValueError("Somehow you got not classified's up in here. That's not supported in this mode.")
            
            out[i] = tree[row[level]]["idx"]
            if i == len(self.levels) - 1:
                return torch.tensor(out).long()
            
            tree = tree[row[level]]

    def get_label(self, row:pd.Series) -> Optional[Tensor]:
        """
        Label is a tensor of indicies for each level of the tree.
        0 represents not_classified (or ignored, like in cases with only one class)
        """
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
        idx: int
    ) -> Tuple[Tuple[Optional[Tensor], Optional[np.ndarray]], Optional[Tensor]]:
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
          if self.mode.value & 2 else None

        flat_label, label = self.get_label_flat(row), self.get_label(row)

        return (genetics, image), (label,flat_label)

    def __len__(self):
        return len(self.df)

def balanced_sample(
    source_df: pd.DataFrame, 
    hierarchy: dict, 
    count_per_leaf: Split, 
    train_not_classified_proportions: List[float],
    seed: int = 2024, 
    log = print
):
    """
    Returns a balanced sample of the source_df based on the class_specification and count_per_leaf.
    """
    source_df = source_df.sample(frac=1, random_state=seed)
    tree, levels = hierarchy["tree"], hierarchy["levels"]  

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
        log=print
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Shortage, dict[str, Tuple[int, int, int]]]:
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
                    
                    temp_count_per_leaf = count_per_leaf.scale(class_size)

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
                    raise ValueError("Please provide a dataset with enough samples to satisfy the train,val,test proportions.")

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
                        
                        temp_count_per_leaf = count_per_leaf.scale(len(not_classified))

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
            s = count_per_leaf.split(len(not_classified))
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

def create_new_ds(
    source: str,                        # path to dataset tsv file
    image_root_dir: str,
    gen_aug_params:CfgNode,
    train_val_test_split:Split,
    train_not_classified_proportions: List[float],
    hierarchy: Dict, 
    run_name:str,
    transform_mean:tuple,
    transform_std:tuple,
    mode:Mode,
    seed: int = 2024,
    log: Callable = print
) -> Tuple[TreeDataset, TreeDataset, TreeDataset, TreeDataset, transforms.Normalize]:
    """
    Creates train, train_push, validation, and test dataloaders for the tree dataset.
    
    source_file - tsv that contains all needed data (ex. metadata_cleaned_permissive.tsv). Note: all entires in source_file must have images in image_root
    image_root_dir - root directory for the images.
    gen_aug_params - object genetic augmentation parameters to apply to the genetic data. (found in cfg.py)
    image_augmentations - list of image augmentations to apply to the image data.
    train_val_test_split - 3-tuple of integers representing the number of true train, validation, and test samples for each leaf node of the tree. In most cases, it's the desired # of samples per species.
    train_end_count - The number of train_end_countsamples to end with in the training set.
    train_not_classified_proportion - An object specifying the porportion of samples at each level that should be not classified.
    tree_specification_file - path to json file tree of valid classes.
    mode - Mode enumerate object
    seed - random seed for splitting, shuffling, transformations, etc.
    """
    np.random.seed(seed) 

    train_df, val_df, test_df, _ = balanced_sample(
        pd.read_csv(source, sep="\t"),
        hierarchy, 
        train_val_test_split, 
        train_not_classified_proportions, 
        seed
    )
    
    # save the train_push dataframe
    train_push_df = train_df.copy()

    # Save all three datasets
    output_dir = os.path.join("pre_existing_datasets", run_name)
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
        image_root_dir=image_root_dir,
        class_specification=hierarchy,
        image_transforms=augmented_img_transforms,
        genetic_transforms=augmented_genetic_transforms,
        mode=mode,
    )
    train_push_dataset = TreeDataset(
        source_df=train_push_df,
        image_root_dir=image_root_dir,
        class_specification=hierarchy,
        image_transforms=push_img_transforms,
        genetic_transforms=None,
        mode=mode,
    )

    val_dataset = TreeDataset(
        source_df=val_df,
        image_root_dir=image_root_dir,
        class_specification=hierarchy,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
    )
    test_dataset = TreeDataset(
        source_df=test_df,
        image_root_dir=image_root_dir,
        class_specification=hierarchy,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
    )

    return train_dataset, train_push_dataset, val_dataset, test_dataset, normalize

def retrieve_cached_ds(
    image_root_dir: str,
    gen_aug_params:CfgNode,
    hierarchy: Dict, 
    cache_ds_dir:str,
    transform_mean:tuple,
    transform_std:tuple,
    mode:Mode,
    log: Callable = print
):

    train_df = pd.read_csv(os.path.join(cache_ds_dir, "train.tsv"), sep="\t")
    train_push_df = pd.read_csv(os.path.join(cache_ds_dir, "train_push.tsv"), sep="\t")
    val_df = pd.read_csv(os.path.join(cache_ds_dir, "val.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(cache_ds_dir, "test.tsv"), sep="\t")

    augmented_img_transforms, \
    augmented_genetic_transforms, \
    push_img_transforms, \
    img_transforms, \
    normalize = create_transforms(transform_mean, transform_std, gen_aug_params)
    
    # create the datasets
    train_dataset = TreeDataset(
        source_df=train_df,
        image_root_dir=image_root_dir,
        hierarchy=hierarchy,
        image_transforms=augmented_img_transforms,
        genetic_transforms=augmented_genetic_transforms,
        mode=mode,
    )
    train_push_dataset = TreeDataset(
        source_df=train_push_df,
        image_root_dir=image_root_dir,
        hierarchy=hierarchy,
        image_transforms=push_img_transforms,
        genetic_transforms=None,
        mode=mode,
    )

    val_dataset = TreeDataset(
        source_df=val_df,
        image_root_dir=image_root_dir,
        hierarchy=hierarchy,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
    )
    test_dataset = TreeDataset(
        source_df=test_df,
        image_root_dir=image_root_dir,
        hierarchy=hierarchy,
        image_transforms=img_transforms,
        genetic_transforms=None,
        mode=mode,
    )

    log("Datasets created")

    return train_dataset, train_push_dataset, val_dataset, test_dataset, normalize


def get_datasets(
    cfg: CfgNode, 
    log: Callable = print
) -> Tuple[TreeDataset, TreeDataset, TreeDataset, TreeDataset, transforms.Normalize]:
    log("Getting Datasets") 
    hierarchy = json.load(open(cfg.DATASET.TREE_SPECIFICATION_FILE, "r"))

    if cfg.DATASET.CACHED_DATASET_FOLDER.strip(): 
        # if we want to retrieve a cached dataset
        return retrieve_cached_ds(
            image_root_dir=cfg.DATASET.IMAGE_PATH,
            gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
            hierarchy = hierarchy, 
            cache_ds_dir=cfg.DATASET.CACHED_DATASET_FOLDER,
            transform_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            transform_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
            mode=Mode(cfg.DATASET.MODE),
            log=log
        )
    else: 
        # if we want to create a new dataset 
        return create_new_ds(
            source=cfg.DATASET.DATA_FILE,
            image_root_dir=cfg.DATASET.IMAGE_PATH,
            gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
            train_val_test_split = Split(*cfg.DATASET.TRAIN_VAL_TEST_SPLIT),
            train_not_classified_proportions=cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS,
            hierarchy = hierarchy, 
            run_name=cfg.RUN_NAME,
            transform_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            transform_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
            mode=Mode(cfg.DATASET.MODE),
            seed=cfg.SEED,
            log=log,
        )

