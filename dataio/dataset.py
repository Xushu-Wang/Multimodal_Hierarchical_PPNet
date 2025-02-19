import os, json, math
from typing import Optional, List, Callable 
import torch 
from torch.utils.data import Dataset 
import pandas as pd
import numpy as np
from skimage import io
from enum import Enum
from yacs.config import CfgNode
from .custom_transforms import GeneticOneHot, create_transforms

class Mode(Enum):  
    '''
    Enumeration object for labeling ppnet mode.  
    '''
    GENETIC = 1
    IMAGE = 2 
    MULTIMODAL = 3 

class TaxNode(): 
    """
    Node class that store taxonomy. 
    Attributes: 
        taxonomy - the name of the order/family/genus/species 
        childs - pointers to TaxNodes in the subclass 
        idx      - Tensor([a, b, c, d]), d-th class of c-th class of b-th class ...
        flat_idx - Tensor([w, x, y, z]), ordered using indices in each depth
        depth    - int within [0, 1, 2, 3, 4] 
        min_species_idx - 
    """
    def __init__(self, taxonomy, idx, flat_idx): 
        self.taxonomy = taxonomy 
        self.childs = dict()
        self.idx = idx
        self.flat_idx = flat_idx
        self.depth = len(idx)
        self.min_species_idx = -1

    def __repr__(self): 
        return f"TaxNode({self.taxonomy}, {self.idx}, {self.flat_idx}, {self.min_species_idx})"

class Level(): 
    """
    Class that stores the counts of each level of a taxonomy. 
    """

    def __init__(self, *args): 
        self.counts = [0] * len(args) 
        self.names = {name : depth for depth, name in enumerate(args)}

    def count(self, level): 
        if isinstance(level, int): 
            return self.counts[level] 
        elif isinstance(level, str): 
            return self.counts[self.names[level]] 
        else: 
            raise ValueError("Cannot retrieve count.") 

    def increment(self, level): 
        if isinstance(level, int): 
            self.counts[level] += 1 
        elif isinstance(level, str): 
            self.counts[self.names[level]] += 1
        else: 
            raise ValueError("Cannot retrieve count for increment.")  

    def __repr__(self): 
        out = "Level\n" 
        for name, depth in self.names.items(): 
            out += f"  {name} [{depth}]\t: {self.counts[depth]}\n" 
        return out 

    def __iter__(self): 
        return iter(self.names)

    def __len__(self): 
        return len(self.counts) 

    def __eq__(self, other): 
        return self.counts == other.counts and self.names == other.names

class Hierarchy(): 
    """
    A Trie Data structure: Stores the hierarchy of a dataset that we will work with. 
    Attributes: 
        tree   - the actual tree with the different classes
        levels - the metadata describing each level, e.g. ["order", "family", "genus", "species"]
    """

    def __init__(self, json_file: str): 
        """
        All json files should be created by build_class_tree.py. 
        Simply input in the path to the json file to instantiate an Hierarchy object. 
        """
        meta = json.load(open(json_file, "r")) 
        self.levels = Level(*meta["levels"])
        self.tree_dict = meta["tree"]

        self.root = self._dict_to_trie(TaxNode("Insect", [], []), self.tree_dict)

    def _dict_to_trie(self, node: TaxNode, d) -> TaxNode: 
        """
        Convert dictionary loaded from json file to actual tree of TaxNodes. Traversed using DFS. 
        flat_idx another form of indexing, where every node of a certain depth D
        is enumerated from 1...N_D
        """ 

        for i, (k, v) in enumerate(d.items(), 1): 
            if isinstance(v, dict): 
                child = self._dict_to_trie(TaxNode(
                    k, 
                    node.idx + [i], 
                    node.flat_idx + [self.levels.count(node.depth)]
                ), v) 
                if i == 1: 
                    node.min_species_idx = child.min_species_idx
            else: 
                child = TaxNode(
                    k, 
                    torch.tensor(node.idx + [i]).long(), 
                    torch.tensor(node.flat_idx + [self.levels.count(node.depth)]).long()
                ) 
                if i == 1: 
                    node.min_species_idx = int(child.flat_idx[-1]) 

            self.levels.increment(node.depth)

            node.childs[k] = child 
        node.idx = torch.tensor(node.idx).long()
        node.flat_idx = torch.tensor(node.flat_idx).long()
        return node 

    def traverse(self, path: List[str]): 
        """
        Given a path of taxonomies, does this exist in the tree? 
        Bool = True if we traversed through whole path. 
        """
        node = self.root 
        for next in path: 
            if next not in node.childs: 
                return False, node
            node = node.childs[next] 
        return True, node

    def __repr__(self): 
        """
        Hacky way to print tree for debugging. Hard-coded for 4 levels. 
        """
        out = self.root.__repr__() + "\n"
        for _, c1 in self.root.childs.items(): 
            out += f"  {c1.__repr__()}\n"
            for _, c2 in c1.childs.items(): 
                out += f"    {c2.__repr__()}\n"
                for _, c3 in c2.childs.items(): 
                    out += f"      {c3.__repr__()}\n"
                    for _, c4 in c3.childs.items(): 
                        out += f"      {c4.__repr__()}\n"

        return out

    def __eq__(self, other): 
        return self.levels == other.levels and self.tree_dict == other.tree_dict 

class TreeDataset(Dataset): 
    """
    Improved dataset with more readable code. 
    """
    def __init__(
        self, 
        hierarchy: Hierarchy, 
        df:pd.DataFrame, 
        mode: Mode, 
        img_trans, 
        gen_trans
    ): 
        self.hierarchy = hierarchy 
        self.mode = mode 
        self.df = df
        self.one_hot_encoder = GeneticOneHot(720, True, True)
        self.img_trans = img_trans
        self.gen_trans = gen_trans

        # check that there are no missing values for the levels 
        for level in hierarchy.levels: 
            assert not ((df[level] == "not_classified").any())

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx): 
        row = self.df.iloc[idx] 
        path = [row.order, row.family, row.genus, row.species]

        # retrive genetics from dataframe 
        genetics = image = None 
        if self.mode.value & 1:
            gen_str = row["nucraw"] 
            gen_str = self.gen_trans(gen_str) if self.gen_trans else gen_str
            genetics = self.one_hot_encoder(gen_str)

        # retrive image from dataframe and filepath
        if self.mode.value & 2: 
            image_path = os.path.join("..", "datasets", "full_bioscan_images",
                row["order"], row["family"], row["genus"], row["species"], 
                row["image_file"]
            ) 
            image = io.imread(image_path)
            image = self.img_trans(image) if self.img_trans else image 

        # get label index from hierarchy
        found, node = self.hierarchy.traverse(path) 
        if found: 
            label = node.idx
        else: 
            padding = [0] * (4 - len(node.idx))
            label = node.idx + padding

        return (genetics, image), (label, node.flat_idx)

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

def balanced_sample(
    hierarchy: Hierarchy, 
    source_df: pd.DataFrame, 
    count_per_leaf: Split, 
    train_not_classified_proportions: List[float],
    seed: int = 2024, 
    log = print
):
    """
    Returns a balanced sample of the source_df based on the class_specification and count_per_leaf.
    """
    source_df = source_df.sample(frac=1, random_state=seed)
    tree = hierarchy.tree_dict 
    levels = list(hierarchy.levels.names.keys())

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
    ):
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
            s = count_per_leaf.scale(len(not_classified))
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

    train, val, test, shortages, count_tree = recursive_balanced_sample(tree, levels, source_df, count_per_leaf, train_not_classified_proportion, seed)

    if shortages.shortage_exists(): 
        raise ValueError(f"Unable to balance dataset. Shortages: {shortages}.")

    log("---- Overall Results ----")
    log(f"Train:\t\t{len(train)}")
    log(f"Validation:\t{len(val)}")
    log(f"Test:\t\t{len(test)}")

    return train, val, test, count_tree

def create_new_ds(
    hierarchy: Hierarchy, 
    gen_aug_params:CfgNode,
    split:Split,
    train_not_classified_proportions: List[float],
    run_name:str,
    trans_mean:tuple,
    trans_std:tuple,
    mode:Mode,
    seed: int = 2024,
    log: Callable = print
):
    """
    Creates train, train_push, validation, and test dataloaders for the tree dataset.
    gen_aug_params - object genetic augmentation parameters to apply to the genetic data. (found in cfg.py)
    train_not_classified_proportion - An object specifying the porportion of samples at each level that should be not classified.
    tree_specification_file - path to json file tree of valid classes.
    """
    np.random.seed(seed) 

    train, val, test, _ = balanced_sample(
        hierarchy, 
        pd.read_csv(os.path.join("..", "datasets", "source_files", "metadata_cleaned_permissive.tsv"), sep="\t"),
        split, 
        train_not_classified_proportions, 
        seed
    )
    
    # save the train_push dataframe
    push = train.copy()

    # Save all three datasets
    output_dir = os.path.join("pre_existing_datasets", run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    push.to_csv(os.path.join(output_dir, "train_push.tsv"), sep="\t", index=False)
    val.to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False)
    test.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)

    log(f"Saved all datasets to tsv files in {output_dir}")

    # generate the transforms for image and genetics datasets
    aug_img_trans, aug_gen_trans, push_img_trans, img_trans, \
    normalize = create_transforms(trans_mean, trans_std, gen_aug_params)
    
    # create the datasets
    train = TreeDataset(hierarchy, train, mode, aug_img_trans, aug_gen_trans) 
    push = TreeDataset(hierarchy, push, mode, push_img_trans, None) 
    val = TreeDataset(hierarchy, val, mode, img_trans, None) 
    test = TreeDataset(hierarchy, test, mode, img_trans, None)

    log("Datasets created")
    return train, push, val, test, normalize

def retrieve_cached_ds(
    hierarchy: Hierarchy, 
    cache_ds_dir:str,
    gen_aug_params:CfgNode,
    trans_mean:tuple,
    trans_std:tuple,
    mode:Mode,
    log: Callable = print
):
    """
    Take a hierarchy and some dataset tsv file in cache_ds_dir 
    Create datasets with the transforms defined in gen_aug_params, trans_mean/std
    """
    train = pd.read_csv(os.path.join(cache_ds_dir, "train.tsv"), sep="\t")
    push = pd.read_csv(os.path.join(cache_ds_dir, "train_push.tsv"), sep="\t")
    val = pd.read_csv(os.path.join(cache_ds_dir, "val.tsv"), sep="\t")
    test = pd.read_csv(os.path.join(cache_ds_dir, "test.tsv"), sep="\t")

    aug_img_trans, aug_gen_trans, push_img_trans, img_trans, \
    normalize = create_transforms(trans_mean, trans_std, gen_aug_params)
    
    # create the datasets
    train = TreeDataset(hierarchy, train, mode, aug_img_trans, aug_gen_trans) 
    push = TreeDataset(hierarchy, push, mode, push_img_trans, None) 
    val = TreeDataset(hierarchy, val, mode, img_trans, None) 
    test = TreeDataset(hierarchy, test, mode, img_trans, None)

    log("Datasets created")
    return train, push, val, test, normalize

def get_datasets(cfg: CfgNode, log: Callable = print): 
    log("Getting Datasets") 
    hierarchy = Hierarchy(cfg.DATASET.TREE_SPECIFICATION_FILE)

    if cfg.DATASET.CACHED_DATASET_FOLDER.strip(): 
        # if we want to retrieve a cached dataset
        return retrieve_cached_ds(
            hierarchy = hierarchy, 
            cache_ds_dir = cfg.DATASET.CACHED_DATASET_FOLDER,
            gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
            trans_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            trans_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
            mode=Mode(cfg.DATASET.MODE),
            log=log
        )
    else: 
        # if we want to create a new dataset 
        return create_new_ds(
            hierarchy = hierarchy, 
            gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
            split = Split(*cfg.DATASET.TRAIN_VAL_TEST_SPLIT),
            train_not_classified_proportions=cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS,
            run_name=cfg.RUN_NAME,
            trans_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
            trans_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
            mode=Mode(cfg.DATASET.MODE),
            seed=cfg.SEED,
            log=log,
        )

