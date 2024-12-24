import os, time, json
from typing import Optional, Dict, Text, Tuple, List
import Augmentor
import torch 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import numpy as np
from skimage import io
from torchvision.transforms import transforms
from model.hierarchical_ppnet import Mode
from yacs.config import CfgNode
from typing_extensions import deprecated

class GeneticOneHot():
    """Map a genetic string to a one-hot encoded tensor, values being in the color channel dimension.

    Args:
        length (int): The length of the one-hot encoded tensor. Samples will be padded (with Ns) or truncated to this length.
        zero_encode_unknown (bool, optional): Whether to encode unknown characters as all zeroes. Otherwise, encode them as (1,0,0,0,0). Default is True.
        include_height_channel (bool, optional): Whether to include a height channel in the one-hot encoding. Default is False.
    """

    def __init__(self, length:int=720, zero_encode_unknown: bool=True, include_height_channel: bool=False):
        self.zero_encode_unknown = zero_encode_unknown
        self.length = length
        self.include_height_channel = include_height_channel

    def __call__(self, genetic_string: str):
        """
        Args:
            genetics (str): The genetic data to be transformed.

        Returns:
            torch.Tensor: A one-hot encoded tensor of the genetic data.
        """
        # Create a dictionary mapping nucleotides to their one-hot encoding
        nucleotides = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}

        # Convert string to (1, 2, 1, 4, 0, ...)
        category_tensor = torch.tensor([nucleotides[n] for n in genetic_string])
        
        # Pad and crop
        category_tensor = category_tensor[:self.length]
        category_tensor = F.pad(category_tensor, (0, self.length - len(category_tensor)), value=0)

        # One-hot encode
        onehot_tensor = F.one_hot(category_tensor, num_classes=5).permute(1, 0)
        
        # Drop the 0th channel, changing N (which is [1,0,0,0,0]) to [0,0,0,0] and making only 4 classes
        if self.zero_encode_unknown:
            onehot_tensor = onehot_tensor[1:, :]

        if self.include_height_channel:
            onehot_tensor = onehot_tensor.unsqueeze(1)

        return onehot_tensor.float()

class ImageGeometricTransform():
    def __init__(self):
        pass

    def __call__(self, image):
        r: int = np.random.randint(0, 4) 
        p = Augmentor.Pipeline()
        match r: 
            case 0: 
                p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
            case 1: 
                p.skew(probability=1, magnitude=0.2) # type: ignore
            case 2: 
                p.shear(probability=1, max_shear_left=10, max_shear_right=10)
            case 3: 
                p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)

        p.flip_left_right(probability=0.5)
        return p.torch_transform()(image)

class GeneticMutationTransform(): 
    def __init__(self, insertion_amount: int, deletion_amount: int, substitution_rate: float): 
        self.insertion_amount = insertion_amount 
        self.deletion_amount = deletion_amount 
        self.substitution_rate = substitution_rate

    def __call__(self, sample: str) -> str:
        insertion_count = np.random.randint(0, self.insertion_amount+1)
        deletion_count = np.random.randint(0, self.deletion_amount+1)

        insertion_indices = np.random.randint(0, len(sample), insertion_count)
        for idx in insertion_indices:
            sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx:]
        
        deletion_indices = np.random.randint(0, len(sample), deletion_count)
        for idx in deletion_indices:
            sample = sample[:idx] + sample[idx+1:]
        
        mutation_indices = np.random.choice(len(sample), int(len(sample) * self.substitution_rate), replace=False)
        for idx in mutation_indices:
            sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx+1:]
        
        return sample

class TreeDataset(Dataset):
    """
    This is a heirarchichal dataset for genetics and images.
    """

    def __init__(
        self, 
        source_df:pd.DataFrame, 
        image_root_dir: str, 
        class_specification, 
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

    def __getitem__(self, idx) -> Tuple[Tuple[Optional[Tensor], Optional[np.ndarray]], Optional[Tensor]]:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root_dir, row["order"],row["family"], row["genus"], row["species"], row["image_file"]) 
        
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

    def get_label(self, row:pd.Series):
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

    def __len__(self):
        return len(self.df)

def oversample(source_df: pd.DataFrame, count: int, seed:int=2024, log=print) -> pd.DataFrame:
    log("Oh no, we oversampled. This will break data augmentation... Must improve data augmentation.")
    # Fairly oversample the source_df to count
    shortage = count - len(source_df)
    output = pd.concat([source_df] + [source_df] * (shortage // len(source_df)) + [source_df.sample(shortage % len(source_df), replace=False, random_state=seed)])
    return output.sample(frac=1, random_state=seed)

def balanced_sample(
    source_df:pd.DataFrame, 
    class_specification:dict, 
    count_per_leaf:tuple, 
    train_not_classified_proportion: dict, 
    seed:int = 2024, 
    log = print
    ):
    """
    Returns a balanced sample of the source_df based on the class_specification and count_per_leaf.
    """
    source_df = source_df.sample(frac=1, random_state=seed)

    tree, levels = class_specification["tree"], class_specification["levels"]

    def recursive_balanced_sample(
        tree:dict, 
        levels, 
        source_df:pd.DataFrame, 
        count_per_leaf:tuple, 
        train_not_classified_proportion, 
        seed:int = 2024, 
        parent_name: Optional[str] = None, 
        log=print) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Tuple[int, int, int]]]:
        train_output = []
        val_output = []
        test_output = []
        count_tree = {}

        def proportional_assign(total_count, count_per_leaf):
            temp_count_per_leaf = np.ceil(np.array(count_per_leaf) * ((total_count - 3) / (count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2])))
            temp_count_per_leaf += 1

            while sum(temp_count_per_leaf) > total_count:
                temp_count_per_leaf[np.random.choice(3)] -= 1

            return temp_count_per_leaf.astype(int)


        if len(levels) == 1:
            # [validation shortage, test shortage]
            shortages = np.zeros(2)
            for k,v in tree.items():
                if k == "not_classified":
                    continue
                class_size = len(source_df[source_df[levels[0]] == k])

                if class_size < 3:
                    raise ValueError(f"Less than 3 samples for {k} of {parent_name} at level {levels[0]}. Unable to proceed.")
                if class_size < count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2]:
                    log(f"Only {class_size} (of needed {count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2]}) samples for {k} ({levels[0]}) of parent {parent_name}. Dividing proportionally")
                    
                    temp_count_per_leaf = proportional_assign(class_size, count_per_leaf)

                    log(f"New counts for {k}")
                    log(f"Train:\t\t{temp_count_per_leaf[0]}")
                    log(f"Validation:\t{temp_count_per_leaf[1]}")
                    log(f"Test:\t\t{temp_count_per_leaf[2]}")

                    shortages += np.array(count_per_leaf)[1:] - temp_count_per_leaf[1:]
                else:
                    temp_count_per_leaf = count_per_leaf

                test_sample = source_df[source_df[levels[0]] == k].sample(temp_count_per_leaf[2], random_state=seed)
                validation_sample = source_df[source_df[levels[0]] == k].drop(index=list(test_sample.index)).sample(temp_count_per_leaf[1], random_state=seed)
                train_sample = pd.DataFrame(source_df[source_df[levels[0]] == k].drop(index=list(test_sample.index)).drop(index=list(validation_sample.index)).sample(temp_count_per_leaf[0], random_state=seed))

                if len(train_sample) < count_per_leaf[0]:
                    train_sample = oversample(train_sample, count_per_leaf[0], seed)

                train_output.append(train_sample)
                val_output.append(validation_sample)
                test_output.append(test_sample)

                count_tree[k] = (len(train_sample), len(validation_sample), len(test_sample))            
        else:
            shortages = np.zeros(2)
            for k,v in tree.items():
                if v==None:
                    not_classified = source_df[source_df[levels[0]] == k]

                    temp_count_per_leaf = count_per_leaf
                    child_shortages = np.zeros(2)

                    if len(not_classified) < count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2]:
                        log(f"Only {len(not_classified)} (of needed {count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2]}) samples for {k} ({levels[0]}) of parent {parent_name}. Dividing proportionally")
                        
                        temp_count_per_leaf = proportional_assign(len(not_classified), count_per_leaf)

                        log(f"New counts for {k}")
                        log(f"Train:\t\t{temp_count_per_leaf[0]}")
                        log(f"Validation:\t{temp_count_per_leaf[1]}")
                        log(f"Test:\t\t{temp_count_per_leaf[2]}")

                        child_shortages += np.array(count_per_leaf)[1:] - temp_count_per_leaf[1:]

                    child_test = not_classified.sample(temp_count_per_leaf[2], random_state=seed)
                    child_val = not_classified.drop(index=list(child_test.index)).sample(temp_count_per_leaf[1], random_state=seed)
                    child_train = not_classified.drop(index=list(child_test.index)).drop(index=list(child_val.index)).sample(temp_count_per_leaf[0], random_state=seed)
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

        # Convert shortages to int
        shortages = shortages.astype(int)

        # Handle not_classified
        not_classified = source_df[(source_df[levels[0]] == "not_classified") | (~source_df[levels[0]].isin([c for c in tree.keys()]))]
        train_not_classified_count = train_not_classified_proportion[levels[0]] * (len(tree.keys()) - 1) * count_per_leaf[0]
        train_not_classified_count = int(train_not_classified_count)

        if len(not_classified) < np.sum(shortages) + train_not_classified_count:
            log(f"Unable to counterbalance with not_classified for {parent_name} at {levels[0]}. Sorry.")
            # TODO - This could be handled by going up one level and adding more not classified.
            not_classified_sample_amounts = np.array([0,0])
            train_not_classified_count, not_classified_sample_amounts[0], not_classified_sample_amounts[1] = proportional_assign(len(not_classified), count_per_leaf)
            # raise NotImplementedError(f"Unable to counterbalance with not_classified for {parent_name} at {levels[0]}. We could handle one level up. This is not yet implemented.")
        else:
            not_classified_sample_amounts = shortages

        test_sample = not_classified.sample(not_classified_sample_amounts[1], random_state=seed)
        validation_sample = not_classified.drop(index=list(test_sample.index)).sample(not_classified_sample_amounts[0], random_state=seed)

        train_sample = not_classified.drop(index=list(test_sample.index)).drop(index=list(validation_sample.index)).sample(train_not_classified_count, random_state=seed)

        train_output.append(train_sample)
        val_output.append(validation_sample)
        test_output.append(test_sample)

        count_tree["not_classified"] = (len(train_sample), len(validation_sample), len(test_sample))

        return pd.concat(train_output), pd.concat(val_output), pd.concat(test_output), shortages - not_classified_sample_amounts, count_tree

    train_df, val_df, test_df, shortages, count_tree = recursive_balanced_sample(tree, levels, source_df, count_per_leaf, train_not_classified_proportion, seed)

    if np.any(shortages > 0):
        raise ValueError(f"Unable to balance dataset. Shortages: {shortages}. This should not happen, I am very confused.")

    log("---- Overall Results ----")
    log(f"Train:\t\t{len(train_df)}")
    log(f"Validation:\t{len(val_df)}")
    log(f"Test:\t\t{len(test_df)}")

    return train_df, val_df, test_df, count_tree

def create_tree_dataloaders(
    source: str,                        # path to dataset tsv file
    image_root_dir: str,
    gen_aug_params:CfgNode,
    train_val_test_split:Tuple[int, int, int],
    oversampling_rate:int,
    train_not_classified_proportions: List[float],
    tree_specification_file:str,
    cached_dataset_folder:str,
    cached_dataset_root:str,
    run_name:str,
    train_batch_size:int,
    train_push_batch_size:int,
    test_batch_size:int,
    transform_mean:tuple,
    transform_std:tuple,
    mode:Mode,
    seed=2024,
    flat_class=False,
    log=print
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, transforms.Normalize]:
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
    # Set pandas random seed
    np.random.seed(seed) 

    # load the tree 
    class_specification = json.load(open(tree_specification_file, "r"))

    if cached_dataset_folder.strip():
        train_df = pd.read_csv(os.path.join(cached_dataset_folder, "train.tsv"), sep="\t")
        train_push_df = pd.read_csv(os.path.join(cached_dataset_folder, "train_push.tsv"), sep="\t")
        val_df = pd.read_csv(os.path.join(cached_dataset_folder, "val.tsv"), sep="\t")
        test_df = pd.read_csv(os.path.join(cached_dataset_folder, "test.tsv"), sep="\t")
    else:
        if len(train_val_test_split) != 3 or not all([isinstance(i, int) and i >= 0 for i in train_val_test_split]):
            raise ValueError("train_val_test_split must be a 3-tuple of positive integers (train, val, test)")

        def objectify_train_not_classified_proportion(
            train_not_classified_proportion: List[float], 
            levels: List[Text]
            ) -> Dict[Text, float]: 
            """
            Takes train_not_classified_proportion in [0,0,0,0] form and converts it to {'order': 0, ...}
            """
            return {level: proportion for level, proportion in zip(levels, train_not_classified_proportion)}

        
        df = pd.read_csv(source, sep="\t") if isinstance(source, str) else source 

        train_df, val_df, test_df, _ = balanced_sample(
            df, 
            class_specification, 
            train_val_test_split, 
            objectify_train_not_classified_proportion(
                train_not_classified_proportions, 
                class_specification["levels"]
            ),
            seed
        )

        train_push_df = train_df.copy()

        old_train_size = len(train_df)
        start_t = time.time()
        if oversampling_rate != 1:
            train_df = oversample(train_df, len(train_df) * oversampling_rate, seed)
        new_train_size = len(train_df)

        log(f"Oversampled train from {old_train_size:,} samples to {new_train_size:,} samples")
        log(f"And it only took {time.time() - start_t:.2f} seconds")

        # Save all three datasets
        output_dir = os.path.join(cached_dataset_root, run_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        train_path = os.path.join(output_dir, "train.tsv")
        train_push_path = os.path.join(output_dir, "train_push.tsv")
        val_path = os.path.join(output_dir, "val.tsv")
        test_path = os.path.join(output_dir, "test.tsv")

        train_df.to_csv(train_path, sep="\t", index=False)
        log(f"Saved train dataset to {train_path}")

        train_push_df.to_csv(train_push_path, sep="\t", index=False)
        log(f"Saved train_push dataset to {train_push_path}")

        val_df.to_csv(val_path, sep="\t", index=False)
        log(f"Saved validation dataset to {val_path}")

        test_df.to_csv(test_path, sep="\t", index=False)
        log(f"Saved test dataset to {test_path}")

    normalize = transforms.Normalize(
        mean=transform_mean, 
        std=transform_std
    )

    push_img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
    ])
    img_transforms = transforms.Compose([
        push_img_transforms,
        normalize
    ])

    augmented_img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        ImageGeometricTransform(),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        normalize
    ])
    augmented_genetic_transforms = GeneticMutationTransform(
        insertion_amount=gen_aug_params.INSERTION_COUNT,
        deletion_amount=gen_aug_params.DELETION_COUNT,
        substitution_rate=gen_aug_params.SUBSTITUTION_RATE
    )
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

def get_dataloaders(cfg, log, flat_class=False):
    log("Getting Dataloaders")

    return create_tree_dataloaders(
        source=cfg.DATASET.DATA_FILE,
        image_root_dir=cfg.DATASET.IMAGE_PATH,
        gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
        train_val_test_split=cfg.DATASET.TRAIN_VAL_TEST_SPLIT,
        oversampling_rate=cfg.DATASET.OVERSAMPLING_RATE,
        train_not_classified_proportions=cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS,
        tree_specification_file=cfg.DATASET.TREE_SPECIFICATION_FILE,
        cached_dataset_folder=cfg.DATASET.CACHED_DATASET_FOLDER,
        cached_dataset_root=cfg.DATASET.CACHED_DATASET_ROOT,
        run_name=cfg.RUN_NAME,
        train_batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
        train_push_batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE,
        test_batch_size=cfg.DATASET.TEST_BATCH_SIZE,
        transform_mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN,
        transform_std=cfg.DATASET.IMAGE.TRANSFORM_STD,
        mode=Mode(cfg.DATASET.MODE),
        seed=cfg.SEED,
        log=log,
        flat_class=flat_class
    )

@deprecated("""Only called in augment_train_dataset, which is deprecated.
    Seems like this has same functionality as GeneticMutationTransform.__call__()""")
def mutate_sample():
    raise NotImplementedError()
