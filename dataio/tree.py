import os
import shutil
import time
import Augmentor
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Any
import torch.nn.functional as F
import numpy as np
from typing import Optional
import copy
from skimage import io, transform
from torchvision.transforms import v2, ToTensor
import tqdm
import json
from torchvision.transforms.functional import center_crop

class GeneticOneHot(object):
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

# class MutateGenetic(object):
#     def __init__(self, substitution_rate: float=0.01, insertion_count: int=5, deletion_count: int=5):
#         self.substitution_rate = substitution_rate
#         self.insertion_count = insertion_count
#         self.deletion_count = deletion_count

#     def get_real_length(self, genetic_tensor: torch.Tensor):
#         # Find the index of the first all zero channel efficiently
#         summed = genetic_tensor.sum(dim=0)
#         real_length = torch.argmin(summed)

#         return real_length
        

#     def __call__(self, genetic_tensor: torch.Tensor):
#         # Genetic tensor is a one-hot encoded tensor of the genetic data.
#         insertion_amount = np.random.randint(0, self.insertion_count)
#         deletion_amount = np.random.randint(0, self.deletion_count)

#         real_length = self.get_real_length(genetic_tensor)

#         print(genetic_tensor)

#         # Substitution
#         substitution_mask = torch.rand((1, 1, genetic_tensor.shape[2])) < self.substitution_rate
#         # Copy the mask to the correct shape
#         substitution_mask = substitution_mask.expand(genetic_tensor.shape[0], genetic_tensor.shape[1], genetic_tensor.shape[2])
#         substitution_values = torch.randint(0, 4, (1, genetic_tensor.shape[2]))
#         # One-hot encode substitution values
#         substitution_values = F.one_hot(substitution_values, num_classes=4).permute(2,0,1).float()
#         genetic_tensor[substitution_mask] = substitution_values[substitution_mask]

#         # Insertion
#         insertion_indices = torch.randint(0, real_length, (insertion_amount,))
#         insertion_values = torch.randint(1, 5, (insertion_amount, genetic_tensor.shape[1]))
#         genetic_tensor = torch.cat([genetic_tensor[:i], insertion_values, genetic_tensor[i:]], dim=0)

#         # Deletion
#         deletion_indices = torch.randint(0, real_length, (deletion_amount,))
#         genetic_tensor = torch.cat([genetic_tensor[:i], genetic_tensor[i+1:]], dim=0)

#         # Make sure the tensor is the same length
#         genetic_tensor = F.pad(genetic_tensor, (0, self.length - len(genetic_tensor)), value=0)

#         return genetic_tensor

class TreeDataset(Dataset):
    """
    This is a heirarchichal dataset for genetics and images.

    mode 0 - error
    mode 1 - returns genetic_tensor, label
    mode 2 - returns image_tensor, label
    mode 3 - returns (genetic_tensor, image_tensor), label
    """

    def __init__(self, source_df:str, image_root_dir: str, class_specification, mode=3, genetic_augmentations={}):
        self.df = source_df
        self.image_root_dir = image_root_dir

        self.tree, self.levels = class_specification["tree"], class_specification["levels"]
        self.indexed_tree = self.generate_tree_indicies(self.tree)
        self.mode = mode
        self.one_hot_encoder = GeneticOneHot(length=720, zero_encode_unknown=True, include_height_channel=True)
        
    def mutate_genetics(self, df:pd.DataFrame):
        return df.apply(self.mutate_sample, axis=1)

    def generate_tree_indicies(self, tree:dict, idx=0):
        """
        This is deeply gross and I appologize for it.
        """
        tree = {k: v for k,v in sorted(tree.items())}
        tree["idx"] = idx

        idx = 0

        for k,v in tree.items():
            if k == "idx":
                continue
            if isinstance(v, dict):
                tree[k] = self.generate_tree_indicies(v, idx)
            else:
                tree[k] = {
                    "idx": idx
                }
            idx += 1

        return tree

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root_dir, row["image_file"])
        
        if self.mode & 1:
            genetics = row["nucraw"]
            genetics = self.one_hot_encoder(genetics)
        else:
            genetics = None
        
        if self.mode >> 1:
            image = io.imread(image_path)
            image = torch.tensor(image)

            # TODO - Replace this w/ a process that applies this to the images in the image folder, instead of doing it constantly.
            if image.shape[0] != 3:
                image = torch.permute(image, (2,0,1))

            image = center_crop(image, (256, 256))
            image = F.pad(image, (0, 0, 256 - image.shape[1], 256 - image.shape[2]), value=0)
        else:
            image = None

        label = self.get_label(row)

        if not genetics:
            return image, label
        if not image:
            return genetics, label

        # TODO - Use a collate function to handle this.
        return (genetics, image), label

    def get_label(self, row:pd.Series):
        """
        Label is a tensor of indicies for each level of the tree.

        0 represents not_classified.
        """
        tensor = torch.zeros(len(self.levels))

        tree = self.indexed_tree

        for i, level in enumerate(self.levels):
            if row[level] == "not_classified":
                tensor[i:] = 0
                break
            else:
                tensor[i] = tree[row[level]]["idx"] + 1
                tree = tree[row[level]]
        
        return tensor

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    """
    This is a collate function for the TreeDataset.
    """
    genetics = []
    images = []
    labels = []

    for item in batch:
        if item[0]:
            genetics.append(item[0])
        if item[1]:
            images.append(item[1])
        labels.append(item[2])

    if genetics:
        genetics = torch.stack(genetics)
    if images:
        images = torch.stack(images)
    labels = torch.stack(labels)

    return genetics, images, labels

def proportional_assign(total_count, count_per_leaf):
    temp_count_per_leaf = np.ceil(np.array(count_per_leaf) * ((total_count - 3) / (count_per_leaf[0] + count_per_leaf[1] + count_per_leaf[2])))
    temp_count_per_leaf += 1

    while sum(temp_count_per_leaf) > total_count:
        temp_count_per_leaf[np.random.choice(3)] -= 1

    return temp_count_per_leaf.astype(int)

def extract_id(path: str):
    return ""

def augment_oversample(source_df: pd.DataFrame, image_path: str, augmented_image_path:str, oversampling_rate: int, pre_existing=False, seed:int=2024, log=print):
    if not pre_existing:
        if oversampling_rate % 4 != 0:
            raise ValueError("Oversampling rate must be a multiple of 4.")
        
        # Create a temporary folder
        temp_image_path = os.path.join(augmented_image_path, "temp")
        if not os.path.exists(temp_image_path):
            os.makedirs(temp_image_path)

        if not os.path.exists(augmented_image_path):
            os.makedirs()

        # Copy each image in source_df["image_path"] to temp_image_path
        for idx, row in source_df.iterrows():
            image_path = os.path.join(image_path, row["image_file"])
            temp_image_path = os.path.join(temp_image_path, row["image_file"])
            shutil.copy(image_path, temp_image_path)

        # Get the indicies of all duplicated IDs in source_df
        duplicated_ids = source_df[source_df.duplicated(subset="image_file")]
        accessed_duplicated_ids = set()

        # Loop over duplicated ids, change their ids to be unique and duplicate the images
        for idx, row in duplicated_ids.iterrows():
            old_id = row["sample_id"]
            if old_id in accessed_duplicated_ids:
                accessed_duplicated_ids.add(old_id)
                continue

            new_id = f"{old_id}_{idx}"
            # Update the source_df with the new id
            source_df.loc[source_df["sample_id"] == old_id, "sample_id"] = new_id

            new_image_file = f"{new_id}.jpg"

            # Copy the image to the new id
            image_path = os.path.join(image_path, row["image_file"])
            new_image_path = os.path.join(temp_image_path, new_image_file)
            shutil.copy(image_path, new_image_path)
            log(f"[Duplicate] Copied {image_path} to {new_image_path}")

            # Update the source_df with the new image file
            source_df.loc[source_df["sample_id"] == new_id, "image_file"] = new_image_file

        log("Duplicated IDs handled")

        # rotation
        p = Augmentor.Pipeline(source_directory=temp_image_path, output_directory=augmented_image_path)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in tqdm(range(oversampling_rate // 4)):
            p.process()
        del p

        log("Rotation done")

        # skew
        p = Augmentor.Pipeline(source_directory=temp_image_path, output_directory=augmented_image_path)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in tqdm(range(oversampling_rate // 4)):
            p.process()
        del p

        log("Skew done")

        # shear
        p = Augmentor.Pipeline(source_directory=temp_image_path, output_directory=augmented_image_path)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in tqdm(range(oversampling_rate // 4)):
            p.process()
        del p

        log("Shear done")
        
        #random_distortion
        p = Augmentor.Pipeline(source_directory=temp_image_path, output_directory=augmented_image_path)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in tqdm(range(oversampling_rate // 4)):
            p.process()
        del p

        log("Random Distortion done")

        # Clean up
        os.rmdir(temp_image_path)
    
    # Get all the images in the augmented_image_path
    augmented_images = os.listdir(augmented_image_path)
    augmented_image_ids = [extract_id(image) for image in augmented_images]

    augmented_image_id_map = {image_id: [] for image_id in augmented_image_ids}
    for image, image_id in zip(augmented_images, augmented_image_ids):
        augmented_image_id_map[image_id].append(image)

    # Oversample the source_df

    output = []

    oversampled_df = pd.concat([source_df] * oversampling_rate)

    for idx, row in oversampled_df.iterrows():
        image_id = extract_id(row["image_file"])
        if len(augmented_image_id_map[image_id]) == 0:
            raise ValueError(f"Image {row['image_file']} has no augmented images. This should not happen.")
        
        augmented_image = augmented_image_id_map[image_id][idx // len(source_df)]
        row["image_file"] = augmented_image
        output.append(row)

    return pd.DataFrame(output).sample(frac=1, random_state=seed)

def oversample(source_df: pd.DataFrame, count: int, seed:int=2024, log=print):
    log("Oh no, we oversampled. This will break data augmentation... Must improve data augmentation.")
    # Fairly oversample the source_df to count
    shortage = count - len(source_df)
    output = pd.concat([source_df] + [source_df] * (shortage // len(source_df)) + [source_df.sample(shortage % len(source_df), replace=False, random_state=seed)])
    return output.sample(frac=1, random_state=seed)

def recursive_balanced_sample(tree:Optional[dict], levels, source_df:pd.DataFrame, count_per_leaf:tuple, train_not_classified_proportion, seed:int=2024, parent_name:str=None, log=print):
    train_output = []
    val_output = []
    test_output = []
    count_tree = {}

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
            validation_sample = source_df[source_df[levels[0]] == k].drop(test_sample.index).sample(temp_count_per_leaf[1], random_state=seed)
            train_sample = source_df[source_df[levels[0]] == k].drop(test_sample.index).drop(validation_sample.index).sample(temp_count_per_leaf[0], random_state=seed)

            if len(train_sample) < count_per_leaf[0]:
                train_sample = oversample(train_sample, count_per_leaf[0], seed)

            train_output.append(train_sample)
            val_output.append(validation_sample)
            test_output.append(test_sample)

            count_tree[k] = (len(train_sample), len(validation_sample), len(test_sample))            
    else:
        shortages = np.zeros(2)
        for k,v in tree.items():
            if k == "not_classified":
                continue

            if v==None:
                not_classified = source_df[source_df[levels[0]] == k][source_df[levels[1]] == "not_classified"]
                child_test = not_classified.sample(count_per_leaf[2], random_state=seed)
                child_val = not_classified.drop(child_test.index).sample(count_per_leaf[1], random_state=seed)
                child_train = not_classified.drop(child_test.index).drop(child_val.index).sample(count_per_leaf[0], random_state=seed)
                child_shortages = np.zeros(2)
                child_count_tree = {"not_classified": (len(child_train), len(child_val), len(child_test))}
            else:
                child_train, child_val, child_test, child_shortages, child_count_tree = recursive_balanced_sample(
                    v,
                    levels[1:],
                    source_df[source_df[levels[0]] == k],
                    count_per_leaf,
                    train_not_classified_proportion,
                    seed,
                    k
                )
            shortages += child_shortages

            train_output.append(child_train)
            val_output.append(child_val)
            test_output.append(child_test)

            count_tree[k] = child_count_tree

    # Convert shortages to int
    shortages = shortages.astype(int)

    # Handle not_classified
    not_classified = source_df[source_df[levels[0]] == "not_classified"]
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
    validation_sample = not_classified.drop(test_sample.index).sample(not_classified_sample_amounts[0], random_state=seed)

    train_sample = not_classified.drop(test_sample.index).drop(validation_sample.index).sample(train_not_classified_count, random_state=seed)

    train_output.append(train_sample)
    val_output.append(validation_sample)
    test_output.append(test_sample)

    count_tree["not_classified"] = (len(train_sample), len(validation_sample), len(test_sample))

    return pd.concat(train_output), pd.concat(val_output), pd.concat(test_output), shortages - not_classified_sample_amounts, count_tree

def objectify_train_not_classified_proportion(train_not_classified_proportion:list, levels:list):
    """
    Takes train_not_classified_proportion in [0,0,0,0] form and converts it to {'order': 0, ...}
    """

    return {level: proportion for level, proportion in zip(levels, train_not_classified_proportion)}

def balanced_sample(source_df:pd.DataFrame, class_specification:tuple, count_per_leaf:tuple, train_not_classified_proportion={}, seed:int=2024, log=print):
    """
    Returns a balanced sample of the source_df based on the class_specification and count_per_leaf.
    """
    source_df = source_df.sample(frac=1, random_state=seed)

    tree, levels = class_specification["tree"], class_specification["levels"]

    train_df, val_df, test_df, shortages, count_tree = recursive_balanced_sample(tree, levels, source_df, count_per_leaf, train_not_classified_proportion, seed)

    if np.any(shortages > 0):
        raise ValueError(f"Unable to balance dataset. Shortages: {shortages}. This should not happen, I am very confused.")

    log("---- Overall Results ----")
    log(f"Train:\t\t{len(train_df)}")
    log(f"Validation:\t{len(val_df)}")
    log(f"Test:\t\t{len(test_df)}")

    return train_df, val_df, test_df, count_tree

def mutate_sample(insertion_amount=5, deletion_amount=5, substitution_rate=0.01):
    insertion_count = np.random.randint(0, insertion_amount+1)
    deletion_count = np.random.randint(0, deletion_amount+1)

    insertion_indices = np.random.randint(0, len(sample), insertion_count)
    for idx in insertion_indices:
        sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx:]
    
    deletion_indices = np.random.randint(0, len(sample), deletion_count)
    for idx in deletion_indices:
        sample = sample[:idx] + sample[idx+1:]
    
    mutation_indices = np.random.choice(len(sample), int(len(sample) * substitution_rate), replace=False)
    for idx in mutation_indices:
        sample = sample[:idx] + np.random.choice(list("ACGT")) + sample[idx+1:]
    
    return sample

def check_cached_images(source, image_cache_dir, log=print):
    """
    Check if the images are already cached. If not, cache them.
    """
    if not os.path.exists(image_cache_dir):
        os.makedirs(image_cache_dir)
    
    for idx, row in source.iterrows():
        image_path = os.path.join(image_cache_dir, row["image_file"])
        if not os.path.exists(image_path):
            log(f"{image_path} not found. Augmenting Images.")
            return False

    return True

def augment_train_dataset(source, image_root_dir, image_cache_dir, gen_aug_params):
    """
    This applies image and genetic augmentations to the training dataset.
    NOTE: This does not oversample the training dataset.
    """

    substitution_rate = gen_aug_params.SUBSTITUTION_RATE
    insertion_amount = gen_aug_params.DELETION_COUNT
    deletion_amount = gen_aug_params.INSERTION_COUNT

    # Mutate genetics
    source = source.apply(lambda x: mutate_sample(x, insertion_amount=insertion_amount, deletion_amount=deletion_amount, substitution_rate=substitution_rate), axis=1)

    # Augment images - Rotation, skew, shear, and random distortion
    p = Augmentor.Pipeline()
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)

    rotation_transform = v2.Compose([
        p.torch_transform(),
        ToTensor()]
    )
    
    # skew_transform = v2.Compose([
    #     v2.RandomAffine(degrees=0, shear=45),
    #     v2.RandomHorizontalFlip(),
    # ])

    shear_transform = v2.Compose([
        v2.RandomAffine(degrees=0, shear=10),
        v2.RandomHorizontalFlip(),
    ])

    distortion_transform = v2.Compose([
        v2.ElasticTransform(alpha=1, sigma=10),
        v2.RandomHorizontalFlip(),
    ])


    return source

def create_tree_dataloaders(source, image_root_dir: str, image_cache_dir: str, gen_aug_params:object, train_val_test_split:tuple, oversampling_rate:int, train_not_classified_proportions:tuple, tree_specification_file:str, pre_existing_images:bool, cached_dataset_folder:str, cached_dataset_root:str, run_name:str, train_batch_size:int, train_push_batch_size:int, test_batch_size:int, mode:int, seed=2024, log=print):
    """
        Creates train, train_push, validation, and test dataloaders for the tree dataset.
        
        source_file - tsv that contains all needed data (ex. metadata_cleaned_permissive.tsv). Note: all entires in source_file must have images in image_root
        image_root_dir - root directory for the images.
        gen_aug_params - object genetic augmentation parameters to apply to the genetic data.
        image_augmentations - list of image augmentations to apply to the image data.
        train_val_test_split - 3-tuple of integers representing the number of true train, validation, and test samples for each leaf node of the tree. In most cases, it's the desired # of samples per species. NOTE: This does not include oversampling of train samples.
        train_end_count - The number of samples to end with in the training set.
        train_not_classified_proportion - An object specifying the porportion of samples at each level that should be not classified.
        tree_specification_file - path to json file tree of valid classes.
        pre_existing_images - If True, the images are assumed to be pre_existing in the image_cache_dir. If False, the images are assumed to be in the image_root_dir and will be copied to the image_cache_dir.
        mode - 0 is illegal (straight to jail), 1 is genetic only, 2 is image only, 3 is both. (Think binary counting)
        seed - random seed for splitting, shuffling, transformations, etc.
    """
    # Set pandas random seed
    np.random.seed(seed)
    class_specification = json.load(open(tree_specification_file, "r"))

    if cached_dataset_folder and len(cached_dataset_folder):
        train_df = pd.read_csv(os.path.join(cached_dataset_folder, "train.tsv"), sep="\t")
        train_push_df = pd.read_csv(os.path.join(cached_dataset_folder, "train_push.tsv"), sep="\t")
        val_df = pd.read_csv(os.path.join(cached_dataset_folder, "val.tsv"), sep="\t")
        test_df = pd.read_csv(os.path.join(cached_dataset_folder, "test.tsv"), sep="\t")
    else:
        if mode == 0:
            raise ValueError("Mode 0 not allowed. This means it's not genetic or image :(.)")

        if len(train_val_test_split) != 3 or not all([isinstance(i, int) and i >= 0 for i in train_val_test_split]):
            raise ValueError("train_val_test_split must be a 3-tuple of positive integers (train, val, test)")
        
        if type(source) == str:
            df = pd.read_csv(source, sep="\t")
        else:
            df = source

        train_not_classified_proportions = objectify_train_not_classified_proportion(train_not_classified_proportions, class_specification["levels"])
        train_df, val_df, test_df, count_tree = balanced_sample(df, class_specification, train_val_test_split, train_not_classified_proportions, seed)

        train_push_df = train_df.copy()

        old_train_size = len(train_df)
        start_t = time.time()
        if oversampling_rate != 1:
            raise NotImplementedError("Sorry, not done yet. Will be done by EOD.")
            train_df = augment_oversample(train_df, image_root_dir, os.path.join(image_root_dir, "..", "temp"), oversampling_rate, pre_existing_images, seed)
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

    train_dataset = TreeDataset(train_df, image_cache_dir if oversampling_rate != 1 else image_root_dir, class_specification, mode)
    train_push_dataset = TreeDataset(train_push_df, image_root_dir, class_specification, mode)
    val_dataset = TreeDataset(val_df, image_root_dir, class_specification, mode)
    test_dataset = TreeDataset(test_df, image_root_dir, class_specification, mode)

    train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=4, pin_memory=False, collate_fn=collate_fn)
    
    train_push_loader = DataLoader(
            train_push_dataset, batch_size=train_push_batch_size, shuffle=True,
            num_workers=4, pin_memory=False, collate_fn=collate_fn)
    
    val_loader = DataLoader(
            val_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False, collate_fn=collate_fn)
    
    test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False, collate_fn=collate_fn)  

    return train_loader, train_push_loader, val_loader, test_loader

def get_dataloaders(cfg, log):
    log("Getting Dataloaders")

    return create_tree_dataloaders(
        source=cfg.DATASET.DATA_FILE,
        image_root_dir=cfg.DATASET.IMAGE_PATH,
        image_cache_dir=cfg.DATASET.AUGMENTED_IMAGE_PATH,
        gen_aug_params=cfg.DATASET.GENETIC_AUGMENTATION,
        train_val_test_split=cfg.DATASET.TRAIN_VAL_TEST_SPLIT,
        oversampling_rate=cfg.DATASET.OVERSAMPLING_RATE,
        train_not_classified_proportions=cfg.DATASET.TRAIN_NOT_CLASSIFIED_PROPORTIONS,
        tree_specification_file=cfg.DATASET.TREE_SPECIFICATION_FILE,
        pre_existing_images=cfg.DATASET.PRE_EXISTING_IMAGES,
        cached_dataset_folder=cfg.DATASET.CACHED_DATASET_FOLDER,
        cached_dataset_root=cfg.DATASET.CACHED_DATASET_ROOT,
        run_name=cfg.RUN_NAME,
        train_batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
        train_push_batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE,
        test_batch_size=cfg.DATASET.TEST_BATCH_SIZE,
        mode=cfg.DATASET.MODE,
        seed=cfg.SEED,
        log=log
    )

class GeneticDataset(Dataset):
    
    """
        A dataset class for the BIOSCAN genetic data. Samples are unpadded strings of nucleotides, including base pairs A, C, G, T and an unknown character N.

        Args:
            source (str): The path to the dataset file (csv or tsv).
            transform (callable, optional): Optional transforms to be applied to the genetic data. Default is None.
            drop_level (str): If supplied, the dataset will drop all rows where the given taxonomy level is not present. Default is None.
            allowed_classes ([(level, [class])]): If supplied, the dataset will only include rows where the given taxonomy level is within the given list of classes. Default is None. Use for validation and test sets.
            one_label (str): If supplied, the label will be the value of one_class
            classes: list[str]
            restraint: ex: ("family", ["Cecidomyiidae"])
            
        Returns:
            (genetics, label): A tuple containing the genetic data and the label (phylum, class, order, family, subfamily, tribe, genus, species, subspecies)
    """

    def __init__(self,
                 datapath: str,
                 transform='onehot',
                 level: str = None,
                 classes: list = None,
                 restraint: tuple = None,
                 max_class_count = 40,
                 log=print
        ):
        
        self.data = pd.read_csv(datapath, sep="\t")
        self.level = level
        
        if transform == 'onehot':
            self.transform = GeneticOneHot(length=720, zero_encode_unknown=True, include_height_channel=True)
        else:
            self.transform = None

        self.taxnomy_level = ["phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"]

        if self.level:
            if not self.level in self.taxnomy_level:
                raise ValueError(f"drop_level must be one of {self.taxnomy_level}")
            self.data = self.data[self.data[self.level] != "not_classified"]

        if classes:
            self.classes = {
                c: i for i,c in enumerate(classes)
            }
            self.data = self.data[self.data[self.level].isin(classes)]

            if restraint:
                log("Classes supplied with restraint. Restraints ignored.")
        
        else:
            if restraint:
                self.data = self.data[self.data[restraint[0]].isin(restraint[1])]

            classes, sizes = self.get_classes(level)

            if len(classes) > max_class_count:
                # Sort by size and take the top max_class_count
                sizes = np.array(sizes)
                classes = np.array(classes)
                classes = classes[sizes.argsort()[-max_class_count:]]
                self.data = self.data[self.data[self.level].isin(classes)]
                classes, sizes = self.get_classes(level)

            self.classes = {
                c: i for i,c in enumerate(classes)
            }
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        genetics = row["nucraw"]
        label = [row[c] for c in self.taxnomy_level]

        if self.transform:
            genetics = self.transform(genetics)

        label = label[self.taxnomy_level.index(self.level)]
        label = torch.tensor(self.classes[label])
            
        return genetics, label
    
    def __len__(self):
        return len(self.data)
    
    def get_classes(self, class_name: str):
        """Get a tuple of the list of the unique classes in the dataset, and their sizes for a given class name, e.x. order."""
        classes = self.data[class_name].unique()
        class_sizes = self.data[class_name].value_counts()

        return list(classes), list(class_sizes[classes])
