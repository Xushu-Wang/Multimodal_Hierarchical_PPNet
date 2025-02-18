import torch
from torch import Tensor
import numpy as np
import Augmentor
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Tuple
from torchvision.transforms import transforms, Compose, Normalize 
from yacs.config import CfgNode

class CustomTransform(ABC): 

    def __init__(self): 
      pass 
    
    @abstractmethod
    def __call__(self, input) -> Any: 
        """Abstract method that must be overwritten by child class methods."""
        pass

class GeneticOneHot(CustomTransform):
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
        super().__init__()

    def __call__(self, genetic_string: str) -> Tensor:
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

class ImageGeometricTransform(CustomTransform): 
    def __init__(self):
        super().__init__()

    def __call__(self, image) -> Tensor:
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

class GeneticMutationTransform(CustomTransform): 
    def __init__(self, insertion_amount: int, deletion_amount: int, substitution_rate: float): 
        self.insertion_amount = insertion_amount 
        self.deletion_amount = deletion_amount 
        self.substitution_rate = substitution_rate
        super().__init__()

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

def create_transforms(
    transform_mean: Tuple[float, float, float], 
    transform_std: Tuple[float, float, float], 
    gen_aug_params: CfgNode
) -> Tuple[Compose, GeneticMutationTransform, Compose, Compose, Normalize]: 

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

    return augmented_img_transforms, \
        augmented_genetic_transforms, \
        push_img_transforms, \
        img_transforms, \
        normalize
  
