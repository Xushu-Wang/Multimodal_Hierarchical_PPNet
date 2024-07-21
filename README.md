# Multimodal_Hierarchical_PPNet

## Configuring Datasets Folder
Navigate to the datasets folder on slurm and run the following commands:
```
ln -s /usr/project/xtmp/xw214/augmented_images augmented_images
ln -s /usr/project/xtmp/xw214/source_files source_files
ln -s /usr/project/xtmp/xw214/full_bioscan_images full_bioscan_images
```

## Using Datasets
The dataset will return (genetics_tensor,image_tensor), label.

The dataloader will return (genetics_tensors, image_tensors), labels

If mode is 2 or 3, you'll get (genetic_tensors, None), labels or (None, image_tensors), labels.