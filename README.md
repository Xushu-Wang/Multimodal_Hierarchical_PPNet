# Multimodal_Hierarchical_PPNet

## Configuring Datasets Folder
Navigate to the datasets folder on slurm and run the following commands:
```
ln -s /usr/project/xtmp/xw214/augmented_images augmented_images
ln -s /usr/project/xtmp/xw214/source_files source_files
ln -s /usr/project/xtmp/xw214/full_bioscan_images full_bioscan_images
```

## Using Datasets
If the dataset is in mode 1 (genetics) or mode 2 (image), it will return (tensor, label), otherwise it will return ((genetics_tensor,image_tensor), label)
