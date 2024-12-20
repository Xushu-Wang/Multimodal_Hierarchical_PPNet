# Multimodal_Hierarchical_PPNet

## Configuring Datasets Folder
Create a datasets folder one level outside of this directory (otherwise VSCode will attempt to index this massive folder making for an unbearable editing experience).

Navigate to the datasets folder on slurm and run the following commands
```
ln -s /usr/project/xtmp/xw214/augmented_images augmented_images
ln -s /usr/project/xtmp/xw214/source_files source_files
ln -s /usr/project/xtmp/xw214/full_bioscan_images full_bioscan_images
```

## Using Datasets
The dataset will return (genetics_tensor,image_tensor), label.

The dataloader will return (genetics_tensors, image_tensors), labels

If mode is 2 or 3, you'll get (genetic_tensors, None), labels or (None, image_tensors), labels.

## Configuring Output Folder
From the project directory run 
```
ln -s /usr/project/xtmp/xw214/output ../output
```

Two files you'll rather run 
build class tree asks you questions about your tree structure and will generate a tree fitting your parameters. 
All trees are stored in class_trees directory. 

main runs the script, which grabs stuff from config files.  
