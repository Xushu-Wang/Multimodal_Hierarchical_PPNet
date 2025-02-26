# Multimodal_Hierarchical_PPNet

## Configuring Datasets Folder
Create the following symlinks from within the repo. Note that `output` and `datasets` should be one directory above. 
```
ln -s /usr/project/xtmp/cjb131/cs474/datasets ../datasets
ln -s /usr/project/xtmp/cjb131/cs474/output ../output
ln -s /usr/project/xtmp/cjb131/cs474/backbones backbones
ln -s /usr/project/xtmp/cjb131/cs474/logs logs
ln -s /usr/project/xtmp/cjb131/cs474/pre_existing_datasets pre_existing_datasets
```

## Using Datasets
The dataset will return (genetics_tensor,image_tensor), label.

The dataloader will return (genetics_tensors, image_tensors), labels

If mode is 2 or 3, you'll get (genetic_tensors, None), labels or (None, image_tensors), labels.

## Configuring Output Folder

Two files you'll rather run 
build class tree asks you questions about your tree structure and will generate a tree fitting your parameters. 
All trees are stored in class_trees directory. 

main runs the script, which grabs stuff from config files.  
