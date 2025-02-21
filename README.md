# Multimodal_Hierarchical_PPNet

Create the following symlinks from within the repo. Note that `output` and `datasets` should be one directory above. 
```
ln -s /usr/project/xtmp/cjb131/cs474/datasets ../datasets
ln -s /usr/project/xtmp/cjb131/cs474/pre_existing_datasets pre_existing_datasets
ln -s /usr/project/xtmp/cjb131/cs474/backbones backbones
ln -s /usr/project/xtmp/cjb131/cs474/output ../output
ln -s /usr/project/xtmp/cjb131/cs474/logs logs
```

The following subsections describes the entire process in training a ProtoPNet. 

## Dataset 

The BIOSCAN-1M dataset consists over 1 million samples of data on different insects (stored in `../datasets/`). More specifically, each sample consists of the following. 
1. A variable-size image of the insect. This is stored in `../datasets/full_bioscan_images/`, and the full filepath of each image is stored in `../datasets/source_files/metadata_cleaned_permissive.tsv`. 
2. A variable-size genetic sequence of the insect. This is stored in `../datasets/source_files/metadata_cleaned_permissive.tsv`. 
3. The full taxonomy of the insect, in the sequence of Class, Order, Family, Genus, Species, Subspecies. However, many of the values are missing. This is stored in `../datasets/source_files/metadata_cleaned_permissive.tsv`. 

The dataset is extremely unbalanced with the majority of samples having no species/subspecies classifications. Considering the taxonomy as a tree, we first focus on the Order, Family, Genus, Species depths, and for each node we look at the set of samples that land on this node. If there are not enough samples or the samples are too unbalanced, we throw away this node completely. By doing this, we can construct a final pruned tree, saved in `class_trees/`. Each json file represents the method in which we filter the dataset, and this process is done in `build_class_tree.py`. The final filtered dataset consists of 10k~20k samples.  

This json file is loaded in as `Hierarchy` class consisting of a tree $\mathcal{T} = (\mathcal{T}_\alpha)$  of `TaxNode`'s (Taxonomy Nodes). The metadata regarding the depth of the tree, the numerical indices of each taxonomy depth, and the number of samples at each depth, is stored in the `Level`. This `Hierarchy` object, plus the dataframe of the filtered dataset, is passed to instantiate a `torch.Dataset`. Given a numerical index $\alpha$, let us label $\mathcal{T}_\alpha$ as the `TaxNode` objected at location $\alpha$ in $\mathcal{T}$. 

Every time we load a `torch.Dataset` we can look at `metadata_cleaned_permissive.tsv` and retrieve the subdataset on here. However, even opening the full tsv file tends to be memory intensive, and so when we load this for the first time we have the option to cache this into another tsv file, stored in `pre_existing_datasets`. In general, the dataset will return a sample of the form `(genetics, image), label`. 

## Backbones 

The next step is to look for the CNN backbone architecture for your ProtoPNet. For images, we can download a pretrained resnet backbone, or we can create one from scratch and train it ourselves. For genetics, there is not much literature on inference on genetic sequences, so all of our backbones are created from scratch. These are done by calling `train_blackbox.py`, which saves the model weights in `backbones/`. Let's label $CNN_{gen}$ and $CNN_{img}$ as the backbones for genetics and images. 

## Model

Let's consider a uni-modal hierarchical protopnet (hppnet), labeled $h$. It should consist of the backbone layer we just described, plus a set of prototypes for each node in our `Hierarchy` tree. This is exactly the two arguments that we pass when instantiating a `HierProtoPNet` object. It stores a reference to the backbone, and as for the prototypes, it constructs a *new* tree of `ProtoNode`s (Prototype Nodes) $\mathcal{P}$ with the same indexing as $\mathcal{T}$. The structure of this tree is the same as that of `Hierarchy`, but it contains prototypes and supports other attributes/methods for `nn.Module`s. Therefore, we can consider 
$$
    h = (CNN, \mathcal{P}), \qquad h(x) = (\mathcal{P}_\alpha (CNN(x))_\alpha$ 
$$
        as the forward pass of the network. 

A multi-modal hierarchical protopnet (mhppnet) is a wrapper class around two hppnets. The difference is that we want to use both the genetics and image prototypes to predict the taxonomy of a sample. Therefore, we can consider the joint model as 

$$ 
    m = (h_{gen}, h_{img}) = ((CNN_{gen}, \mathcal{P}_{gen}), (CNN_{img}, \mathcal{P}_{img})), \qquad m(x) = (h_{gen} (x), h_{img}(x))
$$  

This is really just two hppnets fitting to the data independently, and once we get the logits from the forward pass of each we sum them together before softmaxing. (This approach should be improved. This might be too naive). The linkage between these models is the correspondence loss, which requires that the activations are good. 

