"""
This utility script asks some questions and generates a class tree.
"""
from typing import Optional, Dict, List
import pandas as pd
import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("--source", type=str, help="Source file for the class tree. Should be a tsv file. ", default="../datasets/source_files/metadata_cleaned_permissive.tsv")
argparser.add_argument("--min-samples", type=int, help="Minimum number of samples to include in the tree", default=3)
argparser.add_argument("--min-leaves", type=int, help="Minimum number of leaves for a classification task", default=1)
argparser.add_argument("--default-count", type=int, help="Default count to avoid answering a bunch of questions manually. -1 indicates manual selection. 0 indicates choose all options. ", default=0)
argparser.add_argument("--only-species-leaves", action="store_true", help="Only allow species nodes to be leaves")

args = argparser.parse_args()

if(args.min_samples < 3):
    print("Minimum number of samples should be at least 3")

print("Opening source file... This may take a minute. ")

df = pd.read_csv(args.source, sep="\t")

print("Source file opened.")

levels = ["order", "family", "genus", "species"]

def question_level(
    levels: List[str], 
    data: pd.DataFrame, 
    parent="", 
    min_samples: int = 3, 
    default_count: int = 0, 
    only_species_leaves=False
    ) -> Optional[Dict]: 

    """Recursively generate a tree json file from the dataframe. 
    Args: 
        levels: The list of levels in the tree, e.g. order, family, genus, species. 
                Should be in decreasing hierarchical order, i.e. most general comes first. 
        data: the genetics dataframe read from the tsv file
        parent: the head node of the tree
    """

    if len(levels) == 0:
        return None 

    general_level, *_ = levels 

    # focus on only the class labels of the parent level  
    level_series = data[general_level]
    level_series = pd.DataFrame(level_series[level_series != "not_classified"])

    # make sure that there are viable classes for most general level
    if len(level_series) == 0: 
        return None 


    # get number of viable top level branches and filter them if below min_sample
    top_level_counts = level_series.value_counts() 
    top_level_counts= pd.Series(top_level_counts[top_level_counts > min_samples]).sort_values(ascending=False)
    
    option_count = len(top_level_counts)

    if option_count == 0 or option_count < args.min_leaves: 
        return 

    par_string = f"(parent: {parent})"

    print(top_level_counts) # display the general level counts for user 
    
    while True:
        if default_count == -1: 
            count = input(f"How many of the largest {general_level} {par_string} would you like to include? (max {option_count}) ")
            try:
                count = int(count)
            except ValueError:
                continue
        elif default_count == 0: 
            count = option_count
        else:
            count = min(option_count, default_count)

        if 0 < count <= option_count: 
            break
    
    options = list(top_level_counts.index)[:count]
    print(f"Selected: {options}\n")

    tree = {}

    for o in options:
        if o == "not_classified":
            if not only_species_leaves or general_level == "species":
                tree[o] = None
            continue
        children = question_level(levels[1:], data[data[general_level] == o], o, min_samples, default_count=default_count, only_species_leaves=only_species_leaves)
        if children or not only_species_leaves or general_level == "species":
            tree[o] = children

    return tree

tree = question_level(levels,df, min_samples = args.min_samples, default_count = args.default_count, only_species_leaves=args.only_species_leaves)

def get_tree_stats(tree):
    from collections import defaultdict

    data = defaultdict(int)

    def dfs(start, i): 
        if start is None:
            data[i] += 1
            return 
        for _, val in start.items(): 
            dfs(val, i+1)

    dfs(tree, 0) 

    return data

while True:
    outpath = input("Output path: ")
    
    out = {
        "tree": tree,
        "levels": levels
    } 

    try:
        if not outpath.endswith(".json"):
            outpath += ".json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=4)
        break
    except FileNotFoundError:
        print("Invalid path")
        continue
