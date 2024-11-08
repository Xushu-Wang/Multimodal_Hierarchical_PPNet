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
argparser.add_argument("--min-leaves", type=int, help="Minimum number of leaves for a classification task", default=2)
argparser.add_argument("--default-count", type=int, help="Default count to avoid answering a bunch of questions manually. -1 indicates manual selection. 0 indicates choose all options. ", default=0)
argparser.add_argument("--take-max", action="store_true", help="Take the maximum number of samples for each level")

args = argparser.parse_args()

if(args.min_samples < 3):
    print("Minimum number of samples must be at least 3")
    exit()

if(args.min_leaves < 2):
    print("Minimum number of leaves must be at least 2")
    exit()

print("Opening source file... This may take a minute. ")

df = pd.read_csv(args.source, sep="\t")

print("Source file opened.")

levels = ["order", "family", "genus", "species"]

def question_level(levels: List[str], data: pd.DataFrame, parent="", min_samples: int = 3, default_count: int = 0, take_max=False) -> Optional[Dict]: 
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
    level_series = level_series[level_series != "not_classified"]

    # make sure that there are viable classes for most general level
    if len(level_series) == 0: 
        return None 


    # get number of viable top level branches and filter them if below min_sample
    top_level_counts = level_series.value_counts() 
    top_level_counts= top_level_counts[top_level_counts > min_samples].sort_values(ascending=False)
    
    option_count = len(top_level_counts)

    if option_count == 0 or option_count < args.min_leaves: 
        return 

    par_string = f"(parent: {parent})"

    print(top_level_counts) # display the general level counts for user 
    
    while True:
        if default_count == -1: 
            count = option_count if take_max else input(f"How many of the largest {general_level} {par_string} would you like to include? (max {option_count}) ")
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
    
    options = list(top_level_counts.index[:count])
    print(f"Selected: {options}\n")

    tree = {}

    for o in options:
        if o == "not_classified":
            tree[o] = None
            continue
        tree[o] = question_level(levels[1:], data[data[general_level] == o], o, min_samples, take_max=take_max)

    return tree

tree = question_level(levels,df, min_samples = args.min_samples, default_count = args.default_count, take_max=args.take_max)

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
