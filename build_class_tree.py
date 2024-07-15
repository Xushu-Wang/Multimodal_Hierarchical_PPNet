"""
This utility script asks some questions and generates a class tree.
"""

import pandas as pd
import argparse
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("source", type=str, help="Source file for the class tree")

args = argparser.parse_args()

df = pd.read_csv(args.source, sep="\t")

levels = ["order", "family", "genus", "species"]

def question_level(levels, data, parent=None):
    if len(levels) == 0:
        return
    if not len(data[data[levels[0]] != "not_classified"]):
        print("No samples left. Adding not_classified to the tree.")
        return None

    option_count = len(data[data[levels[0]] != "not_classified"][levels[0]].unique())

    if parent:
        par_string = f" (parent: {parent})"
    else:
        par_string = ""
    
    while True:
        for val, count in data[data[levels[0]] != "not_classified"][levels[0]].value_counts()[:6].items():
            print(f"{val: <16}\t{count} samples")

        count = input(f"How many of the largest {levels[0]}{par_string} would you like to include? (max {option_count}) ")

        try:
            count = int(count)
        except ValueError:
            print("")
            continue

        if count <= option_count and count > 0:
            break
    
    # while True:
    #     include_not_classified = input(f"Would you like to include not_classified {levels[0]}? ([y]/n) ")
    #     if not include_not_classified:
    #         include_not_classified = "y"
    #     if include_not_classified in ["y", "n"]:
    #         break

    options = data[data[levels[0]] != "not_classified"][levels[0]].value_counts().index[:count]
    # if include_not_classified == "y":
    #     options = list(options) + ["not_classified"]

    print(f"Selected: {', '.join(options)}\n")

    tree = {}

    for o in options:
        if o == "not_classified":
            tree[o] = None
            continue
        tree[o] = question_level(levels[1:], data[data[levels[0]] == o], o)

    return tree

tree = question_level(levels,df)

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
            json.dump(out, f)
        break
    except FileNotFoundError:
        print("Invalid path")
        continue