#!/usr/bin/env bash

#SBATCH --job-name=150_Lite_GenFullTest_Weights_40_NEW      # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/prune_fix/150_Lite_GenFullTest_Weights_40_NEW_%j.out

eval "$(conda shell.bash hook)" &&
conda activate intnn &&
python3 main.py --configs configs/genetics.yaml
