#!/usr/bin/env bash

#SBATCH --job-name=2.0_10_oversample_no_global     # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=logs/corr4/2.0_10_oversample_no_global_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python3 new_main.py --configs configs/multi.yaml