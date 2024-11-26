#!/usr/bin/env bash

#SBATCH --job-name=z_accu_single     # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/species/z_accu_single_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python get_species_accuracy.py