#!/usr/bin/env bash

#SBATCH --job-name=image_species_new_backbone     # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                  # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=logs/species/image_species_new_backbone_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python3 main.py --configs configs/image_species.yaml