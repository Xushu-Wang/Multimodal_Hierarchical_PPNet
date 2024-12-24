#!/usr/bin/env bash

#SBATCH --job-name=image_backbone_wd.0001     # Job name
#SBATCH --ntasks=1                     # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb                    # Job memory request
#SBATCH --time=96:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/species/image_backbone_wd.0001_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python3 train_blackbox.py --config configs/image_species.yaml --output backbones