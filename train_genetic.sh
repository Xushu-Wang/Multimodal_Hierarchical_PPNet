#!/usr/bin/env bash

#SBATCH --job-name=gen_no_global_ce_massive     # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/global_ce/gen_no_global_ce_massive_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python3 main.py --configs configs/genetics.yaml