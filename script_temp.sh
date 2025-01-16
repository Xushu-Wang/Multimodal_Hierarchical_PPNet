#!/usr/bin/env bash

#SBATCH --job-name=dirty # Job name
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb                  # Job memory request
#SBATCH --time=15:00:00                # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:2
#SBATCH --output=logs/dirty_%j.out

eval "$(conda shell.bash hook)" 
conda activate intnn
python main.py