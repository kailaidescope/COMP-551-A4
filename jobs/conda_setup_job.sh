#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00 # 1 hour
#SBATCH --output=%N-%j.out
#SBATCH --account=fall2024-comp551

module load miniconda/miniconda-fall2024
conda init bash

echo "Conda setup" > conda_setup.txt