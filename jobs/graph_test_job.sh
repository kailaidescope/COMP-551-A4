#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00 # 5 mins
#SBATCH --output=../output/job-%j.out
#SBATCH --account=fall2024-comp551

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024
source ~/.bashrc
conda activate test_environment #environment

echo "Test" > ../output/test_out.txt
nvidia-smi >> ../output/test_out.txt
python3 ../python/test_graph_saving.py ../output #insert python script here

