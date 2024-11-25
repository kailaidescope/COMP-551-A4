#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00 # 5 mins
#SBATCH --output=../output/job-%j/job.out
#SBATCH --account=fall2024-comp551

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024
source ~/.bashrc
conda activate test_environment #environment

echo "Test" > ../output/job-%j/test_out.txt
nvidia-smi >> ../output/job-%j/test_out.txt
pip3 list >> ../output/job-%j/pip3_list.txt
pip list >> ../output/job-%j/pip_list.txt
conda list >> ../output/job-%j/conda_list.txt

