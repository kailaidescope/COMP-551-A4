#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00 # 1 hour
#SBATCH --output=../output/job-%j/job.out
#SBATCH --account=fall2024-comp551
#SBATCH -e ../output/job-%j/job.err # STDERR

# Access the job number
job_number=$SLURM_JOB_ID

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

echo "BERT Test Job" > ../output/job-$job_number/test_out.txt
nvidia-smi >> ../output/job-$job_number/test_out.txt
python3 ../python/BERT_evaluation_script_2.py ../output/job-2824/7-epoch-head.pth ../output/job-$job_number/confusion_matrix.png #insert python script here

