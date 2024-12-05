#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 #request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00 # 3 hours
#SBATCH --output=../output/job-%j/job.out
#SBATCH --account=fall2024-comp551
#SBATCH -e ../output/job-%j/job.err # STDERR

# Access the job number
job_number=$SLURM_JOB_ID

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

echo "BERT Test Job" > ../output/job-$job_number/test_out.txt
nvidia-smi >> ../output/job-$job_number/test_out.txt
python3 ../python/BERT_experiments/over_hyperparam.py ../output/job-$job_number "head+1" "batch_size" "8,16,32,64,128,256,512" #insert python script here

