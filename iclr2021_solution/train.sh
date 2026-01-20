#!/bin/bash

#SBATCH -p Quick # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=120GB # 100GB per task
#SBATCH --gpus=1 # 1 GPUs
#SBATCH --mail-user=junchen@usf.edu # email for notifications


# Activate virtual environment (if using one)
source ~/.bashrc  
conda activate gap

# Run your script
echo "[START] - Start Pruning Model"

srun python iterate.py --model alexnet --pruner lamp


echo "[FINISH] - Finish Pruning Model"