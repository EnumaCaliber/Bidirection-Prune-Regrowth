#!/bin/bash -l
#SBATCH -p Quick # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=10GB # 100GB per task
#SBATCH --gpus=1 # 1 GPUs
#SBATCH --mail-user=junchen@usf.edu # email for notifications
#SBATCH --output=./prune/resnet/sparsity985.out

# --- Activate conda ---
# module load anaconda             # (sometimes needed on clusters)
source ~/.bashrc                 # makes 'conda' command available
conda activate gap           # replace with your env name



python main.py --m_name resnet20 --pruner lamp --resume
# python main.py --m_name alexnet --pruner lamp --resume
# python main.py --m_name densenet --pruner lamp --resume
# python main.py --m_name googlenet --pruner lamp --resume
# python main.py --m_name vgg16 --pruner lamp --resume