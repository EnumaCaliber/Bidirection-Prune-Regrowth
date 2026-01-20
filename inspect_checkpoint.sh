#!/bin/bash -l
#SBATCH -p Quick # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=10GB # 100GB per task
#SBATCH --gpus=1 # 1 GPUs
#SBATCH --mail-user=junchen@usf.edu # email for notifications
#SBATCH --output=./regrow_nas/vgg/solu22_step45_from995/step45_best_calibBN_eps70_%j.out

# --- Activate conda ---
# module load anaconda             # (sometimes needed on clusters)
source ~/.bashrc                 # makes 'conda' command available
conda activate gap           # replace with your env name


# python inspect_checkpoint.py --file /home/j/junchen/DAC26_Final/rl_regrow_savedir/densenet/solu21_eps500_w_classifier/step05/window_best_epoch_10_densenet.pth --type best --finetune --model densenet --save_dir ./inspection_res/densenet/solu21_step05 --reset_bn_stats

# python inspect_checkpoint.py --file /home/j/junchen/DAC26_Final/rl_regrow_savedir/resnet/solu23_eps500_step1_w_classifier_from95/rl_training_checkpoint_epoch_1.pth --type best --finetune --model resnet20 --save_dir ./inspection_res/resnet/solu23_step1_from95 --reset_bn_stats

python inspect_checkpoint.py --file /home/j/junchen/DAC26_Final/rl_regrow_savedir/vgg/solu22_eps500_step45_w_classifier_from995/rl_training_checkpoint_epoch_70.pth --type best --finetune --model vgg16 --save_dir ./inspection_res/vgg/solu22_step45 --reset_bn_stats

# python inspect_checkpoint.py --file ./rl_regrow_savedir/alexnet/iterative_solution11_eps500/step1/best_allocation_alexnet.pth --type best --finetune --model alexnet --save_dir ./inspection_res/alexnet/step4 --epochs 200 --starting_checkpoint iterative