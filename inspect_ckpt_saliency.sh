#!/bin/bash -l
#SBATCH -p Quick # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=10GB # 100GB per task
#SBATCH --gpus=1 # 1 GPUs
#SBATCH --mail-user=junchen@usf.edu # email for notifications
#SBATCH --output=./regrow_saliency/vgg/solu26_step4_from99_patience30/eps30_%j.out

# --- Activate conda ---
# module load anaconda             # (sometimes needed on clusters)
source ~/.bashrc                 # makes 'conda' command available
conda activate gap           # replace with your env name

# python inspect_checkpoint_flexible_init.py --file /home/j/junchen/DAC26_Final/rl_saliency_regrow_savedir/vgg/solu24_eps500_step05_w_classifier_from995_update_saliency/saliency_rl_checkpoint_epoch_146.pth --type checkpoint --finetune --model vgg16 --save_dir ./inspection_saliency/vgg/solu24_step05_init --init_strategy dual_lottery --reinit_all

# python inspect_checkpoint.py --file /home/j/junchen/DAC26_Final/rl_saliency_regrow_savedir/vgg/solu25_eps500_step05_from995_patience30/saliency_rl_checkpoint_epoch_65.pth --type checkpoint --finetune --model vgg16 --save_dir ./inspection_saliency/vgg/solu25_step05 --patience 30 --seed 42

python inspect_checkpoint.py --file /home/j/junchen/DAC26_Final/rl_saliency_regrow_savedir/vgg/solu26_eps500_step4_from99_patience30/saliency_rl_checkpoint_epoch_30.pth --type checkpoint --finetune --model vgg16 --save_dir /data/junchen/inspection_saliency/vgg/solu26_step4_from99 --patience 30
