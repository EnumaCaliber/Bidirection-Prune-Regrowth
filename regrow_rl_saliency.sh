#!/bin/bash -l
#SBATCH -p Quick # run on partition general
#SBATCH --cpus-per-task=32 # 32 CPUs per task
#SBATCH --mem=10GB # 100GB per task
#SBATCH --gpus=1 # 1 GPUs
#SBATCH --mail-user=junchen@usf.edu # email for notifications
#SBATCH --output=./regrow_saliency/vgg/solu26_eps500_step2_from99_patience30_resume%j.out

# --- Activate conda ---
# module load anaconda             # (sometimes needed on clusters)
source ~/.bashrc                 # makes 'conda' command available
conda activate gap           # replace with your env name

export WANDB_API_KEY=c6d62686d813356615f8fabeaf82ce47ae84ede6


# python rl_regrowth_nas.py  --regrow_step 0.01 --learning_rate 5e-3 --num_epochs 500 --batch_size 1 --save_dir ./rl_regrow_savedir/resnet/solu23_eps500_step1_w_classifier_from95 --m_name resnet20 --starting_checkpoint oneshot --regrow_iterations 1 --reference_model ./resnet20/ckpt_after_prune/pruned_finetuned_mask_0.94.pth --save_freq 1
python rl_saliency_regrowth.py  --regrow_step 0.02 --learning_rate 5e-3 --num_epochs 500 --batch_size 1 --save_dir ./rl_saliency_regrow_savedir/vgg/solu26_eps500_step2_from99_patience30 --m_name vgg16 --save_freq 1 --resume /home/j/junchen/DAC26_Final/rl_saliency_regrow_savedir/vgg/solu26_eps500_step2_from99_patience30/saliency_rl_checkpoint_epoch_113.pth



python rl_saliency_regrowth.py  --regrow_step 0.02 --learning_rate 5e-3 --num_epochs 500 --batch_size 1 --save_dir ./rl_saliency_regrow_savedir/vgg/solu26_eps500_step2_from99_patience30 --m_name vgg16 --save_freq 1 