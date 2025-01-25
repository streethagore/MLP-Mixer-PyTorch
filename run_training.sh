#!/bin/bash
#SBATCH --job-name=mlpmixer
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Source the shell configuration file to apply changes (need for conda)
source /home/tau/strivaud/.bashrc

# activate the conda environment
conda activate gromo

# Run the training script
command="python train.py"
command+=" --batch_size 128"
command+=" --epochs 300"
command+=" --optimizer sgd"
command+=" --lr 1e-3"
command+=" --weight_decay 5e-5"
command+=" --num_workers 4"
command+=" --device cuda"
command+=" --augmentation randaugment"

eval "$command"