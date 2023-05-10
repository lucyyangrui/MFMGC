#!/bin/bash
#SBATCH -J yxr
#SBATCH -N 1
#SBATCH --output=/public/home/robertchen/yxr20214227065/multi-modal-mgc/movie-genres-classification-multimodal-master/log/slurm.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6

source activate yxr20214227065-mgc

python train.py --modals video audio --model_name base6 --dataset_name moviebricks
# sleep 10s
# python train.py --modals summary video audio poster --model_name base5 --dataset_name moviebricks
