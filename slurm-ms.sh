#!/bin/bash
#SBATCH -J yxr-ms
#SBATCH -N 1
#SBATCH --output=/public/home/robertchen/yxr20214227065/multi-modal-mgc/movie-genres-classification-multimodal-master/log/slurm-ms.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6

# python train.py --modals summary --model_name base3 --dataset_name moviescope
# sleep 10s
# python train.py --modals poster --model_name base3 --dataset_name moviescope
# sleep 10s

# python train.py --modals summary --model_name base4 --dataset_name moviescope
# sleep 10s
# python train.py --modals summary --model_name base5 --dataset_name moviescope
# sleep 10s
# python train.py --modals poster --model_name base5 --dataset_name moviescope
# sleep 10s
# python train.py --modals audio --model_name base5 --dataset_name moviebricks
# sleep 10s
python train.py --modals video audio --model_name base6 --dataset_name moviescope
