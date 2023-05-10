#!/bin/bash
#SBATCH -J yxr
#SBATCH -N 1
#SBATCH --output=/public/home/robertchen/yxr20214227065/multi-modal-mgc/movie-genres-classification-multimodal-master/slurm.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6

source activate yxr20214227065-mgc


# echo "================ lr"
# pretrain_lr=(5e-4 5e-3 5e-2)
# other_lt=(5e-3 5e-2)
# hidden_size=(64 128 256 512)
# for element in ${pretrain_lr[@]}
# do
# echo "================ pre-lr=" $element
# python train.py --model_name 'mymodel' --pretrain_model_lr $element --modals summary poster
# sleep 10s
# done

# for element in ${other_lt[@]}
# do
# echo "================ oth-lr=" $element
# python train.py --model_name 'mymodel' --other_model_lr $element --modals summary poster
# sleep 10s
# done

# for element in ${hidden_size[@]}
# do
# echo "================ hidden_size=" $element
# python train.py --model_name 'mymodel' --modals summary poster --hidden_size $element
# sleep 10s
# done

python train.py --model_name 'mymodel' --modals summary poster --other_model_lr 5e-5