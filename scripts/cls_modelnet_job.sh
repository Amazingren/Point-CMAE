#!/bin/bash
#SBATCH --job-name=finetune_modelnet
#SBATCH --nodelist=gcp-eu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --output=./joblogs/finetune_modelnet.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/finetune_modelnet.error     # Redirect stderr to a separate error log file


# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# envs, you can use micromamba or conda
source ~/.bashrc
micromamba activate points

cd /path_to/Point-CMAE

python ../main.py \
    --config cfgs/finetune_modelnet.yaml \
    --finetune_model \
    --exp_name finetune_modelnet \
    --ckpts experiments/path_to_your_pretrained_ckpts/ckpt.pth \
    --seed 0