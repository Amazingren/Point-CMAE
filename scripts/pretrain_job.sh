#!/bin/bash
#SBATCH --job-name=pretrain_cmae
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --output=./joblogs/cmae_pretrain.log   
#SBATCH --error=./joblogs/cmae_pretrain.error

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs, you can use micromamba or conda
source ~/.bashrc
micromamba activate point

cd /path_to/Point-CMAE

python ../main.py \
    --config cfgs/pretrain.yaml \
    --exp_name cmae_pretrain \