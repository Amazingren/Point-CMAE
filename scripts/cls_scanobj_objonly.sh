#!/bin/bash
#SBATCH --job-name=finetune_scanobj_objonly
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --output=./joblogs/cmae_cls_scanobj_objonly.log   
#SBATCH --error=./joblogs/cmae_cls_scanobj_objonly.error

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate points

cd /path_to/Point-CMAE

python ../main.py \
    --config cfgs/finetune_scan_objonly.yaml \
    --finetune_model \
    --exp_name cmae_cls_scanobj_objonly \
    --ckpts experiments/path_to_your_pretrained_ckpts/ckpt.pth \
    --seed 0