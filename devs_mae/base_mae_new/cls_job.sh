#!/bin/bash
#SBATCH --job-name=mae_cls
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/mae_cls_ep300.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/mae_cls_ep300.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point


cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/base_mae

python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name mae_reproduce_ep300 \
    --ckpts experiments/pretrain/cfgs/mae_reproduce/ckpt-epoch-300.pth \
    --seed 0