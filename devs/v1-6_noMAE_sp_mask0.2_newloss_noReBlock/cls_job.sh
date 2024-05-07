#!/bin/bash
#SBATCH --job-name=v1-6_cls
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/v1-6_cls.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/v1-6_cls.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs/v1-6_noMAE_sp_mask0_newloss_noReBlock

python main.py \
    --config cfgs/full/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name v1-6_noMAE_sp_mask0_newloss_noReBlock_ep100 \
    --ckpts experiments/base/pretrain/v1-6_noMAE_sp_mask0_newloss_noReBlock/ckpt-epoch-100.pth \
    --seed 0