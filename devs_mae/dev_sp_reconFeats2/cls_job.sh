#!/bin/bash
#SBATCH --job-name=dev_sp_reconFeats2_cls
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/dev_sp_reconFeats2_cls_ep200.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/dev_sp_reconFeats2_cls_ep200.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_sp_reconFeats2

python main.py \
    --config cfgs/finetune_scan_hardest.yaml \
    --finetune_model \
    --exp_name dev_sp_reconFeats2_ep200 \
    --ckpts experiments/pretrain/cfgs/dev_sp_reconFeats2/ckpt-epoch-200.pth \
    --seed 0