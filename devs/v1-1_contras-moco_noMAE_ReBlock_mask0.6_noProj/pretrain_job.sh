#!/bin/bash
#SBATCH --job-name=v1-1
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/v1-1_noProj.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/v1-1_noProj.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs/v1-1_contras-moco_noMAE_ReBlock_mask0.6_noProj

python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name v1-1_contras-moco_noMAE_ReBlock_mask0.6_noProj
