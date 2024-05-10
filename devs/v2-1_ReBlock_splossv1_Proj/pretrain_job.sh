#!/bin/bash
#SBATCH --job-name=v2-1_spproj
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/v2-1_spproj.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/v2-1_spproj.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs/v2-1_ReBlock_splossv1_Proj

python main.py \
    --config cfgs/pretrain/base.yaml \
    --exp_name v2-1_ReBlock_splossv1_Proj