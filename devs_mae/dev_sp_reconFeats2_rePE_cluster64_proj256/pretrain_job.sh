#!/bin/bash
#SBATCH --job-name=dev_sp_reconFeats2_rePE_cluster64_proj256
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/dev_sp_reconFeats2_rePE_cluster64_proj256.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/dev_sp_reconFeats2_rePE_cluster64_proj256.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_sp_reconFeats2_rePE_cluster64_proj256

python main.py \
    --config cfgs/pretrain.yaml \
    --exp_name dev_sp_reconFeats2_rePE_cluster64_proj256