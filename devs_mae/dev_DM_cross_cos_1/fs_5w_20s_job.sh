#!/bin/bash
#SBATCH --job-name=fs_modelnet40_5way_20shot_f9
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/fs_modelnet40_5way_20shot_f9.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/fs_modelnet40_5way_20shot_f9.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_DM_cross_cos

python main.py \
    --config cfgs/fewshot.yaml \
    --finetune_model \
    --ckpts experiments/pretrain/cfgs/dev_DM_cross_cos/ckpt-epoch-275.pth \
    --exp_name fs_modelnet40_5way_20shot_f9 \
    --way 5 \
    --shot 20 \
    --fold 9
