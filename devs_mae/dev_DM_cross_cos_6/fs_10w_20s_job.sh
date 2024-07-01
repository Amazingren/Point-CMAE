#!/bin/bash
#SBATCH --job-name=fs_mlp3_modelnet40_10way_20shot_f9
#SBATCH --nodelist=gcp-eu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/fs_mlp3_modelnet40_10way_20shot_f9.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/fs_mlp3_modelnet40_10way_20shot_f9.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate points

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_DM_cross_cos

python main.py \
    --config cfgs/mlp3/fewshot.yaml \
    --finetune_model \
    --ckpts experiments/pretrain/cfgs/dev_DM_cross_cos/ckpt-epoch-275.pth \
    --exp_name fs_mlp3_modelnet40_10way_20shot_f9 \
    --way 10 \
    --shot 20 \
    --fold 9