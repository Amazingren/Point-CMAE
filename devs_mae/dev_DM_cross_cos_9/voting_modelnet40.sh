#!/bin/bash
#SBATCH --job-name=voting_modelnet
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/voting_modelnet.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/voting_modelnet.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate point

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/dev_DM_cross_cos

python main.py --test \
    --config cfgs/finetune_modelnet.yaml \
    --exp_name voting_modelnet \
    --ckpts experiments/finetune_modelnet/cfgs/dev_DM_cross_cos_ep275_modelnet_cls/ckpt-best.pth