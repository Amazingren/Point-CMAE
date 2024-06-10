#!/bin/bash
#SBATCH --job-name=ab_DM_cos_6_reg_objbg_ep275_cls
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --output=./joblogs/ab_DM_cos_6_reg_objbg_ep275_cls.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/ab_DM_cos_6_reg_objbg_ep275_cls.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs
source ~/.bashrc
micromamba activate points

cd /home/bin_ren/projects/pointcloud/pcd_cluster/devs_mae/ab_DM_cos_6_reg

python main.py \
    --config cfgs/finetune_scan_objbg.yaml \
    --finetune_model \
    --exp_name ab_DM_cos_6_reg_objbg_ep275_cls \
    --ckpts experiments/pretrain/cfgs/ab_DM_cos_6_reg/ckpt-epoch-275.pth \
    --seed 0