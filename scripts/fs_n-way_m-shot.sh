#!/bin/bash
#SBATCH --job-name=fs_modelnet40_5way_10shot_f9
#SBATCH --nodelist=gcp-eu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --output=./joblogs/fs_modelnet40_5way_10shot_f9.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/fs_modelnet40_5way_10shot_f9.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3.0/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs, you can use `micromamba` or `conda`
source ~/.bashrc
micromamba activate points

cd /path_to/Point-CMAE

# You need to change the fold number to: 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 for different folds
# You can also change the `--way`: 5 or 10
# You can also change the `--shot`: 10 or 20 
python ../main.py \
    --config cfgs/fewshot.yaml \
    --finetune_model \
    --ckpts experiments/path_to_your_pretrained_ckpts/ckpt.pth \
    --exp_name fs_modelnet40_5way_10shot_f9 \
    --way 5 \
    --shot 10 \
    --fold 9
