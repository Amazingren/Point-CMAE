#!/bin/bash
#SBATCH --job-name=partseg
#SBATCH --nodelist=gcp-eu-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100-40g:1
#SBATCH --output=./joblogs/cmae_partseg.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/cmae_partseg.error     # Redirect stderr to a separate error log file

# cuda
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.3/bin:$PATH

# gcc
export PATH=/opt/modules/gcc-10.5.0/bin:$PATH

# envs, you can use micromamba or conda
source ~/.bashrc
micromamba activate point

cd /path_to/Point-CMAE

python ../main.py \
    --root /path_to_your_dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
    --log_dir ./cmae_partseg \
    --ckpts ../experiments/path_to_your_pretrained_ckpts/ckpt.pth