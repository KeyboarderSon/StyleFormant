#!/bin/bash

#SBATCH --job-name=tts16_2
#SBATCH --gres=gpu:1
#SBATCH -o slurm_16_2.out
#SBATCH --time=4-0
. /data/joohye/anaconda3/etc/profile.d/conda.sh
conda activate torch37
python train.py -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml