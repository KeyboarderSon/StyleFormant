#!/bin/bash

#SBATCH --job-name=tts
#SBATCH --gres=gpu:2
#SBATCH -o slurmhalf2.out
#SBATCH --time=3-0
. /data/joohye/anaconda3/etc/profile.d/conda.sh
conda activate torch37
python train.py -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml