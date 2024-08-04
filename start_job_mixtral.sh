#!/usr/bin/env bash
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --gpus=a100:2
#SBATCH --output=log.%x.%j.out
#SBATCH --job-name=test-batchJob
#SBATCH --partition=epyc-gpu-sxm
#SBATCH --time=30

module purge
module load anaconda/2023.09

conda activate nlp
python3 test_mixtral.py

