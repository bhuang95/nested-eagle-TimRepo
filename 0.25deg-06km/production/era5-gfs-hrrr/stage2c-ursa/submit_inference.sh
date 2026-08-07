#!/bin/bash

#SBATCH -J nested-inference
#SBATCH -o logs/inference.%j.out
#SBATCH -e logs/inference.%j.out
#SBATCH --partition=u1-h100
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --qos=gpu
#SBATCH --account=gpu-ai4wp
#SBATCH -t 06:00:00


set -x

source ~/loadMods/load_conda.sh

conda activate eagle

export SCRATCH=/scratch4/BMC/gsienkf/Bo.Huang/expRuns/ML/
cd /scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c

srun eagle-tools inference inference.validation.yaml
echo "Done inference-validation"

srun eagle-tools inference inference.testing.yaml
echo "Done inference-testing"
