#!/bin/bash

#SBATCH -J nested-obs-metrics
#SBATCH -o logs/obs-metrics.%j.out
#SBATCH -e logs/obs-metrics.%j.out
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3936M
#SBATCH --qos=batch
#SBATCH --account=gsienkf
#SBATCH -t 24:00:00

set -x

source ~/loadMods/load_conda.sh

conda activate eagle

export SCRATCH=/scratch4/BMC/gsienkf/Bo.Huang/expRuns/ML/
cd /scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c

#echo "Running for HRRR over validation period"
#srun eagle-tools obs-metrics obs-metrics.hrrr.validation.yaml
#echo " ... Done"
echo "Running for global over validation period"
srun --ntasks=4 --cpus-per-task=16 eagle-tools obs-metrics obs-metrics.global.validation.yaml
echo " ... Done"

#echo "Running for HRRR over testing period"
#srun eagle-tools obs-metrics obs-metrics.hrrr.testing.yaml
#echo " ... Done"
echo "Running for global over testing period"
srun --ntasks=4 --cpus-per-task=16 eagle-tools obs-metrics obs-metrics.global.testing.yaml
echo " ... Done"
