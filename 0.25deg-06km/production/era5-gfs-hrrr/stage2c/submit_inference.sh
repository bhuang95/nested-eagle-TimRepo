#!/bin/bash

#SBATCH -J nested-inference
#SBATCH -o /pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/logs/inference.%j.out
#SBATCH -e /pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/logs/inference.%j.out
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint='gpu&hbm80g'
#SBATCH -t 3:00:00

set -x

module load conda

conda activate eagle

cd /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c

# conda's Open MPI 5.x needs PMIx; the srun default (cray_shasta) hands it PMI1/2,
# which silently starts N independent size-1 singletons instead of one N-rank job.
export SRUN_MPI=pmix

# ufs2arco MPITopology._init_log races on os.makedirs when ranks are mis-initialized;
# creating it up front is harmless and removes the race entirely.
mkdir -p eagle-logs/inference

echo "Start inference-testing"
echo "$(date)"
srun --mpi=${SRUN_MPI} eagle-tools inference inference.testing.yaml || { echo "FAILED inference-testing"; exit 1; }
echo "$(date)"
echo "Done inference-testing"

echo "Start inference-validation"
echo "$(date)"
srun --mpi=${SRUN_MPI} eagle-tools inference inference.validation.yaml || { echo "FAILED inference-validation"; exit 1; }
echo "$(date)"
echo "Done inference-validation"
