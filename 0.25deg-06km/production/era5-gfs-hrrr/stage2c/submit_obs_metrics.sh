#!/bin/bash

#SBATCH -J nested-obs-metrics
#SBATCH -o /pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/logs/obs-metrics.%j.out
#SBATCH -e /pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/logs/obs-metrics.%j.out
#SBATCH --nodes=5
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 3:00:00

set -x

module load conda
conda activate eagle

cd /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c

# conda's Open MPI 5.x needs PMIx; the srun default (cray_shasta) hands it PMI1/2,
# which silently starts N independent size-1 singletons instead of one N-rank job.
export SRUN_MPI=pmix

# ufs2arco MPITopology._init_log races on os.makedirs when ranks are mis-initialized;
# creating them up front is harmless and removes the race entirely.
# global config sets log_path explicitly; the hrrr config has none, so it falls back
# to the eagle-logs/<command> default in eagle/tools/utils.py:init_topo.
mkdir -p eagle-logs/obs_metrics/global-test eagle-logs/obs_metrics

echo "Running for global over testing period"
srun --mpi=${SRUN_MPI} eagle-tools obs-metrics obs-metrics.global.testing.yaml || { echo "FAILED obs-metrics global"; exit 1; }
echo " ... Done"

echo "Running for HRRR over testing period"
srun --mpi=${SRUN_MPI} eagle-tools obs-metrics obs-metrics.hrrr.testing.yaml || { echo "FAILED obs-metrics hrrr"; exit 1; }
echo " ... Done"
