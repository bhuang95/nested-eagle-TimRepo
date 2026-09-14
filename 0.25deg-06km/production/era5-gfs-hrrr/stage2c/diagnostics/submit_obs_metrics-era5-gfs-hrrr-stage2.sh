#!/bin/bash

#SBATCH -J era5-obs-metrics
#SBATCH -o /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics/logs/obs-metrics.%j.out
#SBATCH -e /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics/logs/obs-metrics.%j.out
# 8 tasks/node, not 16: a single rank peaks at ~41 GB during the nested-global
# regrid, but 16 tasks share the node's ~476 GB allocation = ~29.8 GB/task, so a
# node OOMs whenever several ranks peak together (this killed job 57825924).
# 8 nodes keeps 64 ranks total, so the batch count and walltime are unchanged.
#SBATCH --nodes=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 3:00:00

set -x

module load conda
conda activate eagle

cd /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics
logdir="eagle-logs/obs_metrics"
diagdir="/pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics/"
stageids="era5-gfs-hrrr-stage2a era5-gfs-hrrr-stage2b era5-gfs-hrrr-stage2c"

# conda's Open MPI 5.x needs PMIx; the srun default (cray_shasta) hands it PMI1/2,
# which silently starts N independent size-1 singletons instead of one N-rank job.
export SRUN_MPI=pmix

#

for stageid in ${stageids}; do
    mkdir -p ${logdir}/global-${stageid}
    mkdir -p ${logdir}/lam-${stageid}
    mkdir -p ${diagdir}/inference-${stageid}/obs-metrics

    echo "Running for global over ${stageid}"
    srun --mpi=${SRUN_MPI} eagle-tools obs-metrics obs-metrics.global.${stageid}.yaml || { echo "FAILED obs-metrics global ${stageid}"; exit 1; }
    echo " ... ${stageid} Done"

    echo "Running for hrrr over ${stageid}"
    srun --mpi=${SRUN_MPI} eagle-tools obs-metrics obs-metrics.hrrr.${stageid}.yaml || { echo "FAILED obs-metrics hrrr ${stageid}"; exit 1; }
    echo " ... ${stageid} Done"
done
