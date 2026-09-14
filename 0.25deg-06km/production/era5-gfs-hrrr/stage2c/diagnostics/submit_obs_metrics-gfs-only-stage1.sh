#!/bin/bash

#SBATCH -J obs-metrics-gfs-only-s1
#SBATCH -o /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics/logs/obs-metrics-gfs-only-stage1.%j.out
#SBATCH -e /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics/logs/obs-metrics-gfs-only-stage1.%j.out
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
#SBATCH -t 01:30:00

set -x

module load conda
conda activate eagle

cd /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics
logdir="eagle-logs/obs_metrics"
diagdir="/pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics"
stageids="gfs-only-stage1a gfs-only-stage1b gfs-only-stage1c"

# conda's Open MPI 5.x needs PMIx; the srun default (cray_shasta) hands it PMI1/2,
# which silently starts N independent size-1 singletons instead of one N-rank job.
export SRUN_MPI=pmix

# stage1a/1b are plain global runs (model_type: global), so there is only the one
# global pass per stage -- no LAM/hrrr pass like the stage2* nested configs have.
for stageid in ${stageids}; do
    # ufs2arco MPITopology._init_log races on os.makedirs across ranks; create the
    # log dir up front to remove the race entirely.
    mkdir -p ${logdir}/global-${stageid}
    mkdir -p ${diagdir}/inference-${stageid}/obs-metrics

    echo "Running for global over ${stageid}"
    srun --mpi=${SRUN_MPI} eagle-tools obs-metrics obs-metrics.global.${stageid}.yaml || { echo "FAILED obs-metrics global ${stageid}"; exit 1; }
    echo " ... ${stageid} Done"
done
