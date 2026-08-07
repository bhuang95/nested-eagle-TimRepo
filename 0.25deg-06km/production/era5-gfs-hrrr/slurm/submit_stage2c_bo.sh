#!/bin/bash
#SBATCH -J stage2c
#SBATCH -o /pscratch/sd/b/bhuang95/nested-eagle/slurm/stage2c_bo.%j.out
#SBATCH -e /pscratch/sd/b/bhuang95/nested-eagle/slurm/stage2c_bo.%j.out
#SBATCH --nodes=16
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
##SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
##SBATCH --constraint=gpu&hbm80g
#SBATCH --constraint=gpu
#SBATCH -t 32:00:00

set -x

# -----------------------------------------------------------------------------
# Interactive (front terminal) test instead of sbatch:
#   1) Grab an allocation (1 node is enough since num_gpus_per_model=4):
#        salloc --nodes 1 --qos interactive --time 00:30:00 \
#               --constraint 'gpu&hbm80g' --gpus-per-node=4 --account m4718
#   2) Set up the environment:
#        module load conda
#        module load nccl/2.21.5
#        conda activate eagle
#        export NCCL_DEBUG=WARN
#        export PYTHONFAULTHANDLER=1
#   3) cd into this directory (so --config-name resolves), then run:
#        cd /global/u1/b/bhuang95/sorc/fromTim/nested-eagle-era5pretraining/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr
#        srun --jobid $SLURM_JOB_ID --tasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 \
#             /global/u1/b/bhuang95/sorc/fromTim/anemoi/V20250603/anemoi-house/slurm2ddp.sh \
#             anemoi-training train --config-name=stage1a_bo
#   (append `dataloader.limit_batches.training=4 training.max_steps=10` for a fast smoke test)
# -----------------------------------------------------------------------------

# Run from the directory containing stage1a_bo.yaml so --config-name resolves.
#cd "${SLURM_SUBMIT_DIR}"
#mkdir -p slurm

module load conda
module load nccl/2.21.5
conda activate eagle

export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so.2
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

cd /global/homes/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/
srun /global/u1/b/bhuang95/sorc/fromTim/anemoi/V20250603/anemoi-house/slurm2ddp.sh \
     anemoi-training train --config-name=stage2c_bo
