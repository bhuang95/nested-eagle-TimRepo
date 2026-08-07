#!/bin/bash
#SBATCH -J flash-attn-build
#SBATCH -o /scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/production/slurm/flash-attn-build.%j.out
#SBATCH -e /scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/production/slurm/flash-attn-build.%j.err
#SBATCH --partition=u1-compute
#SBATCH --account=ai4wp
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

set -x

module load cuda/12.8.1
export CUDA_HOME=/apps/cuda/cuda-12.8.1

source /scratch4/BMC/gsienkf/Bo.Huang/extApps/miniconda3/etc/profile.d/conda.sh
conda activate eagle

# Building for H100 (this cluster's GPU partition); torch runtime is cu124,
# nvcc 12.8 is forward-compatible for compiling extension kernels against it.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
export TMPDIR=/scratch4/BMC/gsienkf/Bo.Huang/tmp/pip-build
export PIP_CACHE_DIR=/scratch4/BMC/gsienkf/Bo.Huang/.cache/pip

# Compute nodes have no internet access, so install from pre-downloaded local files.
PKG_CACHE=/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/production/slurm/pkg_cache
pip install --no-index --find-links="$PKG_CACHE" ninja
pip install --no-cache-dir --no-build-isolation "$PKG_CACHE/flash_attn-2.7.4.post1.tar.gz"
