#!/bin/bash

#SBATCH -J nested-obs-metrics
#SBATCH -o logs/plot-metrics.%j.out
#SBATCH -e logs/plot-metrics.%j.out
#SBATCH --partition=u1-service
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --qos=batch
#SBATCH --account=gsienkf
#SBATCH -t 24:00:00

set -x

source ~/loadMods/load_conda.sh
conda activate eagle

mkdir -p figures
echo "Running plot_metrics_vs_lam_obs_06km_stage2c.py"
date
python plot_metrics_vs_lam_obs_06km_stage2c.py
echo "Done with ERR=$?"
date

echo "Running plot_metrics_vs_global_obs_stage2c.py"
date
python plot_metrics_vs_global_obs_stage2c.py
echo "Done with ERR=$?"
date
