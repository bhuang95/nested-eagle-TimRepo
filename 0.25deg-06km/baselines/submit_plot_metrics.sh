#!/bin/bash

#SBATCH -J nested-plot-metrics
#SBATCH -o logs/plot-metrics.%j.out
#SBATCH -e logs/plot-metrics.%j.out
#SBATCH --qos=shared
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 1:00:00

set -x

#source ~/loadMods/load_conda.sh
module load conda
# ufs2arco, not eagle: these scripts import only numpy/matplotlib/xarray/seaborn
# (nothing from anemoi or eagle), and seaborn is installed only in ufs2arco.
conda activate ufs2arco

mkdir -p figures
echo "Running plot_metrics_vs_lam_obs_06km_stage2c.py"
date
python plot_metrics_vs_lam_obs_06km_stage2c.py; lam_err=$?
echo "Done with ERR=$lam_err"
date

echo "Running plot_metrics_vs_global_obs_stage2c.py"
date
python plot_metrics_vs_global_obs_stage2c.py; global_err=$?
echo "Done with ERR=$global_err"
date

# propagate failure so sacct/seff show FAILED instead of COMPLETED
exit $(( lam_err | global_err ))
