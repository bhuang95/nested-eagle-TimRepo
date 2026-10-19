#!/bin/bash
# Submit inference, then obs-metrics only if inference finishes successfully.
#
# The obs-metrics job is queued immediately with --dependency=afterok, so this script
# returns right away (no need to keep a terminal open). Slurm starts obs-metrics only
# when inference exits 0; submit_inference-gfs-hrrr.sh exits 1 on any srun failure.
# If inference fails, --kill-on-invalid-dep=yes cancels obs-metrics instead of leaving
# it pending forever with reason DependencyNeverSatisfied.

set -euo pipefail

cd /global/u1/b/bhuang95/sorc/fromTim/era5-pretraining/nested-eagle-TimRepo/0.25deg-06km/production/era5-gfs-hrrr/stage2c/diagnostics

inf_jobid=$(sbatch --parsable submit_inference-gfs-hrrr.sh)
echo "inference   job: ${inf_jobid}"

obs_jobid=$(sbatch --parsable --dependency=afterok:${inf_jobid} --kill-on-invalid-dep=yes \
    submit_obs_metrics-gfs-hrrr-stage1.sh)
echo "obs-metrics job: ${obs_jobid} (runs after ${inf_jobid} completes OK)"

echo "check with: squeue -j ${inf_jobid},${obs_jobid}   or   sacct -j ${inf_jobid},${obs_jobid}"
