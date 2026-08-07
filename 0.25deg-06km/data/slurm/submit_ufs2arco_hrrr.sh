#!/bin/bash

#SBATCH -J hrrr_zarr
#SBATCH -o logs/preprocessing.%j.out
#SBATCH -e logs/preprocessing.%j.out
#SBATCH --partition=u1-service
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3900M
#SBATCH --qos=batch
#SBATCH --account=ai4wp
#SBATCH -t 04:00:00

set -x 

source ~/loadMods/load_conda.sh
conda activate eagle
export SCRATCH=/scratch4/BMC/gsienkf/Bo.Huang/expRuns/ML/

cd  /scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/nested-eagle-TimRepo/0.25deg-06km/data
#python create_grids.py

#srun ufs2arco gfs.testing.yaml --overwrite
#echo "done with gfs data"
srun ufs2arco hrrr.testing.yaml --overwrite
echo "done with hrrr"

## Now copy some things over to community
#mywork=$WORK/nested-eagle/1.00deg-15km/data
#mycomm=$COMMUNITY/nested-eagle/1.00deg-15km/data
#mkdir -p $mycomm
#mkdir -p $mycomm/logs/gfs
#mkdir -p $mycomm/logs/hrrr

##cp -v $mywork/*.nc $mycomm
#cp -v $mywork/*.yaml $mycomm
#cp -v $mywork/logs/gfs/log.0000.*.out $mycomm/logs/gfs
#cp -v $mywork/logs/hrrr/log.0000.*.out $mycomm/logs/hrrr
