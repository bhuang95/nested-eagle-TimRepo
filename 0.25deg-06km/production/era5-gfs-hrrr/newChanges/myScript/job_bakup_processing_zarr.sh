#!/bin/bash
#SBATCH -J bakProcess
#SBATCH -o /pscratch/sd/b/bhuang95/anemoi-house/replay/atmosphere-subsampled/no-log-ERA5/transfer-learning/bakProcess.out
#SBATCH -e /pscratch/sd/b/bhuang95/anemoi-house/replay/atmosphere-subsampled/no-log-ERA5/transfer-learning/bakProcess.out
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH --exclusive
#SBATCH -t 48:00:00

set -x

#trainY40-1979-2018-0.25deg
#test-2021-2022-0.25deg

datasets="
valid-2019-2020-0.25deg
"

scradir="/pscratch/sd/b/bhuang95/anemoi-house/replay/atmosphere-subsampled/no-log-ERA5/transfer-learning/preprocessing/"
csfdir="/pscratch/sd/b/bhuang95/anemoi-house/replay/atmosphere-subsampled/no-log-ERA5/transfer-learning/preprocessing-bakup/"

mkdir -p ${csfdir}
cd ${scradir}
for data in ${datasets}; do
    tarfile=${csfdir}/${data}.dataset.zarr.tar.gz 
    tar -I 'pigz -p 12' -cf ${tarfile} ${data} || exit 100
done
