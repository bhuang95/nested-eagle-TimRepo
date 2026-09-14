#!/bin/bash
#SBATCH -A m4718
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -t 06:00:00
#SBATCH -J untar_testing
#SBATCH -o /pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/testing/logs/untar_testing_%j.out
#SBATCH -e /pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/testing/logs/untar_testing_%j.out

set -euo pipefail

#SRC=/global/cfs/cdirs/m4718/timothys/nested-eagle/0.25deg-06km/data/testing.tar.gz
SRC=/pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/testing/testing.tar.gz
DEST=/pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/testing

mkdir -p "$DEST"

echo "Start: $(date)"
echo "Source: $SRC ($(du -h "$SRC" | cut -f1))"
echo "Dest:   $DEST"

# pigz overlaps I/O + CRC across threads; tar handles the untar.
# -I lets tar use pigz as the decompressor.
time tar -I pigz -xf "$SRC" -C "$DEST"

echo "Done: $(date)"
echo "Extracted size: $(du -sh "$DEST" | cut -f1)"
