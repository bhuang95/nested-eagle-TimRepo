#!/usr/bin/env python
"""Print the `latitudes` and `longitudes` variables of a zarr dataset."""
import zarr
import numpy as np

#ZARR_PATH = "/pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/gfs.zarr"
ZARR_PATH = "/pscratch/sd/b/bhuang95/nested-eagle/data/0.25deg-06km/data/hrrr.zarr"

z = zarr.open(ZARR_PATH, mode="r")

for name in ("latitudes", "longitudes"):
    values = z[name][:]
    print(f"=== {name} ===")
    print("shape:", values.shape, "dtype:", values.dtype)
    print(values)

    # Unique grid lines (sorted high -> low), useful for the 0.25deg grid
    unique_values = np.unique(values)[::-1]
    print(f"number of unique {name}:", unique_values.size)
    print(unique_values)
    print()
