"""
Create the global quarter-degree target grid (with regridding mask) used by
forecast_regrid_kwargs.target_grid_path in the stage2c obs-metrics/prewxvx/spectra
configs, e.g.

    ${SCRATCH}/nested-eagle/0.25deg-06km/mesh-gen/csmswt-trim25/global_quarter_degree_with_mask.nc

1.00deg-15km/data/create_grids.py::get_global_data_grid() built its target grid
with xesmf.util.grid_global(1, 1, cf=True, lon1=360), which happens to line up
exactly with that project's 1-degree GFS grid (cell centers at -89.5..89.5).
That's a coincidence of the 1-degree case, not a general recipe: the actual
gfs.zarr used here is on a 719x1440 grid (field_shape=[719, 1440], centers
-89.75..89.75 / 0.0..359.75 -- i.e. a standard 721x1440 pole-to-pole 0.25deg
grid with both pole rows dropped), which is NOT what grid_global(0.25, 0.25)
would produce (that gives 720x1440 centers offset by 0.125deg). So this script
constructs the grid explicitly to match gfs.zarr's real coordinates via
xesmf.util.cf_grid_2d(), and cross-checks it against the actual dataset.

It then reuses eagle.tools.nested.prepare_regrid_target_mask() -- the exact
function the eagle pipeline itself calls at runtime -- to compute and attach the
"mask" variable needed for conservative_normed regridding.

The cutout kwargs below (hrrr.zarr trim_edge + gfs.zarr) match
0.25deg-06km/production/era5-gfs-hrrr/stage2c/obs-metrics.global.{testing,validation}.yaml.
"""

import os

import numpy as np
import xesmf.util
import anemoi.datasets

from eagle.tools.nested import prepare_regrid_target_mask

STORE_DIR = os.path.join(
    os.environ["SCRATCH"], "nested-eagle", "0.25deg-06km", "mesh-gen", "csmswt-trim25"
)

DATA_DIR = "/scratch4/BMC/gsienkf/Bo.Huang/expRuns/ML/nested-eagle/0.25deg-06km/data/testing"

ANEMOI_REFERENCE_DATASET_KWARGS = {
    "cutout": [
        {"dataset": f"{DATA_DIR}/hrrr.zarr", "trim_edge": [25, 24, 25, 26]},
        {"dataset": f"{DATA_DIR}/gfs.zarr"},
    ],
    "adjust": "all",
    "min_distance_km": 6,
}


def get_global_data_grid():
    # bounds chosen so cell centers land exactly on gfs.zarr's actual
    # 0.25 deg grid: lon 0.0..359.75 (1440 pts), lat -89.75..89.75 (719 pts)
    ds = xesmf.util.cf_grid_2d(
        lon0_b=-0.125, lon1_b=359.875, d_lon=0.25,
        lat0_b=-89.875, lat1_b=89.875, d_lat=0.25,
    )
    ds = ds.drop_vars("latitude_longitude")

    # GFS goes north -> south
    ds = ds.sortby("lat", ascending=False)
    return ds


def check_against_gfs_zarr(ds, gfs_zarr_path):
    """cross-check constructed grid centers against the real gfs.zarr coordinates"""
    ads = anemoi.datasets.open_dataset(gfs_zarr_path)
    field_shape = tuple(ads.field_shape)
    assert field_shape == (len(ds["lat"]), len(ds["lon"])), (
        f"grid shape {(len(ds['lat']), len(ds['lon']))} != gfs.zarr field_shape {field_shape}"
    )
    lat2d = ads.latitudes.reshape(field_shape)
    lon2d = ads.longitudes.reshape(field_shape)
    np.testing.assert_allclose(ds["lat"].values, lat2d[:, 0])
    np.testing.assert_allclose(ds["lon"].values, lon2d[0, :])


if __name__ == "__main__":

    os.makedirs(STORE_DIR, exist_ok=True)

    base_grid_path = f"{STORE_DIR}/global_quarter_degree.nc"
    if not os.path.isfile(base_grid_path):
        gds = get_global_data_grid()
        check_against_gfs_zarr(gds, ANEMOI_REFERENCE_DATASET_KWARGS["cutout"][1]["dataset"])
        gds.to_netcdf(base_grid_path)
        print(f"Stored {base_grid_path}")
    else:
        print(f"Found existing {base_grid_path}, reusing it")

    target_grid_path = prepare_regrid_target_mask(
        anemoi_reference_dataset_kwargs=ANEMOI_REFERENCE_DATASET_KWARGS,
        horizontal_regrid_kwargs={
            "target_grid_path": base_grid_path,
            "regridder_kwargs": {"method": "conservative_normed"},
        },
    )
    print(f"Target grid with mask: {target_grid_path}")
