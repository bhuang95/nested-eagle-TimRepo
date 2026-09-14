#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns


# In[2]:


from niceplots import plot_surface_vars, plot_level_vars


# In[3]:


#plt.style.use("~/nice.mplstyle")


# In[5]:


myprocdir="/pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/"
timprocdir="/global/cfs/cdirs/m4718/timothys/nested-eagle/0.25deg-06km/"
valortests = ["testing"]
metrics = ["rmse", "mae"]

for valortest in valortests:
    for metric in metrics:
        data = {}

        data["NE-era5-gfs-hrrr 0.25\u00B0/06km"] = xr.load_dataset(f"{myprocdir}/era5-gfs-hrrr/stage2c/inference-{valortest}/obs-metrics/{metric}.convobs.nested-lam.nc", decode_timedelta=True)
        data["NE-gfs-hrrr 0.25\u00B0/06km"] = xr.load_dataset(f"{timprocdir}/production/gfs-hrrr/stage1c/inference-{valortest}/obs-metrics/{metric}.convobs.nested-lam.nc", decode_timedelta=True)
        data["HRRR"] = xr.load_dataset(f"{timprocdir}/baselines/hrrr-forecasts-vs-hrrr-analysis/trim25/obs-metrics-{valortest}/{metric}.convobs.lam.nc", decode_timedelta=True)
        data["GFS"] = xr.load_dataset(f"{timprocdir}/baselines/gfs-forecasts-vs-hrrr-analysis/trim25/obs-metrics-{valortest}/{metric}.convobs.lam.nc", decode_timedelta=True)

        if metric == "rmse": 
            rmse = data
        elif metric == "mae": 
            mae = data
        else:
            raise ValueError("metric are not correctly defined")


    fig, axs = plot_surface_vars(rmse, metric_name="RMSE", sel={"fhr": slice(49)}, marker="o")
    fig.savefig(f"figures/{valortest}.lam_surface_48h_rmse_5fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plot_surface_vars(rmse, metric_name="RMSE")
    fig.savefig(f"figures/{valortest}.lam_surface_10d_rmse_5fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plot_surface_vars(mae, metric_name="MAE")
    fig.savefig(f"figures/{valortest}.lam_surface_mae_5fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plot_level_vars(rmse, metric_name="RMSE", sel={"fhr": slice(49)}, marker="o")
    fig.savefig(f"figures/{valortest}.lam_plevel_48h_rmse_6fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plot_level_vars(rmse, metric_name="RMSE")
    fig.savefig(f"figures/{valortest}.lam_plevel_10d_rmse_6fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plot_level_vars(mae, metric_name="MAE")
    fig.savefig(f"figures/{valortest}.lam_plevel_mae_6fields.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for ds in list(rmse.values()) + list(mae.values()):
        ds.close()
