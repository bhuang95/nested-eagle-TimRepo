"""Plot per-variable validation MSE vs epoch for the era5-gfs-hrrr transfer chain
(stage1a -> stage1b -> stage2a -> stage2b -> stage2c), one 1x5 panel figure per
variable with a shared y-axis range.

anemoi-training logs these under
    metrics/val_data_mse<_domain>_metric/data/<variable>/<rollout_step>_scale_0
computed on denormalized fields, so values are MSE in physical units. There is
no per-variable training loss.

- stage1a/1b are global ERA5 models (ERA5 long names, global domain only);
  stage2a/2b/2c are nested GFS+HRRR models (short names). VARIABLES maps the
  two and rescales the stage1 values where ERA5 units differ (geopotential
  m2/s2 -> height m, precipitation m -> mm).
- "whole domain" is val_data_mse_metric in every stage; stage2 panels also
  show val_data_mse_inside_lam_metric (HRRR region only).
- Stages 2b/2c validate over the training rollout; only rollout step 1 (6 h)
  exists for every epoch of every stage, so it is the default.

Stages trained over several slurm jobs list several mlflow run directories;
their metric files are concatenated in the order given (see plot_loss_curves).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from plot_loss_curves import MLFLOW, STAGES

G = 9.80665

# stage2 name: (stage1 name, factor applied to stage1 MSE, units of the MSE)
VARIABLES = {
    "gh_500": ("geopotential_500", 1 / G**2, "m$^2$"),
    "t_850": ("temperature_850", 1.0, "K$^2$"),
    "u_850": ("u_component_of_wind_850", 1.0, "m$^2$ s$^{-2}$"),
    "v_850": ("v_component_of_wind_850", 1.0, "m$^2$ s$^{-2}$"),
    "pl_gh": ("pl_geopotential", 1 / G**2, "m$^2$"),
    "pl_t": ("pl_temperature", 1.0, "K$^2$"),
    "pl_u": ("pl_u_component_of_wind", 1.0, "m$^2$ s$^{-2}$"),
    "pl_v": ("pl_v_component_of_wind", 1.0, "m$^2$ s$^{-2}$"),
    "pl_w": ("pl_vertical_velocity", 1.0, ""),
    "pl_q": ("pl_specific_humidity", 1.0, "kg$^2$ kg$^{-2}$"),
    "sfc_t2m": ("sfc_2m_temperature", 1.0, "K$^2$"),
    "sfc_u10": ("sfc_10m_u_component_of_wind", 1.0, "m$^2$ s$^{-2}$"),
    "sfc_v10": ("sfc_10m_v_component_of_wind", 1.0, "m$^2$ s$^{-2}$"),
    "sfc_sp": ("sfc_surface_pressure", 1.0, "Pa$^2$"),
    "sfc_accum_tp": ("sfc_total_precipitation_6hr", 1e6, "mm$^2$"),
    "sfc_t_surface": (None, 1.0, "K$^2$"),
    "sfc_sh2": (None, 1.0, "kg$^2$ kg$^{-2}$"),
    "sfc_u80": (None, 1.0, "m$^2$ s$^{-2}$"),
    "sfc_v80": (None, 1.0, "m$^2$ s$^{-2}$"),
    "sfc_total_column_water": ("sfc_total_column_water", 1.0, "kg$^2$ m$^{-4}$"),
}
STAGE1_ONLY = {"sfc_total_column_water"}

DOMAINS = {
    "whole domain": ("val_data_mse_metric", "#eb6834"),
    "HRRR region": ("val_data_mse_inside_lam_metric", "#2a78d6"),
}
STEP_STYLES = {1: "-", 2: "--", 3: ":", 4: "-."}
OUTDIR = "figures-era-gfs-hrrr/per_variable"


def read_series(run_dirs, metric_dir, name, rstep):
    """Concatenate a per-variable metric across runs.

    Epochs are defined by the rollout-step-1 file, which has one row per
    validation epoch; later rollout steps start at later epochs and are placed
    by matching their training-step column against it. Returns (epochs, values)
    or None if the metric was not logged.
    """
    epochs, vals, offset = [], [], 0
    for run in run_dirs:
        base = os.path.join(MLFLOW, run, "metrics", metric_dir, "data", name)
        if not os.path.isfile(os.path.join(base, "1_scale_0")):
            return None
        steps1 = np.loadtxt(os.path.join(base, "1_scale_0"), ndmin=2)[:, 2]
        fname = os.path.join(base, f"{rstep}_scale_0")
        if os.path.isfile(fname):
            data = np.loadtxt(fname, ndmin=2)
            idx = np.searchsorted(steps1, data[:, 2])
            epochs.append(offset + idx + 1)
            vals.append(data[:, 1])
        offset += len(steps1)
    if not vals:
        return None
    return np.concatenate(epochs), np.concatenate(vals)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vars", nargs="+", default=list(VARIABLES), choices=list(VARIABLES),
                        metavar="VAR", help="stage2 variable names (default: all)")
    parser.add_argument("--rollout-steps", nargs="+", type=int, default=[1],
                        choices=sorted(STEP_STYLES), help="rollout steps to plot (default: 1)")
    parser.add_argument("--no-lam", action="store_true", help="omit the HRRR-region lines")
    parser.add_argument("--logy", action="store_true", help="log-scale y-axis")
    args = parser.parse_args()

    domains = {k: v for k, v in DOMAINS.items() if not (args.no_lam and k == "HRRR region")}
    os.makedirs(OUTDIR, exist_ok=True)

    for var in args.vars:
        s1_name, s1_factor, units = VARIABLES[var]
        fig, axes = plt.subplots(1, len(STAGES), figsize=(22, 4.5), sharey=True)
        plotted = []
        for ax, (stage, runs) in zip(axes, STAGES.items()):
            is_stage1 = stage.startswith("stage1")
            name = s1_name if is_stage1 else (None if var in STAGE1_ONLY else var)
            factor = s1_factor if is_stage1 else 1.0
            lines = 0
            if name is not None:
                for label, (metric_dir, color) in domains.items():
                    for rstep in args.rollout_steps:
                        series = read_series(runs, metric_dir, name, rstep)
                        if series is None:
                            continue
                        epochs, y = series
                        y = y * factor
                        ax.plot(epochs, y, color=color, ls=STEP_STYLES[rstep], lw=2,
                                marker="o", ms=3)
                        plotted.append(y)
                        lines += 1
            if lines == 0:
                ax.text(0.5, 0.5, "not in this stage" if name is None else "not logged",
                        transform=ax.transAxes, ha="center", va="center",
                        color="0.35", fontsize=10)
                ax.set_xticks([])
            title = stage if name is None or name == var else f"{stage}  ({name})"
            ax.set_title(title, fontsize=10 if len(title) > 20 else 12)
            ax.set_xlabel("Epoch")
            ax.set_xlim(left=0)
            ax.grid(True, color="0.9", lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        finite = np.concatenate(plotted) if plotted else np.array([])
        finite = finite[np.isfinite(finite)]
        if finite.size and args.logy:
            axes[0].set_yscale("log")
            axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            axes[0].yaxis.set_major_formatter(FormatStrFormatter("%g"))
            axes[0].yaxis.set_minor_formatter(NullFormatter())
        elif finite.size:
            axes[0].set_ylim(0, 1.05 * finite.max())
        axes[0].set_ylabel(f"val MSE [{units}]" if units else "val MSE")

        handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=3, label=k)
                   for k, (_, c) in domains.items()]
        if len(args.rollout_steps) > 1:
            handles += [plt.Line2D([], [], color="0.3", ls=STEP_STYLES[r], lw=2,
                                   label=f"rollout step {r} ({6 * r} h)")
                        for r in args.rollout_steps]
        fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles))
        fig.suptitle(f"{var}: validation MSE vs epoch", y=0.99)

        fig.tight_layout(rect=(0, 0.07, 1, 1))
        suffix = "".join([
            "" if args.rollout_steps == [1] else "_r" + "".join(map(str, args.rollout_steps)),
            "_logy" if args.logy else "",
        ])
        outfile = os.path.join(OUTDIR, f"val_mse_{var}_stage1a-2c{suffix}.png")
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"saved {outfile}")


if __name__ == "__main__":
    main()
