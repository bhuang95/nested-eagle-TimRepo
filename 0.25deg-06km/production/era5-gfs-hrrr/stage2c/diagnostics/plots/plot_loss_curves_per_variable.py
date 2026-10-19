"""Plot per-variable validation MSE vs epoch for the three stage2a diagnostic
experiments of plot_loss_curves_stage2a.py, one single-panel figure per
variable: colour = experiment, linestyle = domain.

anemoi-training logs these under
    metrics/val_data_mse<_domain>_metric/data/<variable>/<rollout_step>_scale_0
computed on denormalized fields, so values are MSE in physical units. There is
no per-variable training loss.

- all three runs are nested GFS+HRRR models on the same variable set, so the
  ERA5 long-name / unit-rescaling mapping of the stage1a-2c version of this
  script (plot_loss_curves_per_variable_allStages.py) is not needed here.
- "whole domain" is val_data_mse_metric, "HRRR region" is
  val_data_mse_inside_lam_metric.
- all three train at rollout 1 (training.rollout.max: 1), so step 1 (6 h) is
  the only one logged; --rollout-steps is kept for runs that roll out further.
- each variable is written twice, linear and log-y; the log panel spreads out
  the converged tail where the three experiments are closest.

Experiments trained over several slurm jobs list several mlflow run directories;
their metric files are concatenated in the order given (see
plot_loss_curves_stage2a).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from plot_loss_curves_stage2a import MLFLOW, STAGES

# local to this script: the from-scratch run is green here, black in
# plot_loss_curves_stage2a.py, so that figure is left untouched
COLORS = {
    "gfs-hrrr S1a_BoRun": "green",
    "era5-gfs-hrrr S2a": "blue",
    "era5-gfs-hrrr S2a_gfs-hrrrS1aParm": "red",
}

# stage2 variable name: units of the MSE
VARIABLES = {
    "gh_500": "m$^2$",
    "t_850": "K$^2$",
    "u_850": "m$^2$ s$^{-2}$",
    "v_850": "m$^2$ s$^{-2}$",
    "pl_gh": "m$^2$",
    "pl_t": "K$^2$",
    "pl_u": "m$^2$ s$^{-2}$",
    "pl_v": "m$^2$ s$^{-2}$",
    "pl_w": "",
    "pl_q": "kg$^2$ kg$^{-2}$",
    "sfc_t2m": "K$^2$",
    "sfc_u10": "m$^2$ s$^{-2}$",
    "sfc_v10": "m$^2$ s$^{-2}$",
    "sfc_sp": "Pa$^2$",
    "sfc_accum_tp": "mm$^2$",
    "sfc_t_surface": "K$^2$",
    "sfc_sh2": "kg$^2$ kg$^{-2}$",
    "sfc_u80": "m$^2$ s$^{-2}$",
    "sfc_v80": "m$^2$ s$^{-2}$",
}

# domain: (metric directory, linestyle)
DOMAINS = {
    "whole domain": ("val_data_mse_metric", "-"),
    "HRRR region": ("val_data_mse_inside_lam_metric", "--"),
}
STEP_MARKERS = {1: None, 2: "o", 3: "s", 4: "^"}
OUTDIR = "figures-era-gfs-hrrr/per_variable_stage2a"


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


def render(var, series, domains, rollout_steps, logy):
    """Draw one variable's panel from series already read off disk."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = []
    for stage, ls, rstep, epochs, y in series:
        ax.plot(epochs, y, color=COLORS[stage], ls=ls, lw=2,
                marker=STEP_MARKERS[rstep], ms=3)
        plotted.append(y)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"val MSE [{VARIABLES[var]}]" if VARIABLES[var] else "val MSE")
    ax.set_xlim(left=0)
    finite = np.concatenate(plotted)
    finite = finite[np.isfinite(finite)]
    if finite.size and logy:
        ax.set_yscale("log")
        ax.set_ylim(0.9 * finite.min(), 1.1 * finite.max())
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", color="0.9", lw=0.8)
    else:
        if finite.size:
            ax.set_ylim(0, 1.05 * finite.max())
        ax.grid(True, color="0.9", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], color=COLORS[s], lw=2, label=s) for s in STAGES]
    handles += [plt.Line2D([], [], color="0.3", ls=ls, lw=2, label=k)
                for k, (_, ls) in domains.items()]
    if len(rollout_steps) > 1:
        handles += [plt.Line2D([], [], color="0.3", marker=STEP_MARKERS[r], ms=3, lw=2,
                               label=f"rollout step {r} ({6 * r} h)")
                    for r in rollout_steps]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3)
    fig.suptitle(f"{var}: validation MSE vs epoch", y=0.99)

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    suffix = "".join([
        "" if rollout_steps == [1] else "_r" + "".join(map(str, rollout_steps)),
        "_logy" if logy else "",
    ])
    outfile = os.path.join(OUTDIR, f"val_mse_{var}_stage2a_diag{suffix}.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vars", nargs="+", default=list(VARIABLES), choices=list(VARIABLES),
                        metavar="VAR", help="variable names (default: all)")
    parser.add_argument("--rollout-steps", nargs="+", type=int, default=[1],
                        choices=sorted(STEP_MARKERS), help="rollout steps to plot (default: 1)")
    parser.add_argument("--no-lam", action="store_true", help="omit the HRRR-region lines")
    args = parser.parse_args()

    domains = {k: v for k, v in DOMAINS.items() if not (args.no_lam and k == "HRRR region")}
    os.makedirs(OUTDIR, exist_ok=True)

    for var in args.vars:
        # read once, render linear and log-y from the same series
        series = []
        for stage, runs in STAGES.items():
            for label, (metric_dir, ls) in domains.items():
                for rstep in args.rollout_steps:
                    got = read_series(runs, metric_dir, var, rstep)
                    if got is None:
                        continue
                    series.append((stage, ls, rstep, *got))
        if not series:
            print(f"{var}: not logged in any experiment, skipped")
            continue
        for logy in (False, True):
            render(var, series, domains, args.rollout_steps, logy)


if __name__ == "__main__":
    main()
