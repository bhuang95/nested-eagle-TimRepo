"""Plot train/val loss vs epoch for era5-gfs-hrrr stage2a and its trial
stage2a_t1 on a single panel, one color per stage; solid = train, dashed = val.

Stages trained over several slurm jobs list several mlflow run directories;
their metric files are concatenated in the order given.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

MLFLOW = "/pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/"

STAGES = {
    "gfs-hrrr-s1a[tp: 8, mlr: 1e-3]": ["gfs-hrrr_bo/stage1a/logs/mlflow/528295371695142637/6289a75511ac44eca2a8490dd0dc22aa",
                   "gfs-hrrr_bo/stage1a/logs/mlflow/528295371695142637/0515305bc3e14ed9ac5fd38e20f1da42"],
    "era5-gfs-hrrr-s2a-t2[tp: 8, mlr: 5e-4]": ["era5-gfs-hrrr/stage1a/logs/mlflow/124465700651661291/3bf770ae540c4d9eb055bd890890d0fe",
                   "era5-gfs-hrrr/stage1a/logs/mlflow/124465700651661291/417fbbd1ed83470eab683072c5799a4d"],
    "era5-gfs-hrrr-s2a-t1[tp: 8, mlr: 1e-3]": ["era5-gfs-hrrr/stage1a/logs/mlflow/471321083918611265/94e381bca95a4a94adccfef8f8ba42c9",
                   "era5-gfs-hrrr/stage1a/logs/mlflow/471321083918611265/872d99077ace4f77a4a23695584040fa"],
    "era5-gfs-hrrr-s2a-t5[tp: 8, mlr: 3e-3]": ["era5-gfs-hrrr/stage1a/logs/mlflow/636725823954791046/faeff2b2074c4275aab2774dc1b77238",
                                               "era5-gfs-hrrr/stage1a/logs/mlflow/636725823954791046/fa275bbec6484c52a9000a08f70ba370",
                                               "era5-gfs-hrrr/stage1a/logs/mlflow/636725823954791046/8369e3267eaa49ff98a7757f6ed67d88",
                                               "era5-gfs-hrrr/stage1a/logs/mlflow/636725823954791046/81a4f0db4150488882647f0ba701e857"],
    "era5-gfs-hrrr-s2a-t4[tp: 8, mlr: 5e-3]": ["era5-gfs-hrrr/stage1a/logs/mlflow/966481630053585104/e7409e12c2a344788671f9a0c9034961",
                                               "era5-gfs-hrrr/stage1a/logs/mlflow/966481630053585104/b022f9b4741144dc8bc8aad919d44e6e"],
}

#    "era5-gfs-hrrr-s2a-t6[tp: 8, mlr: 7e-3]": ["era5-gfs-hrrr/stage1a/logs/mlflow/591724424657175537/b440692843294fe4afb1ee7aa114fd78"],
#"era5-gfs-hrrr-s2a[tp: 8, mlr: 1e-2]": ["era5-gfs-hrrr/stage1a/logs/mlflow/166947759278334038/cb49564832e34253963ff07819c9b73a"],
#    "era5-gfs-hrrr-s2a[tp: 0, mlr: 1e-4]": ["era5-gfs-hrrr/stage1a/logs/mlflow/997885074291130244/245d94e908844ae0bd03407940e4280d",
#                "era5-gfs-hrrr/stage1a/logs/mlflow/997885074291130244/9925759aaa1b4e3ead6c9ffd9677d076"],
#    "era5-gfs-hrrr-s2a[tp: 0, mlr: 1e-4]": "blue",
#    "era5-gfs-hrrr-s2a[tp: 8, mlr: 1e-2]": "pink",
#    "era5-gfs-hrrr-s2a-t6[tp: 8, mlr: 7e-3]": "red",
COLORS = {
    "gfs-hrrr-s1a[tp: 8, mlr: 1e-3]": "black",
    "era5-gfs-hrrr-s2a-t2[tp: 8, mlr: 5e-4]": "blue",
    "era5-gfs-hrrr-s2a-t1[tp: 8, mlr: 1e-3]": "purple",
    "era5-gfs-hrrr-s2a-t5[tp: 8, mlr: 3e-3]": "green",
    "era5-gfs-hrrr-s2a-t4[tp: 8, mlr: 5e-3]": "orange",
}
LINESTYLES = {"train": "-", "val": ":"}
OUTFILE = "figures-era-gfs-hrrr/loss_vs_epoch_stage2a_diag.png"
OUTFILE_LOG = "figures-era-gfs-hrrr/loss_vs_epoch_stage2a_diag_logy.png"


def read_metric(run_dirs, name):
    """Concatenate the value column (2nd) of an mlflow metric across runs."""
    vals = []
    for run in run_dirs:
        data = np.loadtxt(os.path.join(MLFLOW, run, "metrics", name), ndmin=2)
        vals.append(data[:, 1])
    return np.concatenate(vals)


def plot_losses(losses, outfile, logy=False):
    """One panel of all stages; logy spreads out the converged tail."""
    all_vals = np.concatenate([v for d in losses.values() for v in d.values()])

    fig, ax = plt.subplots(figsize=(9, 5))
    for stage, d in losses.items():
        for key in ("train", "val"):
            y = d[key]
            if np.all(np.isnan(y)):
                print(f"{stage}: {key} loss is NaN for all epochs, skipped")
                continue
            epochs = np.arange(1, len(y) + 1)
            ax.plot(epochs, y, color=COLORS[stage], ls=LINESTYLES[key], lw=2,  ms=3,
                    label=f"{stage} {'train' if key == 'train' else 'validation'}")
            # marker="o",
    ax.set_xlabel("Epoch")
    ax.set_ylabel("multi_dataset_loss (epoch mean)")
    ax.set_xlim(left=0)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(0.9 * np.nanmin(all_vals), 1.1 * np.nanmax(all_vals))
        ax.grid(True, which="both", color="0.9", lw=0.8)
    else:
        ax.set_ylim(0, 1.05 * np.nanmax(all_vals))
        ax.grid(True, color="0.9", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"saved {outfile}")


def main():
    losses = {
        stage: {
            "train": read_metric(runs, "train_multi_dataset_loss_epoch"),
            "val": read_metric(runs, "val_multi_dataset_loss_epoch"),
        }
        for stage, runs in STAGES.items()
    }

    plot_losses(losses, OUTFILE)
    plot_losses(losses, OUTFILE_LOG, logy=True)
    for stage, d in losses.items():
        print(f"{stage}: {len(d['train'])} train epochs, {len(d['val'])} val epochs")


if __name__ == "__main__":
    main()
