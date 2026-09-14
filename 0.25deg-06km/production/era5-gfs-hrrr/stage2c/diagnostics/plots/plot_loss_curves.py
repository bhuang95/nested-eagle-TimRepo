"""Plot train/val loss vs epoch for the era5-gfs-hrrr transfer chain
(stage1a -> stage1b -> stage2a -> stage2b -> stage2c) as a 1x5 panel figure
with a shared y-axis range.

Stages trained over several slurm jobs list several mlflow run directories;
their metric files are concatenated in the order given.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

MLFLOW = "/pscratch/sd/b/bhuang95/nested-eagle/0.25deg-06km/production/era5-gfs-hrrr/stage1a/logs/mlflow"

STAGES = {
    "stage1a": ["989864899567876967/29a70980e917421697a9dd9d9e3c07ff",
                "989864899567876967/e3aba498df5042238b64b4eaa8ab5fe2"],
    "stage1b": ["994146407324332179/61e40fefaa664a9c8261eb57063971c2"],
    "stage2a": ["997885074291130244/245d94e908844ae0bd03407940e4280d",
                "997885074291130244/9925759aaa1b4e3ead6c9ffd9677d076"],
    "stage2b": ["463306433209169418/1d4ac8d7e3104c5186b4f60564ec68a5"],
    "stage2c": ["989032688975304236/f14410a5af554f72b0dc9aa260b4d2e7"],
}

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
OUTFILE = "figures-era-gfs-hrrr/loss_vs_epoch_stage1a-2c.png"


def read_metric(run_dirs, name):
    """Concatenate the value column (2nd) of an mlflow metric across runs."""
    vals = []
    for run in run_dirs:
        data = np.loadtxt(os.path.join(MLFLOW, run, "metrics", name), ndmin=2)
        vals.append(data[:, 1])
    return np.concatenate(vals)


def main():
    losses = {
        stage: {
            "train": read_metric(runs, "train_multi_dataset_loss_epoch"),
            "val": read_metric(runs, "val_multi_dataset_loss_epoch"),
        }
        for stage, runs in STAGES.items()
    }

    all_vals = np.concatenate([v for d in losses.values() for v in d.values()])
    ymax = 1.05 * np.nanmax(all_vals)

    fig, axes = plt.subplots(1, len(STAGES), figsize=(22, 4.5), sharey=True)
    for ax, (stage, d) in zip(axes, losses.items()):
        for key, color, ls in [("train", TRAIN_COLOR, "-"), ("val", VAL_COLOR, "--")]:
            y = d[key]
            epochs = np.arange(1, len(y) + 1)
            if np.all(np.isnan(y)):
                ax.text(0.5, 0.5, f"{key} loss is NaN\nfor all epochs",
                        transform=ax.transAxes, ha="center", va="center",
                        color="0.35", fontsize=10)
                continue
            ax.plot(epochs, y, color=color, ls=ls, lw=2, marker="o", ms=3, label=key)
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.set_xlim(left=0)
        ax.set_ylim(0, ymax)
        ax.grid(True, color="0.9", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("multi_dataset_loss (epoch mean)")
    axes[-1].legend(handles=[
        plt.Line2D([], [], color=TRAIN_COLOR, ls="-", lw=2, marker="o", ms=3, label="train"),
        plt.Line2D([], [], color=VAL_COLOR, ls="--", lw=2, marker="o", ms=3, label="validation"),
    ], frameon=False, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    fig.savefig(OUTFILE, dpi=150)
    print(f"saved {OUTFILE}")
    for stage, d in losses.items():
        print(f"{stage}: {len(d['train'])} train epochs, {len(d['val'])} val epochs")


if __name__ == "__main__":
    main()
