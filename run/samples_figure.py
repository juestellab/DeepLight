import os
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import deeplight as dl
import cmasher as cmr


import plotting as dlp

dlp.use_paper_style()


def load_data():
    paths = dl.load_paths('./paths.toml')
    data_path = os.path.join(paths['data_dir'], "test_data.h5")
    data = dl.load_data(data_path, '65')

    mu, dmu, E, dE = (d[[80, 0, 13]] for d in data[:4])
    return (mu[..., 0], mu[..., 1], dmu[..., 0], dmu[..., 1], E[..., 0],
            dE[..., 0])


def make_figure(mu_a, mu_s, dmu_a, dmu_s, E, dE):
    # drawn at its printed width (0.95\textwidth), so sizes here are print sizes
    f, axs = plt.subplots(3, 6, figsize=(6.4, 3.0), layout="constrained",
                          gridspec_kw=dict(hspace=0, wspace=0))

    def add_scale_bar(ax, label, length_frac=0.25, height=0.03, margin=0.05, color='white'):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x_length = length_frac * (xlim[1] - xlim[0])
        y_height = height * (ylim[1] - ylim[0])
        x_margin = margin * (xlim[1] - xlim[0])
        y_margin = margin * (ylim[1] - ylim[0])

        x_start = x_margin
        y_start = ylim[0] + y_margin

        # Add rectangle as scale bar
        bar = Rectangle(
            (x_start, y_start),
            x_length,
            y_height,
            transform=ax.transData,
            color=color,
            ec=None,
            lw=0
        )
        ax.add_patch(bar)
        ax.text(x_start + x_length/2, y_start + 3 * y_height/2, label, color=color, ha="center", va="bottom")

    def matshow(ax, arr, letter, mode="log", vmax=None):
        if mode == "deriv":
            arr = arr.T/np.max(np.abs(arr))
            cmap = cmr.get_sub_cmap("twilight_shifted", 0.1, 0.9)
            ax.matshow(arr, norm="linear", cmap=cmap, vmin=-1, vmax=1)
            ax.text(0.05, 0.82, letter, transform=ax.transAxes,
                    size=10, weight="bold")
        elif mode == "symlog":
            arr = arr.T
            cmap = cmr.get_sub_cmap("twilight_shifted", 0.1, 0.9)
            norm = matplotlib.colors.SymLogNorm(linthresh=1e-2, vmin=-vmax, vmax=vmax)
            ax.matshow(arr, norm=norm, cmap=cmap)
            ax.text(0.05, 0.82, letter, transform=ax.transAxes,
                    size=10, weight="bold")
        elif mode == "linear":
            arr = arr.T/np.max(np.abs(arr))
            cmap = "inferno"
            ax.matshow(arr, norm="linear", cmap=cmap, vmin=0, vmax=1)
            ax.text(0.05, 0.82, letter, transform=ax.transAxes,
                    size=10, color="w", weight="bold")
        else:
            arr = arr.T
            cmap = "inferno"
            ax.matshow(arr, norm="log", cmap=cmap, vmin=1e-4, vmax=1)
            ax.text(0.05, 0.82, letter, transform=ax.transAxes,
                    size=10, color="w", weight="bold")
        ax.set_axis_off()

    def column_label(ax, text):
        ax.text(-0.02, 0.5, text,
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                transform=ax.transAxes)

    label = iter("abcdefghijklmnopqr")

    for i in range(3):
        matshow(axs[i, 0], mu_a[i], next(label), mode="linear")
        matshow(axs[i, 1], mu_s[i], next(label), mode="linear")
        matshow(axs[i, 2], E[i], next(label), mode="log")
    add_scale_bar(axs[0, 0], "1 cm")
    axs[0, 0].set_title(r"$\mu_a$", y=1, pad=4)
    axs[0, 1].set_title(r"$\mu'_s$", y=1, pad=4)
    axs[0, 2].set_title(r"$E$", y=1, pad=4)

    column_label(axs[0, 0], r"ID$_1$")
    column_label(axs[1, 0], r"ID$_2$")
    column_label(axs[2, 0], r"OOD")
    column_label(axs[0, 3], r"ID$_1$")
    column_label(axs[1, 3], r"ID$_2$")
    column_label(axs[2, 3], r"OOD")

    # the direction drawn for each sample
    directions = [0, 2, 1]
    dE_max = max(np.max(np.abs(dE[i][j])) for (i, j) in enumerate(directions))

    for (i, j) in enumerate(directions):
        matshow(axs[i, 3], dmu_a[i][j], next(label), mode="deriv")
        matshow(axs[i, 4], dmu_s[i][j], next(label), mode="deriv")
        matshow(axs[i, 5], dE[i][j], next(label), mode="symlog", vmax=dE_max)
    add_scale_bar(axs[0, 3], "1 cm", color="black")
    axs[0, 3].set_title(r"$v_a$", y=1, pad=4)
    axs[0, 4].set_title(r"$v'_s$", y=1, pad=4)
    axs[0, 5].set_title(r"$\nabla_v E$", y=1, pad=4)

    cbs = [f.colorbar(axs[0, c].images[0], ax=axs[:, c:c + 1],
                      location="bottom", aspect=27, shrink=0.995)
           for c in range(6)]

    for cb in cbs[3:5]:
        cb.set_ticks([-1, -0.5, 0, 0.5, 1],
                     labels=["-1", "-0.5", "0", "0.5", "1"])
    cbs[5].set_ticks([-1, 0, 1], labels=["-1", "0", "1"])
    cbs[5].ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return f


if __name__ == "__main__":
    f = make_figure(*load_data())
    out = os.path.join(dl.load_paths('./paths.toml')['results_dir'], "figures")
    os.makedirs(out, exist_ok=True)
    f.savefig(os.path.join(out, "samples_wide.png"), dpi=600)