import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import deeplight as dl
import cmasher as cmr


matplotlib.rcParams.update({'font.size': 8,
                            "text.usetex": True,
                            "font.family": "serif",
                            "text.latex.preamble": r"\usepackage{amsmath}"
                            })


E_SCALE = 5e-4
dE_SCALE = 1e-2


def nl(x, c):
    return np.sign(x) * (np.log((np.exp(c) - np.exp(-np.abs(x)))/(np.exp(c) - 1)) + np.abs(x))


def nl_inf(y, c):
    return np.log(np.exp(y) * (np.exp(c) - 1) + 1) - c


def load_data():
    paths = dl.load_paths('./paths.toml')
    data_path = os.path.join(paths['data_dir'], "test_data.h5")
    data = dl.load_data(data_path)

    mu_a, mu_s, E, dmu_a, dmu_s, dE = [], [], [], [], [], []
    for idx in [80, 13, 10, 7]:
        mu_a += [data[0][idx, :, :, 0],]
        mu_s += [data[0][idx, :, :, 1],]
        dmu_a += [data[1][idx, :, :, 0],]
        dmu_s += [data[1][idx, :, :, 1],]
        E += [data[2][idx, :, :, 0],]
        dE += [data[3][idx, :, :, 0],]

    return mu_a, mu_s, dmu_a, dmu_s, E, dE


def make_figure(mu_a, mu_s, dmu_a, dmu_s, E, dE):
    f, axs = plt.subplots(6, 3, figsize=(4, 5), layout="constrained",
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

    def matshow(ax, arr, letter, deriv=False):
        arr = arr.T/np.max(np.abs(arr))
        letter_str = r"\textbf{" + letter + r"}"
        if deriv:
            cmap = cmr.iceburn
            cmap = "managua"
            cmap = cmr.get_sub_cmap("twilight_shifted", 0.1, 0.9)
            ax.matshow(arr, norm="linear", cmap=cmap, vmin=-1, vmax=1)
            ax.text(0.05, 0.82, letter_str, transform=ax.transAxes,
                    size=12)
        else:
            cmap = "inferno"
            norm = "linear"
            ax.matshow(arr, norm=norm, cmap=cmap)
            ax.text(0.05, 0.82, letter_str, transform=ax.transAxes,
                    size=12, color="w")
        ax.set_axis_off()

    def column_label(ax, text, in_layout=True):
        ax.text(-0.02, 0.5, text,
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                in_layout=in_layout,
                transform=ax.transAxes)

    label = iter("abcdefghijklmnopqrstuvwxyzABCDE")

    for i in range(3):
        matshow(axs[i, 0], mu_a[i], next(label))
        matshow(axs[i, 1], mu_s[i], next(label))
        matshow(axs[i, 2], nl(E[i], E_SCALE), next(label))
        for j in range(3):
            if i == 0 and j == 0:
                add_scale_bar(axs[i, j], "1 cm", length_frac=0.25)
    axs[0, 0].set_title(r"$\mu_a$", y=1, pad=4)
    axs[0, 1].set_title(r"$\mu_s^\prime$", y=1, pad=4)
    axs[0, 2].set_title(r"$\sigma(E)$", y=1, pad=4)

    column_label(axs[0, 0], r"ID$_1$", True)
    column_label(axs[1, 0], r"ID$_2$", True)
    column_label(axs[2, 0], r"OOD")
    column_label(axs[3, 0], r"ID$_1$", True)
    column_label(axs[4, 0], r"ID$_2$", True)
    column_label(axs[5, 0], r"OOD", True)

    for i in range(3):
        matshow(axs[3+i, 0], dmu_a[i], next(label), deriv=True)
        matshow(axs[3+i, 1], dmu_s[i], next(label), deriv=True)
        matshow(axs[3+i, 2], nl(dE[i], dE_SCALE), next(label), deriv=True)
        for j in range(3):
            if i == 0 and j == 0:
                add_scale_bar(axs[3 + i, j], "1 cm", length_frac=0.25, color="black")
    axs[3, 0].set_title(r"$v_a$", y=1, pad=4)
    axs[3, 1].set_title(r"$v_s^\prime$", y=1, pad=4)
    axs[3, 2].set_title(r"$\sigma(\nabla_v E)$", y=1, pad=4)

    _ = f.colorbar(axs[0, 0].images[0],
                   ax=axs[:3, :],
                   aspect=27,
                   shrink=0.995)
    cb2 = f.colorbar(axs[3, 0].images[0],
                     ax=axs[3:, :],
                     aspect=27,
                     shrink=0.995)

    cb2.set_ticks([-1, -0.5, 0, 0.5, 1],
                  labels=["-1", "-0.5", "0", "0.5", "1"])
    return f, axs


if __name__ == "__main__":
    f, axs = make_figure(*load_data())
    f.savefig("../paper/Resources/samples_compr_small.png", dpi=600)
