import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib as mpl
import cmasher as cmr
import deeplight as dl

mpl.rcParams.update({'font.size': 10,
                     "text.usetex": True,
                     "font.family": "serif",
                     })

E_SCALE = 5e-4
dE_SCALE = 1e-2


def nl(x, c):
    return np.sign(x) * (np.log((np.exp(c) - np.exp(-np.abs(x)))/(np.exp(c) - 1)) + np.abs(x))


def nl_deriv(x, c):
    return 1/(np.exp(np.abs(x) + c) - 1) * np.exp(np.abs(x) + c)


def matshow(ax, arr, letter, title, deriv=False):
    arr = arr.T/np.max(np.abs(arr))
    letter_str = r"\textbf{" + letter + r"}"
    if deriv:
        cmap = cmr.get_sub_cmap("twilight_shifted", 0.1, 0.9)
        ax.matshow(arr, norm="linear", cmap=cmap, vmin=-1, vmax=1)
        ax.text(0.05, 0.87, letter_str, transform=ax.transAxes,
                size=12)
    else:
        lims = (0, 1)
        cmap = "inferno"
        norm = "linear"
        ax.matshow(arr, norm=norm, cmap=cmap) #, vmin=lims[0], vmax=lims[1])
        ax.text(0.05, 0.87, letter_str, transform=ax.transAxes,
                size=12, color="w")
    ax.set_title(title, y=0.975) #pad=-8)
    ax.set_axis_off()


def add_scale_bar(ax, label, length_frac=0.25, height=0.02, margin=0.05, color='white'):
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Convert fractions to data coordinates
    x_length = length_frac * (xlim[1] - xlim[0])
    y_height = height * (ylim[1] - ylim[0])
    x_margin = margin * (xlim[1] - xlim[0])
    y_margin = margin * (ylim[1] - ylim[0])

    # Lower-right corner position
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



if __name__ == "__main__":
    paths = dl.load_paths('./paths.toml')
    v_range = 1.1
    x = np.linspace(-v_range, v_range, 1000)
    # f, axs = plt.subplots(1, 5, figsize=(5.1, 2.5),
    #                       layout="constrained", gridspec_kw=dict(hspace=0, wspace=0))
    f = plt.figure(figsize=(5.1, 5.3), layout="constrained")
    # f = plt.figure(figsize=(10.2, 4), layout="constrained")
    # gs = GridSpec(2, 5, figure=f, wspace=0.1, hspace=0.05, width_ratios=(3, 3, 2, 2, 2))
    # axs = [f.add_subplot(gs[:, 0]), f.add_subplot(gs[:, 1])]
    subfigs = f.subfigures(2, 1, wspace=0.05, height_ratios=(1, 1.25))

    axs = subfigs[0].subplots(1, 2, gridspec_kw=dict(hspace=0.1, wspace=0))
    axs[0].spines[["left", "bottom"]].set_position(("data", 0))
    axs[0].spines[["top", "right"]].set_visible(False)
    axs[1].spines[["top", "right"]].set_visible(False)

    axs[0].plot(x, nl(x, E_SCALE), color='k', linewidth=1)
    axs[0].plot(x, nl(x, dE_SCALE), color='k', linestyle='--', linewidth=1)
    axs[0].set_xlabel("$x$", loc="right")
    axs[0].set_xlim(-v_range, v_range)
    axs[0].set_xticks([-1, -0.5, 0.5, 1])
    axs[0].set_yticks([-5, 5])
    axs[0].set_yticks([-7.5, -2.5, 2.5, 7.5], minor=True)
    axs[0].set_title(r"$\sigma(x)$")

    axs[0].plot(1, 0, ">k", transform=axs[0].get_yaxis_transform(), clip_on=False, markersize=2)
    axs[0].plot(0, 1, "^k", transform=axs[0].get_xaxis_transform(), clip_on=False, markersize=2)



    x = np.geomspace(5e-5, 7, 1000)
    axs[1].plot(x, nl_deriv(x, E_SCALE), color='k', label=r"$c=5\times 10^{-4}$", linewidth=1)
    axs[1].plot(x, nl_deriv(x, dE_SCALE), color='k', linestyle='--', label=r"$c=10^{-2}$", linewidth=1)
    axs[1].axvline(E_SCALE, color='gray', linewidth=1, linestyle = ':')
    axs[1].axvline(dE_SCALE, color='gray', linewidth=1, linestyle = ':')
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("$x$")
    axs[1].set_xlim(np.min(x), np.max(x))
    axs[1].set_title(r"$\frac{d}{dx}\sigma(x)$")
    axs[1].legend(loc='upper right')

    t = axs[1].yaxis.get_offset_text()
    t.set_size(30)
    t = axs[1].xaxis.get_offset_text()
    t.set_size(30)
    # axs[1].plot(1, 0, ">k", transform=axs[1].get_yaxis_transform(), clip_on=False, markersize=2)
    # axs[1].plot(0, 1, "^k", transform=axs[1].get_xaxis_transform(), clip_on=False, markersize=2)


    axs[0].text(-0, 1.075, r"\textbf{a}", transform=axs[0].transAxes,
                size=12)
    axs[1].text(-0.0, 1.075, r"\textbf{b}", transform=axs[1].transAxes,
                size=12)

    # f.tight_layout()

    axs_E = subfigs[1].subplots(2, 3, gridspec_kw=dict(hspace=0, wspace=0))
    # axs_E = [f.add_subplot(gs[0, i]) for i in range(2, 5)]
    # axs_dE = [f.add_subplot(gs[1, i]) for i in range(2, 5)]

    data = dl.load_data(os.path.join(paths['data_dir'], "test_data.h5"))
    # idx = 12
    idx = 16
    E = data[2][idx, :, :, 0]
    dE = data[3][idx, :, :, 0]

    matshow(axs_E[0, 0], E, 'c', r"$E$")
    matshow(axs_E[0, 1], nl(E, E_SCALE), 'd', r"$\sigma(E)$")
    matshow(axs_E[0, 2], np.log(E), 'e', r"$\log(E)$")

    add_scale_bar(axs_E[0, 0], "1 cm", length_frac=0.25)

    matshow(axs_E[1, 0], dE, 'f', r"$\nabla_v E$", deriv=True)
    matshow(axs_E[1, 1], nl(dE, dE_SCALE), 'g', r"$\sigma(\nabla_v E)$", deriv=True)
    matshow(axs_E[1, 2], np.asinh(dE/dE_SCALE), 'h', r"$\sinh^{-1}(\nabla_v E)$", deriv=True)
    # matshow(axs_E[2, 1], np.sign(dE)*(np.log10(1+np.abs(dE)/(dE_SCALE))), 'h', r"$\mathrm{symlog}(\nabla_v E)$", deriv=True)
    # matshow(axs_E[1, 2], np.sign(dE) * np.log(np.maximum(np.abs(dE), dE_SCALE)), 'h', deriv=True)

    add_scale_bar(axs_E[1, 0], "1 cm", length_frac=0.25, color="black")

    cb2 = f.colorbar(axs_E[0, 0].images[0],
                     ax=axs_E[0, 2],
                     location="right",
                     aspect=27,
                     shrink=0.85)
    cb2.set_ticks([0, 0.25, 0.5, 0.75, 1],
                  labels=["min", "", "", "", "max"])
    # cbar_ticks = cb2.ax.xaxis.get_majorticklabels()
    # cbar_ticks[0].set_horizontalalignment("left")
    # cbar_ticks[-1].set_horizontalalignment("right")
    # cb2.ax.set_xlabel("Absorbed Energy", fontsize=9, labelpad=-2.5, rotation='vertical')
    # cbar_ticks[1].set_transform(cbar_ticks[1].get_transform() + mpl.transforms.ScaledTranslation(0.15, 0.01, f.dpi_scale_trans))
    cb2 = f.colorbar(axs_E[1, 2].images[0],
                     ax=axs_E[1, 2],
                     location="right",
                     aspect=27,
                     shrink=0.85)
    cb2.set_ticks([-1, -0.5, 0, 0.5, 1],
                  labels=["-1", "", "", "", "1"])
    # cbar_ticks = cb2.ax.xaxis.get_majorticklabels()
    # cbar_ticks[0].set_horizontalalignment("left")
    # cbar_ticks[-1].set_horizontalalignment("right")
    # cb2.ax.set_xlabel("Directional Derivative", fontsize=9, labelpad=-7.5)

    subfigs[1].suptitle("Scaling Samples", fontsize=12)

    f.savefig("../paper/Resources/nl_comp_small.png", dpi=600)
