import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import deeplight as dl


# font = {'family': 'serif',
#         # 'weight': 'normal',
#         'size': 8}
#
# matplotlib.rc('font', **font)

mpl.rcParams.update({'font.size': 8,
                     "text.usetex": False,
                     "font.family": "serif",
                     })


def results_figure(loss_df, gain_df, deriv_loss_df, deriv_gain_df, depth_loss):
    f = plt.figure(layout="tight", figsize=(10.2, 3.8))
    gs = gridspec.GridSpec(2, 3, figure=f, width_ratios=(1, 1, 1), wspace=0.65, hspace=0.3,
                           left=0.06, right=0.44, bottom=0.03, top=0.935)
    gs_d = gridspec.GridSpec(2, 2, figure=f, wspace=0.35, hspace=0.5,
                             left=0.56, right=0.99, bottom=0.097, top=0.935)

    gs_id0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], wspace=0)
    gs_id1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0)
    gs_id2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 2], wspace=0)

    gs_ood0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0], wspace=0)
    gs_ood1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0)

    ax_id0 = [f.add_subplot(gs_id0[0, :-1]), f.add_subplot(gs_id0[0, -1])]
    ax_id1 = [f.add_subplot(gs_id1[0, :-1]), f.add_subplot(gs_id1[0, -1])]
    ax_id2 = [f.add_subplot(gs_id2[0, :-1]), f.add_subplot(gs_id2[0, -1])]

    ax_ood0 = [f.add_subplot(gs_ood0[0, :-1]), f.add_subplot(gs_ood0[0, -1])]
    ax_ood1 = [f.add_subplot(gs_ood1[0, :-1]), f.add_subplot(gs_ood1[0, -1])]

    gs_d0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_d[0, 0], hspace=0)
    gs_d1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_d[1, 0], hspace=0)

    gs_d2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_d[0, 1], hspace=0)
    gs_d3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_d[1, 1], hspace=0)

    ax_id_d = [f.add_subplot(gs_d0[0]),]
    ax_id_d += [f.add_subplot(gs_d0[1]),]
    ax_ood_d = [f.add_subplot(gs_d1[0]),]
    ax_ood_d += [f.add_subplot(gs_d1[1]),]

    ax_id_dd = [f.add_subplot(gs_d2[0]),]
    ax_id_dd += [f.add_subplot(gs_d2[1]),]
    ax_ood_dd = [f.add_subplot(gs_d3[0]),]
    ax_ood_dd += [f.add_subplot(gs_d3[1]),]

    figure_panel(loss_df, gain_df, "In Distribution", ax_id0,
                 "   ID Operator", 'a',
                 ylims=(0.004, 0.25),
                 yticks=([0.01, 0.1,], ["0.01", "0.1",]),
                 )
    figure_panel(deriv_loss_df, deriv_gain_df, "In Distribution_1", ax_id1,
                 r"  ID$_1$ Derivative", 'b',
                 ylims=(0.03, 1),
                 yticks=([1, 0.1], ["1.0", "0.1"])
                 )
    figure_panel(deriv_loss_df, deriv_gain_df, "In Distribution_2", ax_id2,
                 r"  ID$_2$ Derivative", 'c',
                 ylims=(0.03, 1),
                 yticks=([1, 0.1], ["1.0", "0.1"])
                 )
    figure_panel(loss_df, gain_df, "Out-of-Distribution", ax_ood0,
                 "  OOD Operator", 'd',
                 ylims=(0.004, 0.25),
                 yticks=([0.01, 0.1,], ["0.01", "0.1",]),
                 )
    figure_panel(deriv_loss_df, deriv_gain_df, "Out-of-Distribution", ax_ood1,
                 " OOD Derivative", 'e',
                 ylims=(0.03, 1),
                 yticks=([1, 0.1], ["1.0", "0.1"])
                 )

    ax_legend = f.add_subplot(gs[1, 2])
    colors = mpl.color_sequences['Set2']
    cmap = sns.diverging_palette(210, 20, s=70, l=55, n=14)
    c2, c1, c3 = cmap[1], cmap[4], cmap[-2]# ".6"
    colors = [c1, c2, c3]
    handles = [mpatches.Patch(facecolor=colors[0], edgecolor='dimgray', label="Baseline"),
               mpatches.Patch(facecolor=colors[1], edgecolor='dimgray', label="DeepLight"),
               mpatches.Patch(facecolor=colors[2], edgecolor='dimgray', label="Rel. Gain"),]

    ax_legend.legend(loc="center left", bbox_to_anchor=(.1, 0.5), borderaxespad=0,
                     handles=handles, fontsize=10, mode='expand', frameon=False, labelspacing=1)
    ax_legend.axis('off')

    ax_id0[0].set_ylabel("Error")
    ax_ood0[0].set_ylabel("Error")
    ax_id2[1].set_ylabel("Rel. Gain")
    ax_ood1[1].set_ylabel("Rel. Gain")
    # depth_loss
    depth_plot(ax_id_d, *depth_loss["ID"], "ID Operator", "f", deriv=False)
    depth_plot(ax_ood_d, *depth_loss["OOD"], "OOD Operator", "h", deriv=False)

    depth_plot(ax_id_dd, *depth_loss["ID_Deriv"], "ID Derivative", "g", deriv=True)
    depth_plot(ax_ood_dd, *depth_loss["OOD_Deriv"], "OOD Derivative", "i", deriv=True)

    divider_x = (ax_id2[-1].get_position().x1 + ax_id_d[0].get_position().x0)/2 - 0.005
    # fig_height = f.bbox.height
    # fig_width = f.bbox.width
    # fig_x = divider_x

    # Use normalized figure coordinates for placement
    f.add_artist(plt.Line2D([divider_x, divider_x], [0.017, 0.983], linewidth=1, color='k', transform=f.transFigure))

    # f.tight_layout()
    return f


def figure_panel(loss_df, gain_df, dist, axs, label, letter,
                 ylims=None, yticks=None, yminorticks=None):
    c1, _, _, _, _, c2 = sns.color_palette("rocket")
    cmap = sns.diverging_palette(210, 20, s=70, l=55, n=14)
    c1, c2, c3 = cmap[0], cmap[-2], ".6"
    c2, c1, c3 = cmap[0], cmap[4], cmap[-2]# ".6"
    # c1, c2, c3 = "#587291", "#f68e5f", ".6"
    palette = [c1, c2]
    sns.violinplot(x='dist', y='Rel. RMSE', hue='type',
                   data=loss_df[loss_df["dist"] == dist],
                   split=True, inner="quartile", cut=0,
                   ax=axs[0], legend=False,
                   log_scale=True,
                   # bw_adjust=0.75,
                   hue_order=["Baseline", "Sobolev"],
                   palette=palette, saturation=1)
    axs[0].text(-0.30, 1.075, letter, transform=axs[0].transAxes,
                size=12, weight='bold')
    axs[0].set_title(label, loc='left')
    axs[0].set_xlabel(None)
    axs[0].set_ylabel(None)
    axs[0].set_xticks([])
    if ylims is not None:
        axs[0].set_ylim(*ylims)

    if yticks is not None:
        axs[0].set_yticks(*yticks)
    if yminorticks is not None:
        axs[0].set_yticks(*yminorticks, minor=True)

    axs[1].plot([-1, 0.5], [0, 0], color="black")
    sns.violinplot(x='dist', y='Gain',
                   data=gain_df[gain_df["dist"] == dist],
                   split=True, inner="quartile", cut=0,
                   ax=axs[1], legend=False,
                   color=c3,
                   bw_adjust=0.75)
    axs[1].set_xlabel(None)
    axs[1].set_ylabel(None)
    axs[1].set_xticks([])
    axs[1].yaxis.tick_right()
    axs[1].yaxis.set_label_position("right")

    axs[1].set_ylim(-1.1, 1.1)
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def depth_plot(axs, err, err_base, title, letter, deriv):
    n_pixels = err.shape[0]
    c1 = plt.cm.Set2(0)
    c2 = plt.cm.Set2(1)
    c1, _, _, _, _, c2 = sns.color_palette("rocket")
    cmap = sns.diverging_palette(210, 20, s=70, l=55, n=14)
    c1, c2, c3 = cmap[1], cmap[-2], ".6"
    c2, c1, c3 = cmap[1], cmap[4], cmap[-2]# ".6"

    # c1, c2, c3 = "#587291", "#F68E5F", "#42304B"
    depth = np.linspace(0, n_pixels/100, n_pixels)

    cutoff = 235
    depth = depth[:cutoff]
    err = err[:cutoff]
    err_base = err_base[:cutoff]

    axs[0].scatter(depth, err_base, label="Baseline", color=c1, s=2)
    axs[0].scatter(depth, err, label="Sobolev", color=c2, s=2)

    axs[0].fill_between(depth, err_base, np.zeros(depth.shape), color=c1, alpha=0.2)
    axs[0].fill_between(depth, err, np.zeros(depth.shape), color=c2, alpha=0.2)

    rel_gain = (err_base - err)/err_base
    axs[1].scatter(depth, rel_gain, s=2, color=c3)
    axs[1].plot([0, n_pixels/100], [0, 0], color="black", linewidth=1, linestyle='dashed')

    # axs[0].set_ylim(0, np.percentile(err[np.where(np.isfinite(err))], 95)*1.2)
    axs[0].set_ylim(0, None)
    axs[0].set_ylabel("Mean Error")
    axs[1].set_ylabel("Rel. Gain")
    if deriv:
        axs[0].set_ylim(0, 0.17)
        axs[1].set_ylim(-0.8, 0.8)
    else:
        axs[1].set_ylim(-0.35, 0.35)

    axs[1].set_xlabel("Depth [cm]")
    axs[0].set_xlim(0, cutoff/100)
    axs[1].set_xlim(0, cutoff/100)
    axs[0].set_xticks([])
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticks([0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25], minor=True)
    axs[0].set_title(title)
    axs[0].text(-0.05, 1.15, letter, transform=axs[0].transAxes,
                size=12, weight='bold')

    y_ticks = axs[0].get_yticks()
    axs[0].set_yticks(y_ticks[1:])
    # axs[0].set_title("Error in Depth under skin")


if __name__ == "__main__":
    paths = dl.load_paths('./paths.toml')
    path = os.path.join(paths['results_dir'], "quantitative_results_with_deriv.h5")
    loss_df = pd.read_hdf(path, "loss")
    deriv_loss_df = pd.read_hdf(path, "deriv_loss")
    gain_df = pd.read_hdf(path, "gain")
    deriv_gain_df = pd.read_hdf(path, "deriv_gain")

    with h5.File(path, "r") as f:
        err_id = f["ID"]["Error"][()]
        derr_id = f["ID"]["Diff_Error"][()]
        E_id = f["ID"]["Energy"][()]
        c_id = f["ID"]["Count"][()]

        err_ood = f["OOD"]["Error"][()]
        derr_ood = f["OOD"]["Diff_Error"][()]
        E_ood = f["OOD"]["Energy"][()]
        c_ood = f["OOD"]["Count"][()]

        depth_loss = {
            "ID": [e/c_id for e in err_id],
            "OOD": [e/c_ood for e in err_ood],
            "ID_Deriv": [e/c_id for e in derr_id],
            "OOD_Deriv": [e/c_ood for e in derr_ood]
        }

    h = results_figure(loss_df, gain_df, deriv_loss_df, deriv_gain_df, depth_loss)
    h.savefig("../paper/Resources/quantitative_results_with_deriv_small.png", dpi=600)
