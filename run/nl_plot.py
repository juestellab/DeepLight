import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import deeplight as dl

import plotting as dlp

dlp.use_paper_style()
mpl.rcParams.update({'font.size': 9})


# The transforms the networks were trained with (dataloader units): sigma^E for
# the absorbed energy, sigma^nabla for the directional derivatives.
TRANSFORMS = ((r"$\sigma^E$", 0.1, 5e-4, "#1f4e79", "-"),
              (r"$\sigma^\nabla$", 1.0, 1e-2, "#b35b00", "--"))


def nl_weight(x, a, c):
    """d/dx sigma_{a,c}(x): the weight the loss gives an error at magnitude x."""
    e = np.exp((np.abs(x) + c) / a)
    return e / (a * (e - 1))


if __name__ == "__main__":
    # drawn at its printed width (0.64\textwidth), so sizes here are print sizes
    f = plt.figure(figsize=(4.32, 1.92), layout="constrained")
    axs = f.subplots(1, 2, gridspec_kw=dict(wspace=0.3))

    # (a) both transforms, cropped in y so the knees stay visible
    v_range = 0.6
    x = np.linspace(-v_range, v_range, 2001)
    axs[0].spines[["left", "bottom"]].set_position(("data", 0))
    axs[0].spines[["top", "right"]].set_visible(False)
    for (name, a, c, color, ls) in TRANSFORMS:
        axs[0].plot(x, np.asarray(dl.nl(x, a, c)), color=color, ls=ls, lw=1,
                    label=name)
    axs[0].set_xlabel("$x$", loc="right")
    axs[0].set_xlim(-v_range, v_range)
    axs[0].set_ylim(-10, 10)
    axs[0].set_xticks([-0.5, 0.5])
    axs[0].set_yticks([-5, 5])
    axs[0].set_title(r"$\sigma(x)$")
    # upper left: the only quadrant both curves leave empty
    axs[0].legend(loc="upper left", frameon=False, borderaxespad=0.1,
                  handlelength=1.5)
    axs[0].plot(1, 0, ">k", transform=axs[0].get_yaxis_transform(), clip_on=False, markersize=2)
    axs[0].plot(0, 1, "^k", transform=axs[0].get_xaxis_transform(), clip_on=False, markersize=2)

    # (b) the derivative of sigma^E alone: 1/c below c, 1/|x| between c and a,
    # 1/a above a, shown by thin guides behind the curve
    (name, a, c, color, ls) = TRANSFORMS[0]
    xlim = (1e-5, 10)
    x = np.geomspace(*xlim, 1000)
    axs[1].spines[["top", "right"]].set_visible(False)
    guide = dict(color="0.6", lw=0.7, ls=":")
    axs[1].plot(x, 1 / x, **guide)
    axs[1].axhline(1 / c, **guide)
    axs[1].axhline(1 / a, **guide)
    axs[1].plot(x, nl_weight(x, a, c), color=color, lw=1.2)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlim(*xlim)
    axs[1].set_ylim(1, 2e4)
    axs[1].set_xticks([c, a], labels=["$c$", "$a$"])
    axs[1].xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axs[1].set_yticks([1 / a, 1 / c], labels=["$1/a$", "$1/c$"])
    axs[1].yaxis.set_minor_locator(mpl.ticker.NullLocator())
    axs[1].set_xlabel("$|x|$")
    axs[1].set_title(r"$\sigma'(x)$")
    axs[1].text(3e-3, 1 / 3e-3 * 1.6, r"$1/|x|$", color="0.4", rotation=-38)

    for (ax, letter) in zip(axs, "ab"):
        ax.text(0, 1.075, letter, transform=ax.transAxes, size=11, weight="bold")

    out = os.path.join(dl.load_paths("./paths.toml")["results_dir"], "figures")
    os.makedirs(out, exist_ok=True)
    f.savefig(os.path.join(out, "nl_simple.png"), dpi=600)