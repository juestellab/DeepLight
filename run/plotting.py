"""Figure helpers shared by the plotting scripts in run/.

Lives beside them rather than in the `deeplight` package, which a training run
imports and which has no use for seaborn and matplotlib:

    import plotting as dlp

Three layers, and no fourth:

1. Style -- the palette, as constants. One definition, so the panels of one
   figure cannot drift apart in colour.
2. Marks -- `violin`, `median_line`. Each takes an existing Axes as its first
   argument, draws, and returns None. None of them creates a figure: layout
   stays in the figure functions.
3. Cosmetics -- `style_axis`, kept separate from the marks so that a new panel
   type does not have to re-thread label kwargs through a drawing function, and
   so it can style axes the marks did not draw.

The log-scale helpers at the bottom serve the error-against-magnitude panels:
bin centres and padded ranges, the reference lines and tick thinning. They take
colours and dashes as plain keywords rather than a regime object, so the
paper's naming for which regime is which stays in the figure script with the
rest of the naming.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter

# ------------------------------------------------------------------- style

# One diverging ramp, sampled at three points. Previously re-derived in every
# drawing function, with the unpacking overwritten on the next line, so the
# panels of one figure silently disagreed about which blue they meant.
_RAMP = sns.diverging_palette(210, 20, s=70, l=55, n=14)
BASELINE = _RAMP[4]     # the no-derivative arm
ARM = _RAMP[1]          # anything trained with a derivative term
GAIN = _RAMP[-2]        # the baseline-relative improvement

# Point marks need a different palette from the violin fills. BASELINE and ARM
# are three steps apart on one 14-step diverging ramp: as the two halves of a
# split violin, position tells them apart and the near hues read as one family,
# which is the point. As loose scatter points there is no position to help, and
# a pale #b0cbd0 on white sits at 1.4:1 against the page and vanishes beside
# #649ca7. So the scatter figures take the validated categorical set instead
# (`palette.md`), with a neutral dark for the reference series -- the same
# answer the density figure's `linear` regime reaches for the same reason.
POINT_REFERENCE = "#55555c"
POINT_COLORS = ("#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7")


def use_paper_style():
    """Apply the figure rcParams. Call once, from a script's entry point."""
    plt.rcParams["font.size"] = 8
    # STIX: a Times-like face, as the journal asks for, with matching maths
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["text.usetex"] = False


# The journal's floor for any text at print size. Every figure is drawn at its
# printed width, so this is also the smallest size passed to matplotlib.
MIN_TEXT = 8


# ------------------------------------------------------------------- marks

_VIOLIN_KW = dict(inner="quartile", cut=0, saturation=0.9, legend=False)


def violin(ax, df, x, y, hue, *, order, hue_order, palette, log=False,
           **kwargs):
    """Violins of `y` against `x`, the `hue` levels dodged side by side.

    The project's violin conventions (quartile inner, no tail extension, 0.9
    saturation, no per-axes legend) are applied here so every panel in every
    figure agrees on them.
    """
    sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, order=order,
                   hue_order=list(hue_order), palette=list(palette),
                   log_scale=log, **{**_VIOLIN_KW, **kwargs})


# The category slot a violin fills. Well under 1 because the panels are narrow
# and a violin that touches its neighbours reads as an area chart rather than as
# a distribution.
VIOLIN_WIDTH = 0.62


def median_line(ax, x, curve, *, color, lw=0.7, ls="-", alpha=1.0, label=None,
                emphasis=False, band=None):
    """One median curve, with its seed envelope if it has one.

    The style arrives as keywords rather than as a regime object so this stays
    free of the paper's naming -- the figure script unpacks whatever it calls a
    regime here.
    """
    style = dict(color=color, lw=lw, ls=ls, alpha=alpha, label=label,
                 zorder=3 if emphasis else 2)
    if band is not None and np.any(band[1] > band[0]):
        ax.fill_between(x, band[0], band[1], color=color, alpha=0.25, lw=0,
                        zorder=1)
    ax.plot(x, curve, **style)


# -------------------------------------------------------------- cosmetics

def style_axis(ax, *, title=None, letter=None, xlabel=False, ylabel=None,
               xticks=None, right_axis=False, letter_xy=(-0.30, 1.075)):
    """Panel cosmetics, independent of what drew on the axes.

    Every argument defaults to leaving the axes alone, except `xlabel`, which
    defaults to clearing seaborn's -- the panels are labelled by title, not by
    an x axis whose single category is already in the title.
    """
    if title is not None:
        ax.set_title(title, loc="left")
    if letter is not None:
        ax.text(*letter_xy, letter, transform=ax.transAxes, size=10,
                weight="bold")
    if xlabel is False:
        ax.set_xlabel(None)
    elif xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if right_axis:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")


# ------------------------------------------------------- log-scale helpers
#
# The error-against-magnitude panels: |error| against |field|, on log axes in
# both directions.

def centers(edges):
    """Geometric bin centres, the edges being log-spaced."""
    return np.sqrt(edges[:-1] * edges[1:])


def pad_log(values, decades=0.25):
    """A log-axis range around `values`, `decades` clear at each end."""
    finite = values[np.isfinite(values) & (values > 0)]
    return (10 ** (np.log10(finite.min()) - decades),
            10 ** (np.log10(finite.max()) + decades))


# Where the fraction labels sit: this fraction (0-1, log scale) of the panel's
# visible y-range.
FRACTION_LABEL_HEIGHT = 0.88


def fraction_lines(ax, axes, fractions):
    """`fractions` of |field|, as diagonals, labelled at one shared height.

    The x and y limits have to be final before this is called -- the label
    height reads the panel's *current* view, not the store's own edges, so a
    panel that gets cropped after this is called would misplace them.

    All labels sit above their line at `FRACTION_LABEL_HEIGHT`. Each label's x
    position is then wherever its own line crosses that height, clipped to the
    visible x-range, which is what lines them up even though they are drawn at
    a different slope. They are outlined, to still read against whatever they
    sit over.
    """
    for fraction in fractions:
        ax.plot(axes.field, fraction * axes.field, color="0.7", lw=0.6,
                ls="--", zorder=0)
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        y_at = ylo * (yhi / ylo) ** FRACTION_LABEL_HEIGHT
        at = min(max(y_at / fraction, xlo), xhi)
        text = ax.annotate(f"{fraction:.0%}", (at, fraction * at),
                           size=MIN_TEXT, color="0.3", ha="center",
                           va="bottom", textcoords="offset points",
                           xytext=(0, 3), zorder=4)
        text.set_path_effects([pe.withStroke(linewidth=2, foreground="w")])


def scale_lines(ax, knees, edges, labels="bottom"):
    """Dotted verticals at `{label: value}` scale knees inside `edges`.

    For the `nl` transform these are its two knees: below `c` it is linear in
    the field, above `a` it is logarithmic, which is where median curves are
    expected to part company. Infinite ones (the limiting regimes) and any
    outside the binned range have nothing to mark and are skipped.
    """
    for (label, value) in knees.items():
        if not edges[0] <= float(value) <= edges[-1]:
            continue
        ax.axvline(value, color="0.7", lw=0.8, ls=":", zorder=0)
        # `labels`: "bottom", "top", or None for a strip under a labelled panel
        if labels is not None:
            top = labels == "top"
            ax.annotate(label, (value, float(top)),
                        xycoords=("data", "axes fraction"), size=MIN_TEXT,
                        color="0.5", ha="left", va=labels,
                        textcoords="offset points", xytext=(2, -2 if top else 2))


def thin_log_ticks(ax, x=3, y=None):
    """At most `x` by `y` labelled decades, for panels an inch across.

    Eight decades label every one of them by default, which at a column's width
    is a grey smear where the tick labels should be. The minor ticks stay: they
    are what still says the axis is logarithmic once the labels are this sparse.
    """
    ax.xaxis.set_major_locator(LogLocator(numticks=x))
    axes = [ax.xaxis]
    if y is not None:  # None: a linear y axis, left alone
        ax.yaxis.set_major_locator(LogLocator(numticks=y))
        axes.append(ax.yaxis)
    for axis in axes:
        axis.set_minor_locator(LogLocator(subs="auto", numticks=999))
        axis.set_minor_formatter(NullFormatter())