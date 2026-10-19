"""The paper's quantitative figures, from the stores that hold the results.

    python results_figure.py                          # all four, to results_dir/figures
    python results_figure.py --out /tmp/check         # somewhere else

Writes `loss_transform`, `quantitative_results_combined`, `seed_checkpoint` and
`depth_channel`. `run/compute_error.py` writes `errors.h5` (and, with
`--depth-above`, `errors_depth_ext.h5`) and `run/compute_density.py` writes the
density stores; this reads them. Scoring is expensive and resumable, plotting is
cheap and repeated, which is why they are separate programs.

Both stores are final -- they hold the networks the paper reports and will not
grow -- and that is what this file is written against. Two consequences run
through everything below:

1. `config_digest` is the homogeneity guarantee. A digest pins every training
   setting except `direction_arm`, `alpha` and `seed`, `n_train_samples`
   included. Naming one literal digest and then naming the arm/alpha/seed makes
   a frame homogeneous by construction, so there is nothing left for a
   pool-detector to catch and none of the machinery that used to do the catching.
2. One plotted point is one network. Nothing here averages two runs together:
   every mark is a median over the *samples* of a single trained network, and a
   comparison between two marks is a comparison between two networks.

Drawing lives in `plotting.py`; this file owns the selection, the paper's
naming, and each figure's layout.
"""
import argparse
import os
import re

import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (FixedLocator, FormatStrFormatter, FuncFormatter,
                               LogLocator, MaxNLocator, NullFormatter,
                               NullLocator)
import numpy as np
import pandas as pd
import seaborn as sns

import deeplight as dl
import plotting as dlp
import train_network as tn

dlp.use_paper_style()

# --------------------------------------------------------- what is in the store
#
# The digests of the campaign. `SWEEP` is the settled configuration everything in
# the paper is read at; `SIZE` is that same configuration trained on a quarter of
# the data, which is the only axis the training-set panels vary. The remaining
# digests in the store are the loss-transform study, and they are drawn from the
# density store instead (`REGIMES`, below) where the comparison is exact.

# Computed from train_network's run groups, since the digest covers data_dir.
_DATA_DIR = dl.load_paths("./paths.toml")["data_dir"]
# 3 arms x 4 alphas x 2 seeds, + 5 seeds at the paper's arm, + 5 baseline
# seeds: 32 runs
SWEEP = tn.config_digest(tn.sweep_config(_DATA_DIR))
# the same configuration at n_train_samples=512
SIZE = tn.config_digest(tn.sweep_config(_DATA_DIR, "size"))

# The store scores every run under four transforms and three metrics. Narrowing
# to one of each is not optional: pooling `rmse` rows with `mae` and `psnr` rows
# would put three different numbers into one violin.
TRANSFORM, METRIC = "nl_train", "rmse"

# The three run selections the figures are built out of. `BASELINE` is the
# reference arm of the campaign -- trained without a Jacobian term at all, so
# `alpha=0` and `direction_arm="no_deriv"` select the same runs, the first saying
# why they are the reference. `ARM` is the configuration the paper reports.
BASELINE = dl.BASELINE
ARM = {"direction_arm": "2_per_sampler", "alpha": 1.0}

# Which of a run's two checkpoints to read. Every figure but `checkpoint_figure`
# reads the operator one; that figure exists to justify the choice.
OPERATOR = {"val_criterion": "operator"}

# Paper naming for the row splits of the test file.
SPLIT_LABELS = {"ID": "In Distribution", "OOD": "Out-of-Distribution"}

# The stored tangent-sampler names are legacy and actively misleading: `ID_1`
# (`V_TISSUE`) and `SIGN` (`scaled_sign_tangent`) are both drawn from
# distributions the network trained on, while `ID_2` (`V_RANDOM_TISSUE`) --
# despite the name -- is the one held out of training entirely. They are three
# different tangent families rather than draws of one, so they are never pooled.
# Named here rather than in the store, so the figures read correctly whatever
# the store ends up calling them.
#
# Which tangent family a derivative was taken along is independent of whether
# the row it was taken on was trained on, so a dE split is the pair of them,
# written `"<row>/<tangent>"` -- see `evaluation.density_splits`, which bins the
# density store the same way.
# The two trained-on families are pooled into one in-distribution tangent group
# and the held-out one is its own: three families, but only two *questions*, and
# the derivative panels ask the question. Pooling happens at scoring time
# (`compute_density.py --tangent-groups`, `evaluation.density_splits`), never
# here -- a pooled median is the median of the pooled pixels, which two finished
# median curves can no longer be made into.
TANGENT_GROUPS = {"ID": ("ID_1", "SIGN"), "OOD": ("ID_2",)}
TANGENT_OF = {tangent: group for (group, members) in TANGENT_GROUPS.items()
              for tangent in members}

# The stored names say what an arm does to the training set; the paper counts
# JVPs per training sample.
ARM_PROSE = {"1dir": "one deriv. per sample",
             "1_per_sampler": "one deriv. per tangent sampler",
             "2_per_sampler": "two deriv. per tangent sampler"}


# ----------------------------------------------------------------- selection

def load(path, transform=TRANSFORM, metric=METRIC):
    """The frame every error figure starts from, read and narrowed once.

    `dist` is the paper's name for the row split, attached here so no figure has
    to remember the mapping.

    `panel` names the comparison a row belongs to, and E and dE do not name
    theirs the same way. A row of the test file is in- or out-of-distribution,
    and that settles it for E. A derivative also has a direction, and whether
    *that* was trained on is a property of the tangent sampler -- independent of
    the row -- so a dE panel is the pair, `"<row>/<tangent>"`. This is the key
    `evaluation.density_splits` bins the density store by, so a panel of one
    store is the same set of pixels as the panel of that name in the other.
    """
    df = dl.read_errors(path, transform=transform, metric=metric)
    df = df[df["split"].isin(SPLIT_LABELS)].copy()
    df["dist"] = df["split"].map(SPLIT_LABELS)
    df["panel"] = np.where(df["field"] == "E", df["split"],
                           df["split"] + "/"
                           + df["sampler"].map(TANGENT_OF).astype(str))
    return df


def pick(df, **spec):
    """Rows matching every `column=value` in `spec`; a list value matches any.

    The single narrowing verb of this file. Because a `config_digest` fixes
    every setting the figures do not sweep, `pick(df, config_digest=SWEEP,
    **ARM)` is already a homogeneous frame -- there is no residual axis for it
    to be silently averaging over.
    """
    return df[dl.filter_mask(df, spec)]


# What identifies a run, carried through the per-sample collapse so a figure has
# the run's metadata beside its number. `val_op_loss` is constant per run and
# rides along because `best_seed` needs it.
RUN_KEYS = ("run_id", "direction_arm", "alpha", "seed", "n_train_samples",
            "val_criterion", "val_op_loss")


def per_run(sub):
    """One median per network per panel: the per-sample axis, collapsed.

    This is where "one point is one network" is enforced. Every figure below
    plots the output of this function, so none of them can pool two runs into a
    mark.

    Only the matched panels are kept, as in the main figure: dE along each
    split's own tangent group, not the held-out pairings `ID/OOD` and `OOD/ID`.
    """
    sub = sub[(sub["panel"] == sub["split"])
              | (sub["panel"] == sub["split"] + "/" + sub["split"])]
    return dl.sample_median(sub, by=[*RUN_KEYS, "dist", "field"])


def best_seed(sub, by=()):
    """The `run_id` of the lowest-validation-loss seed, per `by` group.

    Selection is on `val_op_loss` rather than `val_loss`: the total loss is
    `operator + alpha * jacobian`, so comparing it across alphas would rank a
    run by how heavily its own objective weighted the Jacobian term rather than
    by how well it did. With no `by`, returns the single winning `run_id`.
    """
    runs = sub.drop_duplicates("run_id")
    if runs.empty:
        raise LookupError("no run to pick a best seed from")
    if not by:
        return runs.loc[runs["val_op_loss"].idxmin(), "run_id"]
    return runs.loc[runs.groupby(list(by))["val_op_loss"].idxmin(), "run_id"]


def paper_runs(df):
    """`(arm_id, base_id)`: the two networks the main figure compares.

    Pinned to the settled arm and its best seed rather than scanned for over the
    whole store, so the published figure cannot change identity because some
    unrelated run was scored into the store later.
    """
    sweep = pick(df, config_digest=SWEEP, **OPERATOR)
    return (best_seed(pick(sweep, **ARM)), best_seed(pick(sweep, **BASELINE)))


# ------------------------------------------------------------ point figures
#
# The two figures that draw one mark per network rather than a distribution.
# Both use `dlp.POINT_*` rather than the violin fills: see the note there on why
# a split violin's two near hues do not survive being taken apart into points.
# Every series carries a marker as well as a colour.

BASELINE_STYLE = dict(color=dlp.POINT_REFERENCE, marker="s")
ARM_STYLE = dict(color=dlp.POINT_COLORS[0], marker="o")

POINT_KW = dict(s=34, zorder=3, edgecolor="white", linewidth=0.8)


def _scatter_points(ax, marks, jitter=0.13, rng=None):
    """Draw `(x, values, style)` marks as jittered points on a padded log axis.

    Seeds of one configuration land on nearly the same number, so without the
    jitter five networks draw as one mark. The padding is explicit rather than
    left to autoscale: with two groups a decade apart, matplotlib puts the
    outermost points exactly on the frame, where the marker is half cut off by
    it.
    """
    rng = rng or np.random.default_rng(0)
    for (x, values, style) in marks:
        ax.scatter(x + rng.uniform(-jitter, jitter, len(values)), values,
                   **POINT_KW, **style)
    ax.set_yscale("log")
    ax.set_ylim(*dlp.pad_log(np.concatenate([v for (_, v, _) in marks]),
                             decades=0.07))


def seed_figure(df, fig):
    """Per-seed error, instead of only the best network per arm.

    Draws into `fig`, a figure or subfigure.

    The published comparison picks one baseline and one derivative run by
    validation loss, which cannot on its own distinguish a real effect from a
    lucky initialisation. Plotting every seed is what settles that.

    Read at the paper's single arm and alpha: without that, every other arm and
    alpha in the sweep would land in the same "DeepLight" group and this would
    be the spread of five sweep axes rather than the seed spread of one trained
    configuration.

    ID and OOD get their own panel with their own y-limits rather than sharing
    one log axis as two x categories: OOD error is roughly a decade above ID's,
    so a shared axis draws the ID seed spread as a near-flat line at the bottom
    -- exactly the variability this figure exists to show.
    """
    sweep = pick(df, config_digest=SWEEP, **OPERATOR)
    med = per_run(pd.concat([pick(sweep, **BASELINE), pick(sweep, **ARM)]))
    is_base = med["direction_arm"] == BASELINE["direction_arm"]

    axs = fig.subplots(2, 2)
    for (row, field) in enumerate(("E", "dE")):
        for (col, (split, dist)) in enumerate(SPLIT_LABELS.items()):
            ax = axs[row, col]
            in_panel = (med["field"] == field) & (med["dist"] == dist)
            values = lambda mask: med[in_panel & mask]["rel_err"].to_numpy()
            _scatter_points(ax, [(0, values(is_base), BASELINE_STYLE),
                                 (1, values(~is_base), ARM_STYLE)])
            ax.set_xlim(-0.6, 1.6)
            # angled: side by side, the two names collide at print width
            ax.set_xticks([0, 1], ["Baseline", "DeepLight"], rotation=25,
                          ha="right", rotation_mode="anchor")
            # a few plain decimal ticks: most panels span less than a decade,
            # where the log axis has no labelled major tick of its own
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            # short titles: the full split names do not fit the panel width
            dlp.style_axis(ax, title=f"{split}, {field}",
                           letter="abcd"[2*row + col],
                           ylabel=f"Rel. RMSE ({field})" if col == 0 else None)
    return fig


def checkpoint_figure(df, fig, letter):
    """Error shift between the operator- and total-checkpoint of the same run.

    Draws into `fig`, a figure or subfigure.

    Every run of the sweep -- every arm, alpha and seed, plus the baseline -- as
    the ratio of its two checkpoints' errors: x is E, operator over total, y is
    the same for dE. A run at (1, 1) picked essentially the same network either
    way; a run left of 1 paid a fraction of its E to pick the total checkpoint,
    and reading its y says whether that bought dE back (also left of 1) or was a
    straight loss (right of 1).

    Colour carries `alpha`, the Sobolev weight, light-to-dark by size, rather
    than the direction arm: the divergence is driven by how strongly the total
    loss's Jacobian term outweighs its operator term, which is an axis the arm
    alone does not capture. The baseline (alpha=0) is left out: its total loss
    is its operator loss, so its two checkpoints are the same network by
    construction and say nothing about the criterion.
    """
    sweep = pick(df, config_digest=SWEEP)
    med = per_run(sweep[sweep["dist"] == "In Distribution"])
    med = med[med["alpha"] > 0]
    keys = ["direction_arm", "alpha", "seed"]
    wide = med.pivot(index=[*keys, "field"], columns="val_criterion",
                     values="rel_err").reset_index()
    if "total" not in wide.columns:
        raise LookupError("no total-checkpoint rows scored at this config digest")
    wide["ratio"] = wide["operator"] / wide["total"]
    spread = (wide.pivot(index=keys, columns="field", values="ratio")
              .dropna().reset_index())

    # The single most extreme run (by the larger of the two fields' deviation
    # from 1) sets the axis span for everyone else -- with 32 runs clustered
    # within a percent of 1, that one run alone compresses the rest into a dot
    # at the centre. Dropped, not cropped: there is no fixed value here to
    # anchor an annotation to.
    dev = (spread[["E", "dE"]] - 1).abs().max(axis=1)
    spread = spread[dev != dev.max()]

    alphas = sorted(spread["alpha"].unique())
    colors = dict(zip(alphas, sns.color_palette("viridis", len(alphas))))
    ax = fig.subplots()
    ax.axhline(1, color="0.6", lw=1, zorder=1)
    ax.axvline(1, color="0.6", lw=1, zorder=1)
    for a in alphas:
        rows = spread[spread["alpha"] == a]
        ax.scatter(rows["E"], rows["dE"], color=colors[a], s=40,
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=fr"$\alpha={a:g}$")
    span = max((spread[["E", "dE"]] - 1).abs().to_numpy().max(), 1e-6)
    ax.set_xlim(1 - span, 1 + span)
    ax.set_ylim(1 - span, 1 + span)
    ax.set_aspect("equal")
    # below the axes: at print width, any place inside covers some of the points
    ax.legend(frameon=False, fontsize=dlp.MIN_TEXT, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8,
              handletextpad=0.2)
    dlp.style_axis(ax, letter=letter,
                   xlabel=r"$E_{\mathrm{op}} / E_{\mathrm{total}}$",
                   ylabel=r"$dE_{\mathrm{op}} / dE_{\mathrm{total}}$",
                   letter_xy=(-0.25, 1.03))
    return fig


def seed_checkpoint_figure(df):
    """The seed spread (a-d) and the checkpoint comparison (e) as one figure."""
    # drawn at its printed width (0.85\textwidth), so sizes here are print sizes
    f = plt.figure(figsize=(5.74, 3.2), layout="constrained")
    (left, right) = f.subfigures(1, 2, width_ratios=(1.4, 1))
    seed_figure(df, left)
    checkpoint_figure(df, right, letter="e")
    return f


# --------------------------------------------------------------------------
# The loss-transform comparison, off the density store. `run/compute_density.py`
# writes it: the distribution of |error| given |E|, with the store's own exact
# medians, taken over the pixels at scoring time -- a median read back off the
# density's error axis could only ever be a bin centre, and those are 1.3x
# apart, wider than the difference between two of these curves at the loud end.
#
# Every regime here is a `no_deriv` baseline, so the comparison is between loss
# transforms and nothing else. Its `nl` regime is trained at the main sweep's
# own (a_E, c_E), which is what makes this the justification for that choice.
# --------------------------------------------------------------------------

# The nl run whose linear knee sits at the log floor: with c at the bottom of
# the range there is nothing left for the linear part of the transform to do, so
# it is the ablation of the knee rather than another point of the sweep.
ABLATION_C = 1e-6


class Regime:
    """How one training regime is drawn, wherever it is drawn."""

    def __init__(self, label, color, *, lw=0.7, ls="-", emphasis=False,
                 alpha=1.0):
        self.label, self.color, self.alpha = label, color, alpha
        self.lw, self.ls, self.emphasis = lw, ls, emphasis

    def style(self):
        """The keywords `dlp.median_line` draws this regime with."""
        return dict(color=self.color, lw=self.lw, ls=self.ls, alpha=self.alpha,
                    label=self.label, emphasis=self.emphasis)


# Drawn in this order, which is also the legend's. nl is the arm the paper
# argues for, so it gets its own colour; its ablation shares that colour and is
# told apart by the dash and by its own legend entry.
#
# Every regime also carries a dash pattern, so identity survives a greyscale
# print and a red-green reader -- long and sparse ones, which read as a line
# rather than as a texture at this size, the emphasised arm keeping the solid.
# Above a_E linear and nl agree to within the width of a stroke, and the one
# drawn second would otherwise simply erase the other: the dashes let the lower
# curve through the gaps, and the slight transparency through the strokes.
REGIMES = {
    "linear": Regime("Linear", "#5D759C", ls=(0, (7, 1.8))),
    "log": Regime("Log",  "#B35B59", ls=(0, (7, 1.8, 1.2, 1.8))),
    "nl": Regime(r"$\sigma$", "#808040", lw=1.0, emphasis=True, alpha=1),
    "nl_ablation": Regime(r"$\sigma$, $c_E = 10^{-6}$", "#808040",
                          ls=(0, (3, 1.8))),
}


def mode_of(row):
    """The `nl` regime a run was trained under: linear, log or nl.

    Runs from the main sweep carry no `nl_mode` -- they predate this comparison
    -- so the regime is read back off the E parameters themselves, which is
    what identifies it anyway. A missing attribute reaches the frame as NaN,
    not as a missing key, and NaN is truthy: test the type, not the truth
    value.
    """
    if isinstance(row.get("nl_mode"), str):
        return row["nl_mode"]
    a, c = float(row["a_E"]), float(row["c_E"])
    if np.isinf(c):
        return "linear"
    return "log" if np.isinf(a) else "nl"


def regime_of(row):
    """The `REGIMES` key a run belongs to: its mode, and nl split by its knee."""
    mode = mode_of(row)
    if mode == "nl" and np.isclose(float(row["c_E"]), ABLATION_C):
        return "nl_ablation"
    return mode


def read_regimes(path, split):
    """`(df, axes, curves)`: the density store's medians, grouped into regimes.

    A regime's median curves cannot be pooled exactly -- see
    `dl.pooled_median` -- so a regime with several seeds reports their
    geometric mean and envelope. This is the one place in the file where a mark
    is not a single network.
    """
    df, axes = dl.read_density(path, split=split)
    if df.empty:
        raise LookupError(f"{path} holds no run for split {split!r}")
    df = df.assign(regime=[regime_of(row) for _, row in df.iterrows()])
    keys = [k for k in REGIMES if k in set(df["regime"])]
    return (df, axes,
            {k: dl.pooled_median(df[df["regime"] == k]) for k in keys})


def median_panel(ax, curves, axes, styles):
    """The medians of every series over one another, log on both axes.

    `styles` maps each key of `curves` to the keywords `dlp.median_line` draws
    it with.
    """
    x = dlp.centers(axes.field)
    for (key, (median, low, high)) in curves.items():
        dlp.median_line(ax, x, median, band=(low, high), **styles[key])

    ax.set(xscale="log", yscale="log", xlim=(axes.field[0], axes.field[-1]),
           # the curves, not the reference lines: y=x reaches E's own loudest
           # pixel and would spend two decades of the panel on empty space
           ylim=dlp.pad_log(np.concatenate([c[1:] for c in curves.values()])))
    return ax


def scale_knees(row, field):
    """The `nl` transform's two knees, as `dlp.scale_lines` wants them.

    Below `c` the transform is linear in E and above `a` it is logarithmic,
    which is where the median curves are expected to part company.
    """
    return {f"${key}_{{{field}}}$": float(row[f"{key}_{field}"])
            for key in ("c", "a")}


def magnitude_label(field):
    """The x axis of a magnitude panel: the field whose bins it is drawn over."""
    return rf"$|{'E' if field == 'E' else r'\nabla_v E'}|$"


RESIDUAL_LABEL = {"E": r"$|\hat{E} - E|$",
                  "dE": r"$|\widehat{\nabla_v E} - \nabla_v E|$"}


# The derivative panel's magnitude bins run three decades further than any
# curve has pixels to show for -- `read_regimes` bins to the store's edges,
# not to where the regimes stop differing -- so its x axis is cropped here
# rather than left to run out into flat, empty extremes.
DE_XLIM = (5e-3, 1.1e1)
DE_YLIM_LOW = 1e-4


def _crop_to_data(ax, curves, axes, xlim):
    """Crop a median panel to `xlim` and refit its y axis to what is left.

    `median_panel` already sized the y axis to the full store range; cropping
    the x axis after that leaves headroom from bins no longer on screen. Refit
    from the same medians and bands `median_panel` drew, restricted to `xlim`.
    """
    x = dlp.centers(axes.field)
    mask = (x >= xlim[0]) & (x <= xlim[1])
    values = np.concatenate([np.asarray(v)[mask]
                             for curve in curves.values() for v in curve])
    ax.set_xlim(*xlim)
    ax.set_ylim(*dlp.pad_log(values))


# The v6 label-noise analysis: 5 independent MC reps of the same 10 validation
# phantoms, run at 1e8 rays. Production data is scored at 9x as many rays
# (val/test default to 900M), and MC noise falls as 1/sqrt(rays), so the reps'
# std is rescaled down by sqrt(9) to the noise level the reported errors
# actually carry.
NOISE_H5 = "/home/storage/Data/DeepLight/v6_noise_full/noise_analysis_0-19.h5"
NOISE_RAYS_RATIO = 9
# dE's 6 stored directions are 2 per tangent sampler, in direction_sampler_index
# order (ID_1, ID_2, SIGN); the loss transform's dE panel is drawn against the
# "ID/ID" split, which is ID_1+SIGN -- OOD (ID_2, indices 2:4) is excluded.
NOISE_ID_DIRECTIONS = (0, 1, 4, 5)


# The density store's own bins (70, one per `EDGES` entry in evaluation.py)
# are sized for the error medians, scored over the whole test set. The noise
# analysis is 10-20 phantoms, far fewer pixels per bin, so its curve is binned
# two store-bins wide instead -- smoother, at half the x resolution.
NOISE_BIN_MERGE = 2


def mc_noise_curve(field, edges):
    """`(x, median noise)` per |field| bin, from the v6 noise-analysis reps.

    The field magnitude a pixel is binned under is its mean over the 5 reps;
    the noise is their std, rescaled to production's ray count. Binned on
    `edges` merged `NOISE_BIN_MERGE`-wide, not `edges` itself -- see above --
    so `x` is coarser than `dlp.centers(edges)` and the caller must plot
    against it rather than the store's own bin centers.
    """
    with h5.File(NOISE_H5, "r") as f:
        data = f[field][()] * 1e4  # dataloader.py's own rescaling of E and dE
    if field == "dE":
        data = data[:, :, NOISE_ID_DIRECTIONS]
    value = np.abs(data.mean(axis=1)).ravel()
    noise = (data.std(axis=1) / np.sqrt(NOISE_RAYS_RATIO)).ravel()
    edges = edges[::NOISE_BIN_MERGE]
    x = dlp.centers(edges)
    medians = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        mask = (value >= edges[i]) & (value < edges[i + 1])
        if mask.any():
            medians[i] = np.median(noise[mask])
    return x, medians


def transform_figure(store_e, store_de):
    """Operator and derivative, side by side: error by magnitude, per regime.

    ID only -- the loss transform is calibrated in distribution. One legend for
    both panels, since they draw the same three regimes. dE's store is scored
    at the matched split only (`"ID/ID"`), so the two panels compare the
    regimes on the same question the main figure's derivative rows ask.
    """
    specs = (("E", store_e, "ID", "Operator", None),
             ("dE", store_de, "ID/ID", "Derivative", DE_XLIM))
    # drawn at its printed width (\textwidth), so sizes here are print sizes
    f, axs = plt.subplots(1, 2, figsize=(6.75, 2.8), layout="constrained")
    regime_keys = []
    for (ax, (field, path, split, title, xlim), letter) in zip(axs, specs, "ab"):
        df, axes, curves = read_regimes(path, split)
        regime_keys = [k for k in REGIMES if k in curves and k not in regime_keys] \
            + regime_keys
        fractions = (0.01, 0.1)
        styles = {k: REGIMES[k].style() for k in curves}
        median_panel(ax, curves, axes, styles)
        if xlim is not None:
            _crop_to_data(ax, curves, axes, xlim)
        # the MC noise analysis is not part of this release; without its store
        # the panel is drawn without the noise line
        if os.path.exists(NOISE_H5):
            x_noise, noise = mc_noise_curve(field, axes.field)
            dlp.median_line(ax, x_noise, noise, color="0.5", lw=1.0, ls=":",
                            label="MC noise" if field == "E" else None)
            valid = noise[np.isfinite(noise) & (noise > 0)]
            if valid.size:
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(min(ylo, valid.min() / 1.3), yhi)
        if field == "dE":
            ax.set_ylim(DE_YLIM_LOW, ax.get_ylim()[1])
        # drawn last, once the view is final: its labels read the panel's
        # current (cropped) x/y limits, not the store's own edges
        dlp.fraction_lines(ax, axes, fractions)
        ax.set_title(title)
        dlp.style_axis(ax, letter=letter, letter_xy=(-0.18, 1.06),
                       xlabel=magnitude_label(field),
                       ylabel=f"median {RESIDUAL_LABEL[field]}")
        if "nl" in curves:
            knees = scale_knees(df[df["regime"] == "nl"].iloc[0], field)
            panel_range = xlim if xlim is not None else axes.field
            knees = {k: v for k, v in knees.items()
                    if panel_range[0] <= float(v) <= panel_range[-1]}
            dlp.scale_lines(ax, knees, axes.field)
        dlp.thin_log_ticks(ax, x=5, y=5)

    handles = [Line2D([], [], **{k: v for k, v in REGIMES[k].style().items()
                                 if k != "emphasis"})
               for k in regime_keys]
    handles.append(Line2D([], [], color="0.5", lw=1.0, ls=":", label="MC noise"))
    f.legend(handles=handles, frameon=False, fontsize=dlp.MIN_TEXT, ncol=1,
             loc="outside right center")
    return f


# ------------------------------------------------- baseline against the arm
#
# The same median-against-magnitude reading as above, but comparing the two
# networks the paper reports rather than four loss transforms. It answers where
# in the field the derivative term actually buys anything.
#
# Its own stores (`density_arms*.h5`), not `density.h5`: `regime_of` classifies
# a run by its (a, c, nl_mode) alone, and the arm shares all three with the
# baseline, so both would land in the transform figure's `nl` regime and be
# pooled into one curve.

# One store per field, written by `compute_density.py --field`: the bin edges
# and the splits are each field's own, so they cannot share a file.
ARM_STORES = {"E": "density_arms.h5", "dE": "density_arms_dE.h5"}


def arm_curves(path):
    """`(row, axes, {split: {network: (curve, low, high)}})`.

    The median over a single run returns `low == high == curve`, so no band is
    drawn -- these are one network each, not a group of seeds.
    """
    df, axes = dl.read_density(path)
    series = {"Baseline": BASELINE, "DeepLight": ARM}
    out = {}
    for split in df["split"].unique():
        rows = df[df["split"] == split]
        out[split] = {net: dl.pooled_median(rows[dl.filter_mask(rows, spec)])
                      for (net, spec) in series.items()}
    return df.iloc[0], axes, out


def arm_strips(results_dir):
    """`{(panel, field): (axes, curves, knees)}` for the magnitude column.

    One store per field, and a store's splits are already the panels: E is
    binned per row split and dE per (row split, tangent group), which is exactly
    what `load` puts in the `panel` column. So the violins and the magnitude
    panels of a row describe the same pixels without either side having to be
    told which.

    `knees` are the two scales of that field's own loss transform, read off the
    run the store holds -- both networks share a configuration digest, so both
    were trained against the same pair.
    """
    out = {}
    for (field, name) in ARM_STORES.items():
        row, axes, per_split = arm_curves(os.path.join(results_dir, name))
        knees = scale_knees(row, field)
        for (split, curves) in per_split.items():
            out[(split, field)] = (axes, curves, knees)
    return out


# The magnitude panels' x lower bound, one per field -- round numbers a bin
# edge away from where the noise floor of each field's own near-zero pixels
# starts dominating.
ROW_XLO = {"E": 1e-5, "dE": 5e-4}


def _pad_linear(values, frac=0.12):
    """A linear-axis range around `values`, `frac` of the span clear either side."""
    finite = values[np.isfinite(values)]
    lo, hi = finite.min(), finite.max()
    pad = (hi - lo) * frac or 0.05
    return (lo - pad, hi + pad)


def _fit_violin_axes(ax, values, decades=0.12):
    """Fit a violin panel's log y axis to its own data, not a stale hand pick.

    A fixed `(ylim, yticks)` per column, tried first, drifts out of date
    exactly when the campaign is rescored and a violin no longer reaches as
    far as the bound that used to fit it -- which is the whitespace this
    exists to remove. Ticks are the bare decades within that range
    (`subs=(1.0,)`) rather than `LogLocator`'s default: a range spanning
    a little over one decade, which every panel here does, is exactly where
    that default inserts a handful of sub-decade ticks (2, 3, 4, 6...) to
    avoid looking sparse -- busier than the one or two labels a hand-picked
    pair used to carry.
    """
    ylim = dlp.pad_log(values.to_numpy(), decades=decades)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    # A range that never crosses a decade boundary (rare, but possible for a
    # very tight cluster) would otherwise carry no major tick at all.
    if not ax.yaxis.get_majorticklocs().size:
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    # `violin`'s own `log_scale=True` leaves seaborn's minor-tick formatter in
    # place, which labels every sub-decade tick (2, 3, 4, 6...) whenever the
    # major ticks alone look sparse to it -- precisely the busy tick band a
    # tightened, sub-decade range now always triggers.
    ax.yaxis.set_minor_formatter(NullFormatter())
    # A single category on a categorical x axis gets matplotlib's default
    # +-0.5 span plus its usual 5% margin on top, which is built for a slot
    # several categories wide -- here it is one, so the same margin reads as
    # dead space flanking the violin on both sides instead. Trimmed to the
    # violin's own half-width (each side of the split fills half of
    # `VIOLIN_WIDTH`) plus a sliver of real clearance.
    half = dlp.VIOLIN_WIDTH / 2 + 0.04
    ax.set_xlim(-half, half)


def _finite_hi_edge(axes, curves):
    """The rightmost bin edge either curve of `curves` still has data in.

    OOD splits are binned over the same edges as ID (one store, one set of
    bins per field) but do not reach every bin -- their curves run out to NaN
    wherever no OOD pixel fell. `magnitude_xlim` uses this to find how far a
    row's axis can honestly extend.
    """
    base, arm = curves["Baseline"][0], curves["DeepLight"][0]
    idx = np.flatnonzero(np.isfinite(base) & np.isfinite(arm))
    return axes.field[idx[-1] + 1] if idx.size else axes.field[-1]


# Colour and linestyle both carry arm identity, redundantly on purpose: a
# greyscale print or a red-green reader still tells the three arms apart.
ALPHA_SWEEP_ARM_COLORS = {"1dir": "#522648", "1_per_sampler":"#AD6938",
                          "2_per_sampler": "#649ca7"}
ALPHA_SWEEP_ARM_LS = {"1dir": "-.", "1_per_sampler": "--", "2_per_sampler": "-"}

# Where the E panels are cropped: the 1dir arm at alpha=10 is enough of an
# outlier (RMSE ~2x the next worst, gain strongly negative) that letting the
# axes autoscale around it flattens every other arm's variation into a thin band
# at the bottom. The point itself is kept, marked at the crop edge rather than
# dropped or left to silently distort the axis.
RMSE_E_CAP = 3.5e-3
GAIN_E_FLOOR = -0.2


def _emphasis_kw(is_emphasized):
    return dict(lw=1.5, alpha=1.0, zorder=3, ms=3.5) if is_emphasized \
        else dict(lw=1.3, alpha=0.8, zorder=1, ms=3.5)


def _annotate_off_scale(ax, x, edge, color, marker):
    """Mark a point clipped to `edge` with an arrow `marker`.

    The true value is not printed: at print size and the 8pt floor the label
    covers the panel; the caption says the arrows are off-scale.
    """
    ax.plot([x], [edge], marker=marker, color=color,
            ms=6, zorder=5, markeredgecolor="white", markeredgewidth=0.5)


# The gain panel's own palette: `dlp.GAIN` for the paper's arm, same as every
# other figure's gain colour, and a desaturated tint of it (same hue, ~1/3 the
# saturation) for the 512-sample run, so the two gain violins still read as
# "the same kind of quantity" rather than as two unrelated series.
GAIN_STYLES = {"DeepLight, 512": "#e9c4bc", "DeepLight, 2048": dlp.GAIN}


# --------------------------------------------------------------------------
# The main results figure, one row per data type and one column per analysis.
#
# Rows are ID operator, ID derivative, OOD operator, OOD derivative; columns are
# the training-set size violins, the relative error against magnitude, and the
# alpha sweep.
# --------------------------------------------------------------------------

# This figure is drawn at its printed width, \textwidth (6.75 in on the spie
# class), so every size below -- type, lines, markers, also in the shared
# helpers -- is its size in print, and all text is the journal's 8pt floor.
FIGSIZE = (6.75, 7.0)
COLUMN_WIDTHS = (1., 1.15, 1.0)   # training size | magnitude | alpha sweep
VIOLIN_GAIN_WIDTHS = (2, 1.3)       # training column: error violins | gain violins
GAIN_VIOLIN_WIDTH = 0.75            # each gain violin, as a share of its slot
MAGNITUDE_HEIGHTS = (1.4, 1)        # magnitude column: curves over gain strip
ALPHA_HEIGHTS = (1.4, 1)            # alpha column: RMSE over gain
WSPACE = 0.42
RIGHT_WSPACE = 0.45                # magnitude column to the alpha column
HSPACE = 0.42                       # between E and dE, within a group
GROUP_HSPACE = 0.17
MARGINS = dict(left=0.09, right=0.98, bottom=0.125, top=0.97)
LETTER_XY = (-0.16, 1.04)           # panel letters (a, b, c, ...), close to their axes

ROWS = (("ID", "E"), ("ID", "dE"), ("OOD", "E"), ("OOD", "dE"))

# The magnitude panels' top-panel y range, one per field and shared between
# its ID and OOD rows.
MAGNITUDE_YLIMS = {"E": (0.0008, 0.20), "dE": (0.003, 0.30)}

BASELINE_COLOR = "0.75"
SIZE_COLORS = {"Baseline, 2048": BASELINE_COLOR,
               "DeepLight, 512": dlp.BASELINE,
               "DeepLight, 2048": dlp.ARM}


def panel_of(split, field):
    """The `panel` key `load` and `arm_strips` use for a row."""
    return split if field == "E" else f"{split}/{split}"


def field_label(field):
    """The row label naming what is predicted: `$E$` or `$\\nabla_v E$`."""
    return r"$E$" if field == "E" else r"$\nabla_v E$"


def size_frame(df, arm_id, base_id):
    """The three training-size networks, labelled, with per-sample gain.

    Baseline and DeepLight at 2048 are the main figure's pair (`paper_runs`),
    so this column and the magnitude column show the same two networks.
    """
    df = pick(df, **OPERATOR)
    small = pick(df, config_digest=SIZE, **ARM)
    runs = {"Baseline, 2048": base_id, "DeepLight, 512": best_seed(small),
            "DeepLight, 2048": arm_id}
    chosen = df[df["run_id"].isin(runs.values())].copy()
    chosen["series"] = chosen["run_id"].map({r: l for (l, r) in runs.items()})
    keys = [k for k in dl.DEFAULT_GAIN_KEYS if k != "seed"]
    return dl.add_gain(chosen, reference={"run_id": base_id}, on=keys)


def size_cell(axs, cell, field, letter):
    """Three error violins and the two non-baseline gain violins."""
    labels = list(SIZE_COLORS)
    dlp.violin(axs[0], cell, x="series", y="rel_err", hue="series",
               order=labels, hue_order=labels,
               palette=list(SIZE_COLORS.values()), log=True)
    _fit_violin_axes(axs[0], cell["rel_err"])
    axs[0].set_xlim(-0.6, len(labels) - 0.4)
    # no category labels: at print size they cannot fit under the violins,
    # and the colours are named by the figure legend
    dlp.style_axis(axs[0], letter=letter, letter_xy=LETTER_XY,
                   ylabel=f"RMSE of {field_label(field)}", xticks=[])

    gains = cell[cell["series"] != labels[0]].dropna(subset=["gain"])
    axs[1].axhline(0, color="0.35", linewidth=0.6)
    dlp.violin(axs[1], gains, x="series", y="gain", hue="series",
               order=labels[1:], hue_order=labels[1:],
               palette=[GAIN_STYLES[l] for l in labels[1:]], log=False,
               width=GAIN_VIOLIN_WIDTH)
    axs[1].set_ylim(-1.1, 1.1)
    axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axs[1].set_xlim(-0.8, len(labels[1:]) - 0.2)
    dlp.style_axis(axs[1], right_axis=True, ylabel="Rel. Gain", xticks=[])


def _rel_gain(value, ref):
    """`1 - value / ref`, NaN where `ref` is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ref > 0, 1 - value / ref, np.nan)


def magnitude_gain(curves):
    """`1 - DeepLight / Baseline` per magnitude bin."""
    return _rel_gain(curves["DeepLight"][0], curves["Baseline"][0])


def magnitude_xlim(axes, curves, field):
    """This row's own x range: the field's floor to the row's last data."""
    return (ROW_XLO[field], _finite_hi_edge(axes, curves))


def magnitude_cell(axs, curves, axes, field, gain_ylim, knees, letter):
    """Both networks' relative median error per magnitude bin, over the gain.

    The top panel divides each curve by the magnitude of its own bin, log-log,
    filled down to its floor so that "DeepLight sits under Baseline" reads at a
    glance. The bottom panel is `1 - DeepLight / Baseline` at each bin, which
    the relative transform does not affect: both curves divide by the same
    magnitude.
    """
    x = dlp.centers(axes.field)
    base_curve = curves["Baseline"][0]
    arm_curve = curves["DeepLight"][0]
    rel_base, rel_arm = base_curve / x, arm_curve / x

    top, bottom = axs
    xlim = magnitude_xlim(axes, curves, field)
    mag_ylim = MAGNITUDE_YLIMS[field]
    floor = mag_ylim[0]
    top.fill_between(x, floor, rel_base, color=BASELINE_COLOR, alpha=0.35,
                     lw=0, zorder=1)
    top.fill_between(x, floor, rel_arm, color=dlp.ARM, alpha=0.45, lw=0,
                     zorder=2)
    dlp.median_line(top, x, rel_base, color=BASELINE_COLOR, lw=1.1,
                    label="Baseline")
    dlp.median_line(top, x, rel_arm, color=dlp.ARM, lw=1.1, label="DeepLight",
                    emphasis=True)
    top.set(xscale="log", yscale="log", xlim=xlim, ylim=mag_ylim)
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    # A range under ~1.5 decades (the OOD panels, at 3%-30%) leaves matplotlib's
    # default log locator sparse enough that it labels the in-between minor
    # ticks too, in its own scientific notation rather than our percent one.
    top.yaxis.set_minor_formatter(NullFormatter())

    bottom.axhline(0, color="0.35", lw=0.6, ls="dashed", zorder=0)
    bottom.plot(x, magnitude_gain(curves), color=dlp.GAIN, lw=1.2, zorder=2)
    bottom.set(xscale="log", xlim=xlim, ylim=gain_ylim)

    # Drop the field subscript ($c_{E}$, $a_{dE}$, ...) from the knee labels:
    # at this column's width it reads as noise, and the field is already
    # named by the row's own label and x axis.
    knees = {re.sub(r"_\{.*?\}", "", label): value
             for (label, value) in knees.items()}
    # labels at the top: the bottom is where the curves' fill sits loudest
    dlp.scale_lines(top, knees, axes.field, labels="top")
    dlp.scale_lines(bottom, knees, axes.field, labels=None)
    dlp.thin_log_ticks(top)
    dlp.thin_log_ticks(bottom)
    # one line: the figure is tall enough for "Rel. Gain" to fit the gain
    # strip's height, which saves the width of a second label line
    dlp.style_axis(top, letter=letter, letter_xy=LETTER_XY, ylabel="Rel. Error")
    top.tick_params(axis="x", which="both", length=0, labelbottom=False)
    dlp.style_axis(bottom, xlabel=magnitude_label(field), ylabel="Rel. Gain")


def alpha_frame(df):
    """`(swept, level)`: best seed per (arm, alpha), and the baseline level.

    In distribution and out of it: gain is against the best baseline network's
    median on the same split and field, taken between the two networks'
    medians, since a point here *is* a network.
    """
    sweep = pick(df, config_digest=SWEEP, **OPERATOR)
    base = per_run(pick(sweep, run_id=best_seed(pick(sweep, **BASELINE))))
    level = base.set_index(["dist", "field"])["rel_err"]

    arms_rows = sweep[~dl.filter_mask(sweep, BASELINE)]
    swept = per_run(arms_rows)
    swept = swept[swept["run_id"].isin(
        best_seed(arms_rows, by=("direction_arm", "alpha")))].copy()
    swept["gain"] = 1 - swept["rel_err"] / np.array(
        [level[k] for k in zip(swept["dist"], swept["field"])])
    return swept, level


def alpha_cell(axs, s, level, crop, letter):
    """RMSE and gain against alpha for every arm, one split and field.

    The paper's arm is drawn heavier and on top, so the other two read as
    context rather than as competing series.
    """
    (cap, floor) = (RMSE_E_CAP, GAIN_E_FLOOR) if crop else (np.inf, -np.inf)
    for arm in [a for a in ALPHA_SWEEP_ARM_COLORS if a in set(s["direction_arm"])]:
        sa = s[s["direction_arm"] == arm].sort_values("alpha")
        color, ls = ALPHA_SWEEP_ARM_COLORS[arm], ALPHA_SWEEP_ARM_LS[arm]
        kw = _emphasis_kw(arm == ARM["direction_arm"])
        x = sa["alpha"].to_numpy()
        rmse_y, gain_y = sa["rel_err"].to_numpy(), sa["gain"].to_numpy()

        over, under = rmse_y > cap, gain_y < floor
        axs[0].plot(x, np.where(over, cap, rmse_y),
                    color=color, ls=ls, marker="o", **kw)
        axs[1].plot(x, np.where(under, floor, gain_y),
                    color=color, ls=ls, marker="o", **kw)
        for xi in x[over]:
            _annotate_off_scale(axs[0], xi, cap, color, marker="^")
        for xi in x[under]:
            _annotate_off_scale(axs[1], xi, floor, color, marker="v")

    axs[0].axhline(level, color="black", lw=1, ls="dotted")
    axs[0].set_yscale("log")
    axs[1].axhline(0, color="black", lw=1, ls="dashed")
    rmse = np.append(s["rel_err"].to_numpy(), level)
    gain = np.append(s["gain"].to_numpy(), 0.0)
    rmse, gain = rmse[rmse <= cap], gain[gain >= floor]
    axs[0].set_ylim(10 ** np.array(_pad_linear(np.log10(rmse))))
    axs[1].set_ylim(_pad_linear(gain))
    if crop:
        axs[0].set_ylim(top=RMSE_E_CAP)
        axs[1].set_ylim(bottom=GAIN_E_FLOOR)
    axs[1].set_xscale("log")
    # every trained alpha ticked, not only the decades the log axis picks
    alphas = np.sort(s["alpha"].unique())
    axs[1].set_xticks(alphas, [f"$10^{{{int(np.log10(a))}}}$" for a in alphas])
    axs[1].xaxis.set_minor_locator(NullLocator())
    # plain decimal major ticks: the RMSE spans less than a decade here, where
    # the log axis would otherwise label its minor ticks in scientific notation
    # -- and none in the top tenth of the axis, where its label would run into
    # the panel letter
    (lo, hi) = axs[0].get_ylim()
    ticks = [t for t in MaxNLocator(nbins=3).tick_values(lo, hi)
             if lo <= t and np.log(t / lo) < 0.9 * np.log(hi / lo)]
    axs[0].yaxis.set_major_locator(FixedLocator(ticks))
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    dlp.style_axis(axs[0], letter=letter, letter_xy=LETTER_XY, ylabel="RMSE")
    axs[0].tick_params(axis="x", which="both", length=0, labelbottom=False)
    dlp.style_axis(axs[1], xlabel=r"Sobolev weight $\alpha$", ylabel="Rel. Gain")


def combined_figure(df, strips):
    """The 4x3 grid: rows are `ROWS`, columns training size | magnitude | alpha."""
    arm_id, base_id = paper_runs(df)
    size = size_frame(df, arm_id, base_id)
    swept, level = alpha_frame(df)

    # The gain strip shares y within a field, over both splits' visible bins:
    # it is a ratio, so ID and OOD read on the same scale. The curve panel
    # above it takes its own fixed range per field instead (`MAGNITUDE_YLIMS`).
    gain_ylims = {}
    for field in ("E", "dE"):
        in_view = []
        for split in SPLIT_LABELS:
            axes, curves, _knees = strips[(panel_of(split, field), field)]
            (lo, hi) = magnitude_xlim(axes, curves, field)
            x = dlp.centers(axes.field)
            in_range = (x >= lo) & (x <= hi)
            in_view.append(magnitude_gain(curves)[in_range])
        gain_ylims[field] = _pad_linear(np.concatenate(in_view))

    f = plt.figure(figsize=FIGSIZE)
    # ID and OOD are two groups of two rows, so the gap between the groups can
    # be wider than the one between E and dE within a group. The magnitude and
    # alpha columns are grouped the same way, sharing a tighter gap than the
    # one to the training column.
    outer = gridspec.GridSpec(2, 1, figure=f, hspace=GROUP_HSPACE, **MARGINS)
    groups = [gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[g],
        width_ratios=(COLUMN_WIDTHS[0], sum(COLUMN_WIDTHS[1:])),
        wspace=WSPACE, hspace=HSPACE) for g in range(2)]

    def cell_spec(r, c):
        """The subplot spec for row `r`, column `c` (0=training, 1=magnitude,
        2=alpha), splitting a group's shared magnitude|alpha slot on demand."""
        (g, rr) = divmod(r, 2)
        if c == 0:
            return groups[g][rr, 0]
        right = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=groups[g][rr, 1], width_ratios=COLUMN_WIDTHS[1:],
            wspace=RIGHT_WSPACE)
        return right[0, c - 1]

    letters = iter("abcdefghijkl")

    for (r, (split, field)) in enumerate(ROWS):
        dist = SPLIT_LABELS[split]
        panel = panel_of(split, field)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=cell_spec(r, 0), width_ratios=VIOLIN_GAIN_WIDTHS,
            wspace=0)
        cell = size[(size["panel"] == panel) & (size["field"] == field)]
        size_cell([f.add_subplot(inner[0, 0]), f.add_subplot(inner[0, 1])],
                  cell, field, next(letters))

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=cell_spec(r, 1), height_ratios=MAGNITUDE_HEIGHTS,
            hspace=0)
        top = f.add_subplot(inner[0, 0])
        axes, curves, knees = strips[(panel, field)]
        magnitude_cell((top, f.add_subplot(inner[1, 0], sharex=top)), curves,
                       axes, field, gain_ylims[field], knees, next(letters))

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=cell_spec(r, 2), height_ratios=ALPHA_HEIGHTS,
            hspace=0)
        top = f.add_subplot(inner[0, 0])
        s = swept[(swept["dist"] == dist) & (swept["field"] == field)]
        alpha_cell((top, f.add_subplot(inner[1, 0], sharex=top)), s,
                   level[(dist, field)],
                   crop=(split == "ID" and field == "E"), letter=next(letters))

    # One distribution label per group, at its top left, rather than one per
    # row -- ID and OOD are the group, E and dE the row within it.
    for g in range(2):
        (split, _field) = ROWS[2 * g]
        box = cell_spec(2 * g, 0).get_position(f)
        f.text(box.x0, box.y1 + 0.012, SPLIT_LABELS[split], ha="left",
               va="bottom", size=dlp.MIN_TEXT, weight="bold")

    network_handles = [Patch(color=color, label=label)
                       for (label, color) in SIZE_COLORS.items()]
    network_handles += [Patch(color=GAIN_STYLES[label],
                              label=f"Gain, {label.split(', ')[1]}")
                        for label in list(SIZE_COLORS)[1:]]
    arm_handles = [Line2D([], [], color=ALPHA_SWEEP_ARM_COLORS[arm],
                          ls=ALPHA_SWEEP_ARM_LS[arm],
                          lw=2.2 if arm == ARM["direction_arm"] else 1.4,
                          label=ARM_PROSE.get(arm, arm))
                   for arm in ALPHA_SWEEP_ARM_COLORS]
    arm_handles.append(Line2D([], [], color="black", ls="dotted",
                              label="baseline"))
    # two rows of four: the networks (and the c, a lines) above the arms
    f.legend(handles=network_handles, frameon=False, fontsize=dlp.MIN_TEXT,
             ncol=len(network_handles), loc="lower center",
             bbox_to_anchor=(0.5, 0.028), columnspacing=1.2)
    f.legend(handles=arm_handles, frameon=False, fontsize=dlp.MIN_TEXT,
             ncol=len(arm_handles), loc="lower center",
             bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    return f


# ----------------------------------------------------- depth-channel ablation
#
# The depth-channel ablation reads its own store (`errors_depth_ext.h5`,
# written by `compute_error.py --depth-above`), not `errors.h5`: it holds
# exactly the four networks this figure needs -- baseline and DeepLight, each
# with and without the depth channel -- scored with an extra margin above the
# skin line into the coupling medium, which `errors.h5`'s depth arrays don't
# carry. One row per (network, depth_channel) combination, so a spec plus the
# `depth_channel` column picks a curve directly; no seed selection needed.
DEPTH_ABOVE = 20                      # pixels above the skin, 0.1mm each
DEPTH_BELOW = 235                     # pixels below the skin
PX_CM = 0.01

# Colour carries the network and line style carries the depth channel, so all
# four curves of a panel read on two independent channels rather than four
# colours.
NETWORK_COLORS = {"Baseline": "0.5", "DeepLight": dlp.ARM}
# label: (depth_channel flag, line style)
DEPTH_CHANNELS = {"with depth channel": (True, dict(ls="-", lw=1.3)),
                  "without depth channel": (False, dict(ls="--", lw=1.1))}


# RMSE only -- `METRIC` is fixed to it everywhere else these stores are
# read, and it is what lets binning sum the raw squared-error and count
# arrays before the sqrt rather than averaging already-reduced RMSE values,
# which is the wrong order for a root-mean-square.
BIN = 4


def binned_rmse(rows, window):
    """RMSE per `BIN`-pixel bin of `window`, pooled before the sqrt.

    Sums `sq_err`/`depth_count` over each bin first and reduces once at the
    end -- an average of the per-pixel RMSE values would not be the RMSE of
    the pooled pixels, the same reason `pooled_depth` pools sums rather than
    curves.
    """
    raw = rows.iloc[0]
    n = ((window.stop - window.start) // BIN) * BIN
    count = raw["depth_count"][window][:n].reshape(-1, BIN).sum(axis=1)
    sq_err = raw["sq_err"][window][:n].reshape(-1, BIN).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, np.sqrt(sq_err / count), np.nan)


DEPTH_CHANNEL_SPLITS = ("ID", "OOD")
DEPTH_CHANNEL_FIELDS = ("E", "dE")


def depth_channel_figure(depth_ext_path):
    """Depth error and gain: baseline and DeepLight, with vs. without depth.

    Includes `DEPTH_ABOVE` pixels above the skin line, into the coupling
    medium, so the panels show the error across the skin boundary rather than
    starting flush at it. Binned `BIN` pixels at a time (`binned_rmse`): a
    single pixel's RMSE is noisy enough at this pixel count to bury the
    depth-channel effect in jitter, which is what the gain row exists to
    show. The top row of each column is the error curves, log scale since
    the depth channel's effect is a fraction of the baseline plateau and gets
    lost against the loud pixels near the skin line on a linear one; the
    bottom row is the gain the depth channel buys at each depth, `(no depth -
    with depth) / no depth`, which is the comparison the top row makes a
    reader trace by eye between two close curves, made explicit. ID and OOD
    get their own pair of columns -- left and right halves of the figure --
    each with its own y scale, since OOD error runs a decade or so above ID's
    and a shared axis would flatten one half of the comparison.
    """
    depth_df = dl.read_depth(depth_ext_path, transform=TRANSFORM)
    window = slice(0, DEPTH_ABOVE + DEPTH_BELOW)
    n_bins = (window.stop - window.start) // BIN
    depth_cm = ((np.arange(n_bins) + 0.5) * BIN - DEPTH_ABOVE) * PX_CM

    columns = [(split, field) for split in DEPTH_CHANNEL_SPLITS
              for field in DEPTH_CHANNEL_FIELDS]
    # drawn at its printed width (0.95\textwidth), so sizes here are print sizes
    f, axs = plt.subplots(2, len(columns), figsize=(6.4, 3.2),
                          layout="constrained", height_ratios=(1.25, 1), sharex=True)
    for (col, (split, field)) in enumerate(columns):
        ax_err, ax_gain = axs[0, col], axs[1, col]
        errs = {}
        for (network, spec) in (("Baseline", BASELINE), ("DeepLight", ARM)):
            for (depth_label, (depth_channel, style)) in DEPTH_CHANNELS.items():
                rows = pick(depth_df, field=field, split=split,
                            depth_channel=depth_channel, **spec)
                errs[(network, depth_label)] = binned_rmse(rows, window)
                ax_err.plot(depth_cm, errs[(network, depth_label)],
                            color=NETWORK_COLORS[network], **style)

            gain = _rel_gain(errs[(network, "with depth channel")],
                             errs[(network, "without depth channel")])
            ax_gain.plot(depth_cm, gain, color=NETWORK_COLORS[network], lw=1.2)

        ax_err.axvline(0, color="0.7", lw=0.8, ls=":", zorder=0)
        ax_err.set_yscale("log")
        ax_err.set_ylim(*dlp.pad_log(np.concatenate(list(errs.values()))))
        dlp.style_axis(ax_err, title=f"{field_label(field)} ({split})", letter="abcd"[col],
                       ylabel="RMSE" if col == 0 else None)

        ax_gain.axvline(0, color="0.7", lw=0.8, ls=":", zorder=0)
        ax_gain.axhline(0, color="0.35", lw=0.6, ls="dashed", zorder=0)
        dlp.style_axis(ax_gain, xlabel="Depth [cm]", letter="efgh"[col],
                       ylabel="Rel. Gain" if col == 0 else None)

    handles = [Line2D([], [], color=NETWORK_COLORS[network], **style,
                      label=f"{network}, {depth_label}")
               for network in ("Baseline", "DeepLight")
               for (depth_label, (_, style)) in DEPTH_CHANNELS.items()]
    f.legend(handles=handles, frameon=False, fontsize=dlp.MIN_TEXT, ncol=2,
             loc="outside lower center")
    return f


# --------------------------------------------------------------------- entry

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="directory to write the figures into")
    p.add_argument("--errors", default=None,
                   help="error .h5 (default: results_dir/errors.h5)")
    p.add_argument("--store", default=None,
                   help="density .h5 (default: results_dir/density.h5)")
    p.add_argument("--store-de", default=None,
                   help="dE loss-transform density .h5 "
                        "(default: results_dir/density_transform_dE.h5)")
    p.add_argument("--depth-ext", default=None,
                   help="depth-channel ablation .h5, scored above the skin "
                        "line (default: results_dir/errors_depth_ext.h5)")
    args = p.parse_args(argv)

    paths = dl.load_paths("./paths.toml")
    results_dir = paths["results_dir"]
    out = args.out or os.path.join(results_dir, "figures")
    errors = args.errors or os.path.join(results_dir, "errors.h5")
    store = args.store or os.path.join(results_dir, "density.h5")
    store_de = args.store_de or os.path.join(results_dir,
                                             "density_transform_dE.h5")
    depth_ext = args.depth_ext or os.path.join(results_dir,
                                               "errors_depth_ext.h5")
    os.makedirs(out, exist_ok=True)

    def write(name, fig, dpi=300):
        target = os.path.join(out, f"{name}.png")
        fig.savefig(target, dpi=dpi)
        plt.close(fig)
        print(f"  wrote {target}")

    df = load(errors)
    write("loss_transform", transform_figure(store, store_de))
    write("quantitative_results_combined",
          combined_figure(df, arm_strips(results_dir)), dpi=600)
    write("seed_checkpoint", seed_checkpoint_figure(df))
    write("depth_channel", depth_channel_figure(depth_ext))


if __name__ == "__main__":
    main()