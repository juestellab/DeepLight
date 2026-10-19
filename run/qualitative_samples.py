"""Qualitative sample panels: ground truth and the two models' error fields.

Model selection and the metric come from the error store, so this shares one
definition of `nl` and one notion of "the best run in an arm" with
run/compute_error.py and run/results_figure.py.

The store is scored in the test file's row order, so the sample a gain quantile
picks out is a row of that file and indexes it directly -- the position map this
used to carry, translating a global index into a per-split dataset, is gone with
the per-split storage.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import cmasher as cmr

import deeplight as dl

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_autotune_level=0 '
)

import plotting as dlp

dlp.use_paper_style()


# Rows of the paper figure: (split, tangent sampler, derivative?). The two
# in-distribution derivative rows used to be two row splits, ID_1 and ID_2;
# every sample now carries a direction from every tangent sampler, so they are a
# selection along the direction axis instead of a different set of test rows.
PANELS = (("ID", None, False),
          ("OOD", None, False),
          ("ID", "ID_1", True),
          ("ID", "SIGN", True),
          ("OOD", "ID_2", True))

# row labels: the derivative rows of ID name their sampler (legacy names: SIGN
# is the paper's ID_2, ID_2 its OOD sampler), all others the split
ROW_LABELS = {"ID_1": r"ID$_1$", "SIGN": r"ID$_2$"}


def quantile_sample(gain_df, split, quantile=0.5, field="E", sampler=None):
    """The (row, direction) at `quantile` of the gain distribution.

    `sampler` narrows the derivative rows to one tangent sampler; the operator
    rows have no direction axis and get direction 0.
    """
    sub = gain_df[(gain_df["split"] == split) & (gain_df["field"] == field)]
    if sampler is not None:
        sub = sub[sub["sampler"] == sampler]
    sub = sub.dropna(subset=["gain"]).sort_values("gain")
    if sub.empty:
        raise LookupError(f"no gain values for split={split} field={field} "
                          f"sampler={sampler}")
    row = sub.iloc[max(int(len(sub) * quantile) - 1, 0)]
    return int(row["sample_index"]), int(row["direction"]), row


def operator_sample(ds, apply_arm, apply_base, index):
    """Ground truth, arm prediction and baseline prediction of one test row."""
    mu, _, E, _ = ds[index]
    primal = mu[jnp.newaxis, :, :, :]
    return (E[:, :, 0], apply_arm(primal)[0, :, :, 0],
            apply_base(primal)[0, :, :, 0])


def deriv_sample(ds, apply_arm, apply_base, index, direction):
    """The same for one directional derivative of one test row.

    A row carries K directions, so the direction has to be chosen as well as the
    row -- which is also what makes the ID_1 and ID_2 panels differ.
    """
    mu, dmu, _, dE = ds[index]
    primal, tangent = mu[jnp.newaxis, ...], dmu[direction][jnp.newaxis, ...]
    _, dE_arm = jax.jvp(apply_arm, (primal,), (tangent,))
    _, dE_base = jax.jvp(apply_base, (primal,), (tangent,))
    return (dE[direction, :, :, 0], dE_arm[0, :, :, 0],
            dE_base[0, :, :, 0])


def panel_sample(gain_df, ds, apply_arm, apply_base, panel, quantile):
    """`(gt, arm, base, errors)` for one panel spec, at a gain quantile.

    `errors` is `(baseline, arm)`: the sample's scored error from the store,
    the normalized RMSE of the transformed fields the sample was selected by.
    """
    split, sampler, deriv = panel
    field = "dE" if deriv else "E"
    index, direction, row = quantile_sample(gain_df, split, quantile, field, sampler)
    if deriv:
        fields = deriv_sample(ds, apply_arm, apply_base, index, direction)
    else:
        fields = operator_sample(ds, apply_arm, apply_base, index)
    return (*fields, (row["rel_err_ref"], row["rel_err"]))


def _loss_str(x):
    return np.format_float_scientific(x, precision=1, exp_digits=1,
                                      unique=False)


def _log_exponent_formatter():
    """Tick labels for an axis whose values already *are* log10(x)."""
    return FuncFormatter(lambda x, _: f"$10^{{{x:g}}}$")


def _plot_row(f, ax, labels, gt, arm, base, errors, deriv):
    """One sample: the field, the baseline error and the arm error.

    `ax` is the sample's three axes, a column of the figure, with the colorbars
    under the field and under the two error panels.

    The operator (E) rows are shown in log10 space: E spans too many decades
    for a soft norm to make deep-tissue contrast readable, but log10(E) is a
    plain, single transform a reader can follow.
    """
    if deriv:
        cmap = cmr.get_sub_cmap("twilight_shifted", 0.1, 0.9)
        hm = ax[0].matshow(gt.T, cmap=cmap, norm=CenteredNorm(0, jnp.percentile(jnp.abs(gt), 99.0)))
    else:
        floor = 1e-6
        log_gt = jnp.log10(jnp.clip(gt, floor, None))
        vmax = float(jnp.percentile(log_gt, 99.9))
        vmin = max(-5, jnp.min(log_gt))
        hm = ax[0].matshow(np.asarray(log_gt.T), cmap="inferno", norm="linear",
                           vmin=vmin, vmax=vmax)
    base_diff = gt - base
    arm_diff = gt - arm
    dnorm = CenteredNorm(0, jnp.maximum(jnp.nanpercentile(jnp.abs(arm_diff), 99.),
                                        jnp.nanpercentile(jnp.abs(base_diff), 99.)))
    error_cmap = "PuOr"
    ax[1].matshow((base_diff).T, cmap=error_cmap, norm=dnorm)
    ax[2].matshow((arm_diff).T, cmap=error_cmap, norm=dnorm)
    for j in range(3):
        ax[j].set_axis_off()
        ax[j].text(0.05, 0.8, next(labels), transform=ax[j].transAxes, size=10,
                   color="white" if (j == 0 and not deriv) else "black")
    ax[1].text(10, 260, f"RMSE={_loss_str(errors[0])}", color="black")
    ax[2].text(10, 260, f"RMSE={_loss_str(errors[1])}", color="black")
    # full panel width (shrink=1), so the bar's ends line up with the
    # panel's edges; a high aspect keeps it thin
    cb0 = f.colorbar(hm, ax=ax[0], location="bottom", aspect=30, shrink=1.0)
    cb1 = f.colorbar(ax[2].images[0], ax=[ax[1], ax[2]], location="bottom",
                     aspect=30, shrink=1.0)
    # three ticks: more do not fit under a fifth of the page width
    for cb in (cb0, cb1):
        cb.ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=deriv or cb is cb1))
    if not deriv:
        cb0.ax.xaxis.set_major_formatter(_log_exponent_formatter())
        # one shared exponent at the end of the bar: three full
        # "1.5x10^-3" labels do not fit under a fifth of the page width
        offset = ScalarFormatter(useMathText=True)
        offset.set_powerlimits((0, 0))
        cb1.ax.xaxis.set_major_formatter(offset)


def qualitative_samples_combined(gain_df, ds, apply_arm, apply_base, quantile):
    """The paper figure: one column per sample, rows ground truth, baseline
    error and DeepLight error, with the colorbars under each column.

    Letters run down the columns, so each sample keeps a consecutive triple
    (a-c, d-f, ...). `quantile` picks each sample from its gain distribution:
    0.5 is the median of the main figure, 0 and 1 the smallest and largest gain.
    """
    labels = iter("abcdefghijklmnopqrstuvwxyz")

    # drawn at its printed width (\textwidth), so sizes here are print sizes
    f, axs = plt.subplots(3, len(PANELS), figsize=(6.75, 3.9),
                          layout="constrained")
    for (col, panel) in enumerate(PANELS):
        (split, sampler, deriv) = panel
        sample = panel_sample(gain_df, ds, apply_arm, apply_base, panel,
                              quantile)
        _plot_row(f, axs[:, col], labels, *sample, deriv=deriv)
        field = r"$\nabla_v E$" if deriv else r"$E$"
        axs[0, col].set_title(f"{ROW_LABELS.get(sampler, split)}: {field}", y=1.01)

    for (ax, text) in zip(axs[:, 0], ("Ground truth", "Baseline error",
                                      "DeepLight error")):
        ax.text(-0.06, 0.5, text, rotation="vertical", ha="right", va="center",
                transform=ax.transAxes)
    return f


def main():
    # `cfg` deliberately not called `paths`: the previous version rebound that
    # name to the tuple from get_optimal_runs and then subscripted it as a dict,
    # which was a guaranteed TypeError.
    cfg = dl.load_paths('./paths.toml')
    errors_path = os.path.join(cfg['results_dir'], "errors.h5")
    out_dir = os.path.join(cfg['results_dir'], "figures")
    os.makedirs(out_dir, exist_ok=True)

    baseline_run, arm = dl.select_arms(dl.run_table(errors_path),
                                       baseline=dl.BASELINE)
    print(f"baseline: {baseline_run['run_id']}\narm     : {arm['run_id']}")

    apply_base = dl.load_model_fn(baseline_run["checkpoint"], jax.devices()[0])
    apply_arm = dl.load_model_fn(arm["checkpoint"], jax.devices()[0])

    meta = dl.store_meta(errors_path)

    # the two arms are picked on validation loss and need not share a seed, so
    # the sample is paired with the baseline's *same row*, not its same seed
    pair_on = tuple(k for k in dl.DEFAULT_GAIN_KEYS if k != "seed")
    gain_df = dl.add_gain(
        dl.read_errors(errors_path, transform="nl_train", metric="rmse",
                       run_id=[baseline_run["run_id"], arm["run_id"]]),
        reference=dl.BASELINE, on=pair_on)
    gain_df = gain_df[gain_df["run_id"] == arm["run_id"]]

    # the samples must come from the data the runs were scored on, which the
    # store records; sample_index is a row of that file
    ds = dl.DeepLightDataset(*dl.load_data(meta["data_file"],
                                          int(arm["oop_slices"])))

    # the median sample is the main figure, the extremes go to the appendix
    for (name, quantile) in (("qualitative_revision", 0.5),
                             ("qualitative_min_gain", 0.),
                             ("qualitative_max_gain", 1.)):
        f = qualitative_samples_combined(gain_df, ds, apply_arm, apply_base,
                                         quantile)
        target = os.path.join(out_dir, f"{name}.png")
        f.savefig(target, dpi=600)
        plt.close(f)
        print("wrote", target)


if __name__ == "__main__":
    main()
