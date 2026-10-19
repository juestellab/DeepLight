"""Reading the error store, and the few things derived from it at plot time.

`evaluation` computes and writes; nothing here touches a network or the test
data. Every run in a store was scored on the same rows in the same order, so the
derived quantities are elementwise and the row metadata is shared -- which rows
are OOD comes from the store's own `sampler_index`, not from a run.

The tidy frame built here exists for seaborn's benefit: the arrays are the
storage format, and this is the adapter on the way to a figure.
"""
import os
from typing import NamedTuple

import h5py as h5
import numpy as np
import pandas as pd

from .evaluation import RATIO_METRICS

# The reference arm of the campaign: the run trained without a Jacobian term at
# all. `direction_arm="no_deriv"` and `alpha=0` select the same runs -- the
# first says *why* they are the reference.
BASELINE = {"direction_arm": "no_deriv"}

# The run columns that identify a training *configuration* rather than a result.
# A figure pooling runs that differ in one of these, without plotting it, is
# silently averaging over it -- oop=25 with oop=65, say.
CONFIG_FIELDS = ("oop_slices", "depth_channel", "c_E", "c_dE", "norm",
                 "feature_counts", "direction_arm", "alpha")

# What a row is paired with its reference by. `split` is a function of
# `sample_index` in an aligned store, so pairing on the index already pairs
# within a split.
DEFAULT_GAIN_KEYS = ("field", "transform", "metric", "direction",
                     "sample_index", "seed")


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    return value.item() if isinstance(value, np.generic) else value


def _names(values):
    return np.asarray([_decode(v) for v in np.atleast_1d(values)], dtype=object)


def store_meta(path):
    """The store's root attributes, plus the per-row and per-direction labels.

    `split[i]` names the primal sampler that made test row `i`, and is therefore
    the label of every run's sample `i` at once -- which is why splitting is a
    plotting decision rather than something baked into the stored arrays.
    """
    with h5.File(path) as f:
        meta = {k: _decode(v) for k, v in f.attrs.items()}
        meta["sampler_names"] = _names(f.attrs["sampler_names"])
        meta["direction_sampler_names"] = _names(
            f.attrs["direction_sampler_names"])
        meta["split"] = meta["sampler_names"][f["sampler_index"][()]]
        meta["sampler"] = meta["direction_sampler_names"][
            f["direction_sampler_index"][()]]
    return meta


def run_table(path):
    """One row per stored run: its attributes, no per-sample data."""
    if not os.path.exists(path):
        return pd.DataFrame()
    with h5.File(path) as f:
        return pd.DataFrame([{k: _decode(v) for k, v in group.attrs.items()}
                             for group in f.get("runs", {}).values()])


def select_arms(table, baseline=BASELINE, key="val_loss"):
    """The best baseline run and the best non-baseline run, by validation loss.

    Selection reads the store rather than rescanning the results directory, so a
    figure can only ever draw runs that were actually scored -- and the store
    carries the checkpoint path to load the network back from. The
    `train_nl_modes` regimes (`run_id` containing `_nl=`) are excluded from both
    pools: they are no-derivative runs trained under a different `nl` loss
    transform, so their `val_loss` is not on the same scale as the settled-on
    regime's and would win either comparison for the wrong reason.
    """
    table = table[~table["run_id"].str.contains("_nl=")]
    is_baseline = filter_mask(table, baseline)
    best = lambda rows: rows.loc[rows[key].idxmin()]
    return best(table[is_baseline]), best(table[~is_baseline])


# ----------------------------------------------------------------- reading

# The tangent samplers each split is evaluated along (legacy names: SIGN is the
# paper's ID_2, ID_2 its held-out OOD sampler).
MATCHED_SAMPLERS = {"ID": ("ID_1", "SIGN"), "OOD": ("ID_2",)}


def _cells(f, depth):
    """Every `(run attrs, field, transform, group)` in the store.

    `depth=False` yields the per-sample cells, `depth=True` the depth ones --
    both are datasets of the same run group, told apart by their name.
    """
    for group in f["runs"].values():
        attrs = {k: _decode(v) for k, v in group.attrs.items()}
        cells = {tuple(name.split("|")[:2]) for name in group
                 if ("|depth_" in name) == depth and "|" in name}
        for (field, transform) in sorted(cells):
            yield attrs, field, transform, group


def read_errors(path, **filters):
    """Tidy frame: one row per (run, field, transform, metric, sample, direction).

    Columns: run_id, every scalar the run recorded (alpha, seed, direction_arm,
    c_E, ...), field ("E"/"dE"), transform, metric, split, sample_index,
    direction, sampler, err, ref, rel_err.

    `err` is the metric of the residual and `ref` the same metric of the truth
    alone. `rel_err` is *not* `err / ref` row by row, instead
    every `err` of a (run, field, transform, split, sampler) is divided by
    that group's own average norm -- the mean, over its samples, of the
    rmse metric's `ref`; each tangent sampler gets its own scale, as their
    derivatives differ in magnitude. 

    `split` labels the sample axis and `sampler` the direction axis, the latter
    being where the ID_1/ID_2 distinction lives now that every sample carries a
    direction from every tangent sampler.
    """
    meta = store_meta(path)
    frames = []
    with h5.File(path) as f:
        for attrs, field, transform, group in _cells(f, depth=False):
            for metric in RATIO_METRICS + ("psnr",):
                cell = f"{field}|{transform}|{metric}"
                err = np.atleast_2d(group[f"{cell}|err"][()].T).T
                ref = group[f"{cell}|ref"][()].reshape(err.shape)
                n_rows, n_dir = err.shape
                frames.append(pd.DataFrame({
                    **attrs, "field": field, "transform": transform,
                    "metric": metric,
                    "split": np.repeat(meta["split"][:n_rows], n_dir),
                    "sample_index": np.repeat(np.arange(n_rows), n_dir),
                    "direction": np.tile(np.arange(n_dir), n_rows),
                    "sampler": (np.tile(meta["sampler"], n_rows)
                                if field == "dE" else None),
                    "err": err.ravel(), "ref": ref.ravel()}))

    df = pd.concat(frames, ignore_index=True)
    keys = ["run_id", "field", "transform", "split", "sampler"]
    scale = (df[df["metric"] == "rmse"]
             .groupby(keys, dropna=False)["ref"].mean())
    norm = df.set_index(keys).index.map(scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rel_err"] = np.where(
            df["metric"].isin(RATIO_METRICS) & (norm > 0),
            df["err"] / norm, np.nan)
    return _filtered(df, filters)


DEPTH_KEYS = ("abs_err", "abs_ref", "sq_err", "sq_ref")


def read_depth(path, **filters):
    """Depth profiles: one row per (run, field, transform, split).

    The metric is *not* an axis here -- the raw sums are, and `pooled_depth`
    reduces them. `depth_count` is the number of pixels behind each sum.

    dE carries a direction axis between the split and the depth; it is summed
    over the directions of the split's own samplers (`MATCHED_SAMPLERS`), and
    the count is scaled by their number to match.
    """
    meta = store_meta(path)
    rows = []
    with h5.File(path) as f:
        for attrs, field, transform, group in _cells(f, depth=True):
            values = {key: group[f"{field}|{transform}|depth_{key.split('_')[0]}"
                                 f"|{key.split('_')[1]}"][()] for key in DEPTH_KEYS}
            for (s, split) in enumerate(meta["sampler_names"]):
                sums, n_dir = {}, 1
                for (key, value) in values.items():
                    if value.ndim == 3:                 # (S, K, D)
                        k = np.isin(meta["sampler"], MATCHED_SAMPLERS[split])
                        n_dir = int(k.sum())
                        sums[key] = value[s, k].sum(axis=0)
                    else:
                        sums[key] = value[s]
                rows.append({**attrs, "field": field, "transform": transform,
                             "split": split, **sums,
                             "depth_count": group["depth_count"][s] * n_dir})
    return _filtered(pd.DataFrame(rows), filters)


class DensityAxes(NamedTuple):
    """The two axes a density was binned on: |field|, and |error| below it."""
    field: np.ndarray
    error: np.ndarray


def read_density(path, **filters):
    """`(df, axes)` from a density store: one row per (run, split).

    The (nE, nErr) counts ride as a cell of the `density` column, as the depth
    sums do -- the array is the storage format and the frame is only what
    carries the run metadata beside it. `median`, `mean` and `median_count`
    ride beside them the same way: the exact median and mean |error| per
    |field| bin, taken over the pixels at scoring time rather than read back
    off the counts, and the pixel count both rest on.
    """
    with h5.File(path) as f:
        if "err_edges" not in f.attrs or "mean" not in next(
                iter(f["runs"].values())):
            raise ValueError(
                f"{path} predates the mean curves and the separate error "
                "axis. Rescore it with run/compute_density.py.")
        axes = DensityAxes(np.asarray(f.attrs["edges"]),
                           np.asarray(f.attrs["err_edges"]))
        splits = _names(f.attrs["sampler_names"])
        rows = [{**{k: _decode(v) for k, v in group.attrs.items()},
                 "split": split, "density": group["density"][s],
                 "median": group["median"][s], "mean": group["mean"][s],
                 "median_count": group["median_count"][s]}
                for group in f["runs"].values()
                for (s, split) in enumerate(splits)]
    return _filtered(pd.DataFrame(rows), filters), axes


def pooled_density(df, normalize: str | None = "column"):
    """Sum `df`'s densities into one (nE, nErr) grid.

    `normalize="column"` divides each |E| column by its own total, which is what
    makes the panel a distribution *given* the magnitude: E's own histogram is
    steeply peaked, and raw counts would draw that peak rather than the error.
    Columns holding no pixels are left as NaN rather than 0/0. `None` keeps the
    counts.
    """
    total = np.sum(np.stack(df["density"].to_numpy()), axis=0)
    if normalize is None:
        return total
    with np.errstate(divide="ignore", invalid="ignore"):
        column = total.sum(axis=1, keepdims=True)
        return np.where(column > 0, total / column, np.nan)


def pooled_median(df):
    """`(median, low, high)` across `df`'s runs, per |field| bin.

    Medians do not add the way counts do, so a group of runs cannot be pooled
    into one exact median from what the store keeps. A group is the seeds of one
    configuration, so they are pooled as the geometric mean of their curves --
    the mean of a log-scaled quantity is a log-scaled mean -- with the envelope
    of the group beside it. A single run leaves `low == high == median`, and a
    bin no run reported stays NaN in all three.
    """
    curves = np.stack(df["median"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        empty = np.all(np.isnan(curves), axis=0)
        safe = np.where(empty, 1.0, curves)         # keep nanmean off all-NaN
        mean = np.where(empty, np.nan, np.exp(np.nanmean(np.log(safe), axis=0)))
        return (mean, np.where(empty, np.nan, np.nanmin(safe, axis=0)),
                np.where(empty, np.nan, np.nanmax(safe, axis=0)))


def pooled_mean(df):
    """`(mean, low, high)` across `df`'s runs, per |field| bin.

    Unlike medians, means pool *exactly*: the store keeps the pixel count each
    one rests on, so a group's mean is the count-weighted mean of its runs'
    means, not an approximation of one. The envelope is still the group's
    min/max, and a single run leaves `low == high == mean`.
    """
    curves = np.stack(df["mean"].to_numpy())
    counts = np.stack(df["median_count"].to_numpy()).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        empty = np.all(np.isnan(curves), axis=0)
        safe = np.where(empty, 1.0, curves)
        weight = np.where(np.isnan(safe), 0.0, counts)
        total = weight.sum(axis=0)
        pooled = (np.nan_to_num(safe) * weight).sum(axis=0) / np.where(
            total > 0, total, 1.0)
        blank = empty | (total <= 0)
        return (np.where(blank, np.nan, pooled),
                np.where(empty, np.nan, np.nanmin(safe, axis=0)),
                np.where(empty, np.nan, np.nanmax(safe, axis=0)))


DEPTH_METRICS = {"mae": ("abs", lambda total, count: total / count),
                 "rmse": ("sq", lambda total, count: np.sqrt(total / count))}


def pooled_depth(df, metric="rmse"):
    """Pool `df`'s depth rows into one `(err, ref, count)` curve triple.

    Use after filtering, e.g. to merge several splits into one profile. The sums
    are added first and the reduction -- the sqrt of an RMSE -- applied exactly
    once, at the end. psnr has no pooled form here: it would need a peak the
    store does not keep against depth.
    """
    power, reduce = DEPTH_METRICS[metric]
    pool = lambda key: np.sum(np.stack(df[key].to_numpy()), axis=0)
    count = pool("depth_count")
    with np.errstate(divide="ignore", invalid="ignore"):
        curves = [np.where(count > 0, reduce(pool(f"{power}_{side}"), count),
                           np.nan)
                  for side in ("err", "ref")]
    return curves[0], curves[1], count


# ----------------------------------------------------------------- derived

def filter_mask(df, filters):
    """Boolean mask of the rows matching every `column=value` in `filters`.

    A list/tuple/set value matches any of its entries. Shared by the filters,
    `add_gain`'s reference and `arm_column`, so all of them speak the same
    vocabulary.
    """
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key not in df.columns:
            raise KeyError(f"{key} is not a column; have {sorted(df.columns)}")
        mask &= (df[key].isin(list(value))
                 if isinstance(value, (list, tuple, set)) else df[key] == value)
    return mask


def _filtered(df, filters):
    return df[filter_mask(df, filters)].reset_index(drop=True)


def arm_column(df, baseline, names=("Baseline", "DeepLight"), column="arm"):
    """Label every row as the baseline arm or the comparison arm.

    `baseline` is the same filter dict `add_gain` takes, so the arm column and
    the gain column cannot disagree about which rows are the reference.
    """
    out = df.copy()
    out[column] = np.where(filter_mask(df, baseline), names[0], names[1])
    return out


def sample_median(df, by, value="rel_err"):
    """Collapse the per-sample axis, leaving one number per group."""
    return df.groupby(list(by), dropna=False)[value].median().reset_index()


def add_gain(df, reference, on=DEFAULT_GAIN_KEYS, value="rel_err"):
    """Attach the relative gain of each row against a reference run.

    `on` is what a row is paired with its reference by: including "seed" pairs
    each network against the *same seed's* baseline, dropping it pools the
    baseline across seeds. That choice is why gain is not precomputed.

    Pairing on a key the reference cannot match -- "seed", against a frame
    holding one run per arm at *different* seeds -- would leave every gain NaN,
    so it is reported instead.
    """
    keys = [k for k in on if k in df.columns]
    ref_values = (_filtered(df, reference).groupby(keys, dropna=False)[value]
                  .mean().rename(f"{value}_ref").reset_index())
    merged = df.merge(ref_values, on=keys, how="left")

    compared = merged.loc[~filter_mask(merged, reference), f"{value}_ref"]
    if len(compared) and not compared.notna().any():
        raise LookupError(
            f"no compared row found a reference: pairing on {keys} against "
            f"{reference} matched nothing. Drop the keys the reference cannot "
            f"match (usually 'seed') from `on`.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = merged[f"{value}_ref"]
        merged["gain"] = np.where(ref > 0, (ref - merged[value]) / ref, np.nan)
    return merged
