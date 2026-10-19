"""Score trained networks on the test set and write the error store.

Every run is scored on every row of the test file, in file order, so entry `i`
of one run's errors and entry `i` of another's are the same tissue sample: a
gain against a reference run is an elementwise ratio, and which rows are
out-of-distribution is the test file's own `sampler_index`, copied in beside the
errors and applied at plot time.

Two axes are crossed inside the batch loop. A *transform* reweights truth and
prediction alike before they are compared; a *metric* reduces the result to one
number per image:

    E   {linear, nl_train, log} x {mae, rmse, psnr}
    dE  {linear, nl_train}      x {mae, rmse, psnr}

`log` is E-only -- E is positive, dE is signed and roughly half negative. `nl`
is odd and needs no such exception, so its three variants are shared and differ
only in `(a, c)`.

Both the metric of the error and the same metric of the truth alone are stored,
so a relative error is a ratio of stored numbers rather than a rescoring.

Beside them, the error against depth below the skin line, kept as raw sums of
|x| and x**2 per row split rather than as finished curves: pooling rows has to
happen before the sqrt of an RMSE, and an average of averages is not an average.

The store, for N rows, K directions, D depths and S row splits:

    /   attrs: data_file, sampler_names, direction_sampler_names, log_floor
        sampler_index            (N,)
        direction_sampler_index  (K,)
        runs/<run_id>/  attrs: everything the run recorded, plus n_scored
            depth_count                                   (S, D)
            <field>|<transform>|<metric>|<err|ref>         (N,) | (N, K)
            <field>|<transform>|depth_<abs|sq>|<err|ref>   (S, D) | (S, K, D)

Beside that store, and written separately, the *magnitude density*: how the
error of E is distributed against the magnitude of E itself, pixel by pixel
rather than image by image. It answers a different question from the metrics
above -- not how large a run's error is, but where in E's dynamic range it sits
-- which is what the linear/log/nl training comparison turns on. See
`score_density`.

Reading both back is `deeplight.results`; this module only computes and writes.
"""
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np

from .dataloader import load_data, load_direction_groups, load_sample_groups, get_skin_index
from .serialization import load_model_fn, load_metadata
from .utils import nl

LOG_FLOOR = 1e-6
IMAGE_AXES = (1, 2, 3)


TRANSFORMS = {
    "E": {"linear": lambda x: x,
          "log": lambda x: jnp.log(jnp.maximum(x, LOG_FLOOR))},
    "dE": {"linear": lambda x: x},
}


def transforms(run):
    """`{field: {name: fn}}`: the fixed ones, plus the run's own training `nl`"""
    out = {}
    out['E'] = dict(TRANSFORMS['E'], nl_train=lambda x: nl(x, float(run["a_E"]), float(run["c_E"])))
    out['dE'] = dict(TRANSFORMS['dE'], nl_train=lambda x: nl(x, float(run["a_dE"]), float(run["c_dE"])))
    return out


def _rmse(err, gt):
    return jnp.sqrt(jnp.mean(err ** 2, axis=IMAGE_AXES))


METRICS = {
    "mae": lambda err, gt: jnp.mean(jnp.abs(err), axis=IMAGE_AXES),
    "rmse": _rmse,
    "psnr": lambda err, gt: 20 * jnp.log10(
        jnp.max(jnp.abs(gt), axis=IMAGE_AXES) / _rmse(err, gt)),
}
RATIO_METRICS = ("mae", "rmse")


# ----------------------------------------------------------------- scoring

def align_to_skin(field, skin, above=0):
    """(B, X, above + Z) shifted so index `above` is the skin line."""
    z = skin[..., None] - above + jnp.arange(above + field.shape[-1])
    inside = (z >= 0) & (z < field.shape[-1])
    return jnp.where(inside, jnp.take_along_axis(
        field, jnp.where(inside, z, 0), axis=2), 0.0)


def depth_count(skin, n_depths, above=0):
    """(B, above + D) columns still inside the grid at each aligned depth."""
    z = skin[..., None] - above + jnp.arange(above + n_depths)
    return jnp.sum((z >= 0) & (z < n_depths), axis=1)


def batch_eval(gt, pred, skin, field_transforms, above=0):
    out = {}
    for name, transform in field_transforms.items():
        t_gt = transform(gt)
        err = transform(pred) - t_gt
        for metric_name, metric in METRICS.items():
            out[f"{name}|{metric_name}|err"] = metric(err, t_gt)
            out[f"{name}|{metric_name}|ref"] = metric(t_gt, t_gt)
        for side, x in (("err", err), ("ref", t_gt)):
            aligned = align_to_skin(x[..., 0], skin, above)
            out[f"{name}|depth_abs|{side}"] = jnp.sum(jnp.abs(aligned), axis=1)
            out[f"{name}|depth_sq|{side}"] = jnp.sum(aligned ** 2, axis=1)
    return out


def make_scorer(apply_fn, field_transforms, depth_above=0):
    @jax.jit
    def scorer(mu, dmu, E, dE, skin):
        E_pred, apply_lin = jax.linearize(apply_fn, mu)
        dE_pred = jax.vmap(apply_lin, in_axes=1, out_axes=1)(dmu)

        out = {f"E|{k}": v for k, v in batch_eval(
            E, E_pred, skin, field_transforms["E"], depth_above).items()}

        per_direction = jax.vmap(
            lambda g, p: batch_eval(g, p, skin, field_transforms["dE"], depth_above),
            in_axes=1, out_axes=1)(dE, dE_pred)
        out.update({f"dE|{k}": v for k, v in per_direction.items()})
        return out, depth_count(skin, E.shape[2], depth_above)
    return scorer


# -------------------------------------------------------------- runs & data

def load_run(path, criterion="operator"):
    run_dir = Path(path).resolve()
    checkpoint = run_dir / f"min_{criterion}_val.h5"

    stored = load_metadata(str(checkpoint))
    config = stored["model_config"]
    run = {"run_id": f"{run_dir.name}_ckpt={criterion}", "run_dir": str(run_dir), "checkpoint": str(checkpoint),
           "val_criterion": criterion, "norm": config["norm"], "n_groups": config["n_groups"],
           "feature_counts": ",".join(str(int(f)) for f in config["feature_counts"]),
           **{k: v for k, v in stored["metadata"].items() if isinstance(v, (int, float, str, bool))}}
    return run


_CACHE = {}


def load_test_data(data_path, oop_slices):
    key = (os.path.abspath(data_path), int(oop_slices))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = (*load_data(data_path, oop_slices),
                       get_skin_index(data_path))
    return _CACHE[key]


def batches(arrays, batch_size):
    n_rows = arrays[0].shape[0]
    for start in range(0, n_rows, batch_size):
        rows = np.arange(start, min(start + batch_size, n_rows))
        idx = np.concatenate([rows, np.full(batch_size - len(rows), rows[-1])])
        yield [jax.device_put(arr[idx]) for arr in arrays], rows


def score_run(run, data_path, batch_size=8, n_samples=None, device=None,
              depth_above=0):
    *data, skin = load_test_data(data_path, int(run["oop_slices"]))
    n_scored = min(n_samples or data[0].shape[0], data[0].shape[0])
    split_index, sampler_names = load_sample_groups(data_path)
    split_index = np.asarray(split_index[:n_scored])

    scorer = make_scorer(load_model_fn(run["checkpoint"], device),
                         transforms(run), depth_above)

    scalars, depth = defaultdict(list), defaultdict(float)
    counts = 0
    for batch, rows in batches([arr[:n_scored] for arr in (*data, skin)],
                               batch_size):
        out, count = jax.device_get(scorer(*batch))
        masks = [split_index[rows] == s for s in range(len(sampler_names))]
        # every per-split accumulation is this: drop the padding, then sum the
        # rows of each split into the running (S, ..., D) total
        pool = lambda v: np.stack([v[:len(rows)][m].sum(axis=0) for m in masks])

        counts = counts + pool(count)
        for name, value in out.items():
            if "|depth_" in name:
                depth[name] = depth[name] + pool(value)
            else:
                scalars[name].append(value[:len(rows)])

    return {"depth_count": counts,
            **{name: np.concatenate(chunks) for name, chunks in scalars.items()},
            **depth}


# -------------------------------------------------------------------- store

def write_run(h5_path, run, scores, data_path):
    """Append one run's scores to the store, replacing it if already there."""
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)) or ".", exist_ok=True)
    with h5.File(h5_path, "a") as f:
        if "sampler_index" not in f:
            # the row and direction provenance travels with the errors, so a
            # figure never has to reopen the test file to know which rows are OOD
            sample_index, sampler_names = load_sample_groups(data_path)
            direction_index, direction_names = load_direction_groups(data_path)
            f.attrs.update(data_file=os.path.abspath(data_path),
                           sampler_names=list(sampler_names),
                           direction_sampler_names=list(direction_names),
                           log_floor=LOG_FLOOR)
            f.create_dataset("sampler_index",
                             data=np.asarray(sample_index, np.int32))
            f.create_dataset("direction_sampler_index",
                             data=np.asarray(direction_index, np.int32))

        runs = f.require_group("runs")
        if run["run_id"] in runs:
            del runs[run["run_id"]]
        group = runs.create_group(run["run_id"])
        group.attrs.update(run)
        group.attrs["n_scored"] = int(scores["E|linear|rmse|err"].shape[0])
        for name, value in scores.items():
            group.create_dataset(name, data=np.asarray(value))


# --------------------------------------------------------- magnitude density

# The grid |E| and |error| are both binned onto. Fixed here rather than per run,
# because the point is to lay runs trained under different transforms over one
# another; and shared between the axes so that a bin is the same decade on
# both. It starts at the log floor -- below it E is physically nothing -- and
# out-of-range values are dropped rather than clipped into the end bins: an
# error under the floor is already indistinguishable from zero, and piling the
# whole tail into one bin would put a spike there that dominates the colour
# scale. The counts are therefore of the pixels that landed, not of every pixel
# scored.
EDGES = {"E": np.geomspace(LOG_FLOOR, 5.0, 71),
         # dE reaches further either way: 1% of its pixels are under 1e-5 and
         # its loudest are ten times E's
         "dE": np.geomspace(1e-7, 20.0, 71)}


def density_splits(data_path, field, tangent_groups=None):
    """`(row_index, direction_index, names)` the density of `field` is split on.

    E is split by the row sampler, as the error store is, and has no direction
    axis to split: `direction_index` is None.

    dE is split by both. Which tissue a derivative was taken on and which
    direction it was taken along are independent questions -- a tangent sampler
    held out of training is out of distribution whether the row it sits on was
    trained on or not -- so a dE split is a `(row sampler, tangent group)` pair,
    named `"<row>/<group>"` and numbered row-major. The error store carries the
    same cross product, which is what lets a figure put the two beside each
    other.

    `tangent_groups` is `{group name: tangent samplers}` and merges samplers
    into one split; the default gives each its own. Merging here rather than at
    plot time is the point of it: the median of a bin is taken over its pixels,
    and pixels can only be pooled before that median is taken -- two medians
    cannot be averaged back into the one their union would have had.
    """
    rows, row_names = load_sample_groups(data_path)
    if field == "E":
        return np.asarray(rows), None, tuple(row_names)
    directions, direction_names = load_direction_groups(data_path)
    groups = dict(tangent_groups or {n: (n,) for n in direction_names})
    of_group = {tangent: g for (g, members) in enumerate(groups.values())
                for tangent in members}
    missing = sorted(set(direction_names) - set(of_group))
    if missing:
        raise ValueError(
            f"tangent samplers {missing} belong to no group; every sampler the "
            f"file stores must be placed, or its pixels vanish silently. "
            f"Groups given: {dict(groups)}")
    return (np.asarray(rows),
            np.asarray([of_group[direction_names[d]] for d in directions]),
            tuple(f"{r}/{g}" for r in row_names for g in groups))


# A bin holding fewer pixels than this reports no median: the thin ends of E's
# range are a handful of pixels of one phantom, and their median is noise drawn
# at full width beside decades that rest on millions.
MEDIAN_MIN_PIXELS = 500


def error_edges(edges):
    return np.asarray(edges) / 10


def magnitude_density(gt, pred, edges):
    """(nE, nErr) pixel counts of (|truth|, |prediction - truth|).

    Everything given is pooled into one histogram; splitting the rows is the
    caller's business. Values outside the axes fall out of the histogram, which
    is `np.histogram2d`'s own behaviour and the wanted one here.
    """
    magnitude = lambda x: np.abs(np.asarray(x)).ravel()
    return np.histogram2d(magnitude(gt), magnitude(pred - gt),
                          bins=(edges, error_edges(edges)))[0]


def magnitude_bin(gt, pred, edges):
    """`(index, error)`: the |truth| bin and |error| of every in-range pixel.

    The pair the exact median is taken over, and deliberately not symmetric with
    `magnitude_density`: a pixel is dropped for the magnitude it is binned at,
    never for the size of its error. Dropping the loud tail is exactly what
    would bias the statistic this feeds.
    """
    index = np.digitize(np.abs(np.asarray(gt)).ravel(), edges) - 1
    keep = (index >= 0) & (index < len(edges) - 1)
    error = np.abs(np.asarray(pred) - np.asarray(gt)).ravel()
    return index[keep].astype(np.int16), error[keep].astype(np.float32)


def pooled_curves(chunks, n_bins, min_pixels=MEDIAN_MIN_PIXELS):
    """`(median, mean, count)` per bin, over one split's `magnitude_bin` chunks.

    Both are taken over the pixels themselves rather than read off the density's
    own error axis, where either could only ever be a bin centre -- 1.3x apart
    on that grid, which is wider than the effect these curves are drawn to show.
    Sorting once by bin turns 140 statistics into one argsort and 70 slices.

    The two answer different questions of a heavy-tailed error: the median is
    what a typical pixel does, the mean is what the total error is made of, and
    on this data they part company by more than either differs between runs. The
    same `min_pixels` gate reports both, so the two curves are always drawn on
    the same bins and a gap in one is a gap in the other.
    """
    index = np.concatenate([c[0] for c in chunks])
    error = np.concatenate([c[1] for c in chunks])
    order = np.argsort(index, kind="stable")
    index, error = index[order], error[order]
    bins = np.arange(n_bins)
    starts = np.searchsorted(index, bins, side="left")
    ends = np.searchsorted(index, bins, side="right")
    median = np.full(n_bins, np.nan)
    mean = np.full(n_bins, np.nan)
    for (i, (a, b)) in enumerate(zip(starts, ends)):
        if b - a >= min_pixels:
            median[i] = np.median(error[a:b])
            mean[i] = np.mean(error[a:b])
    return median, mean, (ends - starts).astype(np.int64)


class Density(NamedTuple):
    """One run's magnitude density and the curves taken beside it."""
    counts: np.ndarray          # (S, n, nErr) pixel counts
    median: np.ndarray          # (S, n) exact median |error| per |field| bin
    mean: np.ndarray            # (S, n) exact mean |error| per |field| bin
    n_pixels: np.ndarray        # (S, n) pixels behind each statistic
    n_scored: int


def score_density(run, data_path, field="E", batch_size=8, n_samples=None,
                  device=None, edges=None, min_pixels=MEDIAN_MIN_PIXELS,
                  tangent_groups=None):
    """The `Density` of `field` for one run, per split.

    The same rows in the same order as `score_run`, pooled per split rather than
    kept per sample -- a per-sample density would be N times the size for a
    distribution nobody reads one image at a time.

    E costs one forward pass per batch and is split by row. dE costs a
    linearisation and one JVP per stored direction, and is split by row *and*
    direction (`density_splits`), so a batch contributes to every split at
    once. `tangent_groups` merges tangent samplers into one split; it has to be
    done here rather than at plot time, since the pooled median is a median over
    the pooled pixels and not a function of the separate medians.

    The exact medians make this hold every scored pixel's error in memory --
    a few hundred MB for E, a few GB for dE's K directions -- rather than the
    density's own kilobytes. That is the price of a median that is not a bin
    centre, and it is paid here, once, rather than at plot time.
    """
    edges = EDGES[field] if edges is None else edges
    mu, dmu, E, dE, _skin = load_test_data(data_path, int(run["oop_slices"]))
    n_scored = min(n_samples or mu.shape[0], mu.shape[0])
    row_index, direction_index, split_names = density_splits(
        data_path, field, tangent_groups)

    apply_fn = load_model_fn(run["checkpoint"], device)
    if field == "E":
        inputs, truth_all = [mu], E
        predict = lambda batch: apply_fn(batch[0])
    else:
        inputs, truth_all = [mu, dmu], dE
        # one linearisation, then the K directions in one vmapped pass, as
        # make_scorer does -- the tangents are what differs, not the point
        @jax.jit
        def predict(batch):
            _, apply_lin = jax.linearize(apply_fn, batch[0])
            return jax.vmap(apply_lin, in_axes=1, out_axes=1)(batch[1])

    counts = 0.0
    pixels = [[] for _ in split_names]
    arrays = [arr[:n_scored] for arr in (*inputs, truth_all)]
    for batch, rows in batches(arrays, batch_size):
        pred = jax.device_get(predict(batch[:-1]))[:len(rows)]
        truth = np.asarray(batch[-1])[:len(rows)]
        # E is a (B, X, Z, 1) field and splits on its rows alone; dE is a
        # (B, K, X, Z, 1) one and splits on the rows *and* the directions, so a
        # split takes a rectangle of the first two axes rather than a slice of
        # one. Both are boolean masks into axes of the same `truth` and `pred`.
        def pieces(s):
            if direction_index is None:
                return lambda arr: arr[row_index[rows] == s]
            row, direction = divmod(s, len(np.unique(direction_index)))
            keep_rows = row_index[rows] == row
            keep_dirs = direction_index == direction
            return lambda arr: arr[keep_rows][:, keep_dirs]

        takes = [pieces(s) for s in range(len(split_names))]
        counts = counts + np.stack(
            [magnitude_density(take(truth), take(pred), edges) for take in takes])
        for (s, take) in enumerate(takes):
            pixels[s].append(magnitude_bin(take(truth), take(pred), edges))

    per_split = [pooled_curves(chunks, len(edges) - 1, min_pixels)
                 for chunks in pixels]
    return Density(counts=counts,
                   median=np.stack([m for (m, _, _) in per_split]),
                   mean=np.stack([m for (_, m, _) in per_split]),
                   n_pixels=np.stack([c for (_, _, c) in per_split]),
                   n_scored=n_scored)


def write_density(h5_path, run, density, data_path, field="E", edges=None,
                  tangent_groups=None):
    """Append one run's density to its own store, replacing it if already there.

    A store of its own rather than another dataset in the error store: these
    runs are not the ones the paper's figures pool over, and every dataset in
    that store is keyed `field|transform|...`, which a density is not. One file
    per field, since the edges and the splits are the field's own.

        /   attrs: data_file, field, sampler_names, edges, err_edges
            sampler_index      (N,) | (K,)
            runs/<run_id>/     attrs: everything the run recorded, plus n_scored
                density        (S, n, nErr)
                median         (S, n)
                mean           (S, n)
                median_count   (S, n)
    """
    edges = EDGES[field] if edges is None else edges
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)) or ".", exist_ok=True)
    with h5.File(h5_path, "a") as f:
        if "sampler_index" not in f:
            rows, directions, sampler_names = density_splits(
                data_path, field, tangent_groups)
            f.attrs.update(data_file=os.path.abspath(data_path),
                           field=field,
                           sampler_names=list(sampler_names),
                           edges=np.asarray(edges),
                           err_edges=error_edges(edges))
            f.create_dataset("sampler_index", data=np.asarray(rows, np.int32))
            # dE splits on both axes, so the row index alone does not say what a
            # split is; the direction index is the other half of the pair
            if directions is not None:
                f.create_dataset("direction_sampler_index",
                                 data=np.asarray(directions, np.int32))
        if not np.array_equal(f.attrs["edges"], edges):
            raise ValueError(
                f"{h5_path} was written on different bin edges; its densities "
                "cannot be compared with these. Write to a new file.")

        runs = f.require_group("runs")
        if run["run_id"] in runs:
            del runs[run["run_id"]]
        group = runs.create_group(run["run_id"])
        group.attrs.update(run)
        group.attrs["n_scored"] = int(density.n_scored)
        group.create_dataset("density", data=np.asarray(density.counts))
        group.create_dataset("median", data=np.asarray(density.median))
        group.create_dataset("mean", data=np.asarray(density.mean))
        group.create_dataset("median_count", data=np.asarray(density.n_pixels))
