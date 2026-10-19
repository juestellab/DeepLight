import os
import time
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import nifty.re as jft
import h5py as h5

WIDTH = 4.
DEPTH = 2.72
DIST = 0.01
X_PIX = int(WIDTH/DIST)
Z_PIX = int(DEPTH/DIST)

MEAN_SKIN_DEPTH = 0.5


def sample_skin_line(key, fluctuations, loglogavgslope):
    cfm = jft.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean=MEAN_SKIN_DEPTH,
                                   offset_std=(0.05, 0.001))
    cfm.add_fluctuations((2 * X_PIX,), (DIST,),
                         loglogavgslope=loglogavgslope,
                         fluctuations=fluctuations)
    cf = cfm.finalize()
    return cf(cf.init(key))[:X_PIX]


def smooth_step_function(x, thickness, h1, h2, start):
    return 0.5 * (1 + jnp.minimum(jnp.tanh(h1*(x - start)),
                                  jnp.tanh(-h2*(x - (start + thickness)))))


def get_masks(depth, skin_line, epidermis_thickness, dermis_thickness, h1, h2):
    def coupling_fn(thickness): return smooth_step_function(depth, thickness, h1, h1, -1)
    coupling_mask = jax.vmap(coupling_fn)(skin_line + 1)

    epidermis_fn = partial(smooth_step_function, depth, epidermis_thickness, h1, h1)
    epidermis_mask = jax.vmap(epidermis_fn)(skin_line)

    dermis_fn = partial(smooth_step_function, depth, dermis_thickness, h1, h2)
    dermis_mask = jax.vmap(dermis_fn)(skin_line + epidermis_thickness)

    tissue_fn = partial(smooth_step_function, depth, jnp.inf, h2, h2)
    tissue_mask = jax.vmap(tissue_fn)(skin_line + epidermis_thickness + dermis_thickness)
    return coupling_mask, epidermis_mask, dermis_mask, tissue_mask


def sample_mask(key, epidermis_params, dermis_params):
    ks, ke, kd = jax.random.split(key, 3)
    depth = jnp.linspace(0, DEPTH, Z_PIX)
    skin_line = sample_skin_line(ks, (0.05, 0.03), (-4, 0.1))
    while jnp.any(skin_line < 0.1):
        _, ks = jax.random.split(ks)
        skin_line = sample_skin_line(ks, (0.05, 0.03), (-4, 0.1))

    _et = epidermis_params["mean"] + epidermis_params["std"] * jax.random.normal(ke)
    _dt = dermis_params["mean"] + dermis_params["std"] * jax.random.normal(kd)
    epidermis_thickness = jnp.maximum(_et, 0)
    derims_thickness = jnp.maximum(_dt, 0)
    m = get_masks(depth, skin_line, epidermis_thickness, derims_thickness, 130, 50)
    masks = {"coupling": m[0], "epidermis": m[1], "dermis": m[2], "tissue": m[3]}
    return masks


def get_anisotropy(masks):
    return (0.5 * masks["coupling"] +
            0.8 * (masks["epidermis"] + masks["dermis"]) +
            0.9 * masks["tissue"])/sum(masks.values())


def cf_maker(key, offset, loglogavgslope, fluctuations, nl, **kwargs):
    cfm = jft.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean=offset["mean"], offset_std=offset["std"])
    cfm.add_fluctuations((2 * X_PIX, 2 * Z_PIX),
                         (DIST, DIST),
                         loglogavgslope=loglogavgslope,
                         fluctuations=fluctuations)
    cf = cfm.finalize()
    return nl(cf(cf.init(key)))[:X_PIX, :Z_PIX]


# Tissue
def tissue_sample(key, masks, params):
    tissue = jnp.zeros((X_PIX, Z_PIX))
    for k in masks.keys():
        key, kt = jax.random.split(key)
        tissue = tissue + masks[k]*cf_maker(kt, **params[k])

    if params["vessels"] is None:
        return tissue

    def acceptable_interval(v):
        A = jnp.sum((v * masks["tissue"]) > 1)/jnp.sum(masks["tissue"] > 0.3)
        return (A < 0.2) and (A > 0.03)

    key, kt = jax.random.split(key)
    vessels = cf_maker(kt, **params["vessels"])
    i = 0
    while not acceptable_interval(vessels) and i < 1000:
        i += 1
        key, kt = jax.random.split(key)
        vessels = cf_maker(kt, **params["vessels"])

    alpha = 0
    while alpha < 0.1:
        key, ka = jax.random.split(key)
        alpha = params["vessels"]["alpha"][0] + params["vessels"]["alpha"][1] * jax.random.normal(ka)

    tissue = tissue + masks["tissue"] * vessels * alpha
    return tissue


def random_tissue_sample(key, masks, params):
    tissue = jnp.zeros((X_PIX, Z_PIX))
    kc, kt = jax.random.split(key)
    tissue = (tissue + masks["coupling"]*cf_maker(kc, **params["coupling"])
              + (1 - masks["coupling"])*cf_maker(kt, **params["correlated"]))
    return tissue


# Tangent samplers
#
# A tangent sampler has the signature (key, masks, mu, params) -> (2, X_PIX, Z_PIX),
# where the leading axis is (absorption, scattering). `mu` is passed so that a
# sampler can make the perturbation depend on the primal fields.
def independent_tangent(field_sampler):
    """Draw the two channels of a tangent independently from `field_sampler`.

    This is the historical behaviour: the tangent of each channel is just
    another draw of a primal-style sampler with its own (zero-offset) params.
    """
    def tangent(key, masks, mu, params):
        k_a, k_s = jax.random.split(key)
        return jnp.stack((field_sampler(k_a, masks, params["v_absorption"]),
                          field_sampler(k_s, masks, params["v_scattering"])))
    tangent.__name__ = field_sampler.__name__
    return tangent


def damp(mu):
    """Compress a primal field's dynamic range, keeping its spatial structure.

    `mu / (mu + median(mu))`, per channel. Saturating: it tends to 1 wherever
    `mu >> median(mu)`, so the result follows where the structure *is* without
    inheriting how large it gets.

    The motivation is that a tangent proportional to `mu` is not what
    reconstruction residuals look like. Backprojected residuals share mu's
    structure, but training drives the large coefficients down first, leaving
    high and low values with comparable *absolute* error. Raw `mu` has the
    opposite profile: measured on 5 phantoms, 47% of its squared magnitude sits
    in the hottest 1% of voxels (the ~1-2 pixel epidermis), and half of it in
    1.2%. Damping with the median takes that to 3.0%.

    `mu_0` is the knob, and it sets how hard the compression bites: the
    saturation is governed by how far `mu_0` sits below the bulk of the field.
    `min(mu)` flattens hardest (top 1% down to 1.3%, peak/bulk 1.08 -- nearly a
    uniform field times the sign patches, so mu's contrast is essentially gone).
    The median keeps more of it. Both are taken within the sample, per channel.

    Note the downstream consequence: `normalize_tangent` rescales each channel
    to `norm(mu)`, so a flatter tangent spreads the *same* total norm over the
    whole field instead of concentrating it in the epidermis. That perturbs the
    bulk tissue light actually traverses, and the dE response grows sharply --
    12.3x going from raw `mu` to the `min` variant. Flattening the tangent and
    keeping the dE response balanced against the other samplers are separate
    knobs; see the scale discussion in tissue_config.
    """
    mu0 = jnp.median(mu, axis=(-2, -1), keepdims=True)
    return mu / (mu + mu0)


def scaled_sign_tangent(key, masks, mu, params):
    """Tangent following the primal structure, modulated by a smooth sign field.

    `s(x)` is a correlated field pushed through a saturating non-linearity, so it
    sits near ±1 over a few patches spanning the FOV with smooth transitions in
    between. Both channels share that field; a single global sign `p = ±1`
    decides whether the scattering perturbation runs with or against the
    absorption one. This reproduces the structure-following, sign-flipping
    derivatives that show up in reconstructions.

    The primal enters through `damp`, not directly -- see there for why.
    """
    k_field, k_sign = jax.random.split(key)
    s = cf_maker(k_field, **params["sign_field"])
    p = 2. * jax.random.bernoulli(k_sign) - 1.
    d = damp(mu)
    return jnp.stack((s * d[0], p * s * d[1]))


def normalize_tangent(v, mu):
    """Scale each channel of a tangent to a unit relative perturbation.

    Per channel `c`, the tangent is rescaled so that ||v_c|| == ||mu_c||. Without
    this the samplers produce wildly different tangent magnitudes (absorption
    lives around 0.1, scattering around 110), which makes the resulting dE
    incomparable across samplers.

    NOTE this normalisation is a placeholder: revisit the choice of norm — and
    whether to normalise jointly rather than per channel — once the dE for the
    new samplers have actually been computed and their magnitudes are known.
    """
    scale = (jnp.linalg.norm(mu.reshape(2, -1), axis=1)
             / jnp.linalg.norm(v.reshape(2, -1), axis=1))
    return v * scale[:, None, None]


# Derivative directions per tangent sampler, per tissue sample. All of a
# sample's directions ride on the same ray ensemble in the transport stage, so
# the marginal cost of a direction is far below that of another sample.
DIRECTIONS_PER_TANGENT = 2


# Rejection bound on the scattering primal, in 1/cm. A draw with any voxel below
# this is redrawn; see `sampler`.
MIN_SCATTERING = 1e-3
MAX_DRAW_ATTEMPTS = 100


def sampler(key, row_spec, skinline_params, params,
            directions_per_tangent=DIRECTIONS_PER_TANGENT):
    """One tissue sample, redrawn until its scattering is physical.

    A draw is rejected if any voxel of mu_s falls below `MIN_SCATTERING`. Only
    the OOD primal can produce one: `scattering.tissue` floors the in-
    distribution field at 80/cm, while `correlated_scattering` is unfloored, so
    its correlated field occasionally wanders far enough negative that softplus
    underflows and leaves bulk *tissue* -- not the coupling layer -- with
    essentially no scattering. Measured on the 512-row test split, the seven
    rows that hit exact zero were also the six worst network errors plus rank
    12, an order of magnitude above the median. They are not samples the
    operator should be scored on.

    Rejection conditions the prior on "no near-zero scattering", the way
    `tissue_sample` already conditions the vessel field on `acceptable_interval`.

    Retries stay inside this row's key stream: attempt 0 uses `key` itself and
    later attempts use `fold_in(key, attempt)`, so a row that passes first try
    is bit-identical to what the unconditioned sampler drew, and a row that
    retries cannot shift any other row -- `compute_dataset` seeds each row from
    `fold_in(key, row)`, not from a running split.
    """
    for attempt in range(MAX_DRAW_ATTEMPTS):
        draw = _draw_sample(key if attempt == 0 else jax.random.fold_in(key, attempt),
                            row_spec, skinline_params, params,
                            directions_per_tangent)
        if float(draw[2][1].min()) >= MIN_SCATTERING:
            return draw
    raise RuntimeError(
        f"no draw with mu_s >= {MIN_SCATTERING} after {MAX_DRAW_ATTEMPTS} "
        f"attempts for row spec {row_spec.name}; the scattering prior is "
        "almost certainly mis-specified")


def _draw_sample(key, row_spec, skinline_params, params,
                 directions_per_tangent=DIRECTIONS_PER_TANGENT):
    """One unconditioned draw: primal fields plus every tangent sampler's directions.

    The directions are grouped by tangent sampler and ordered
    `(tangent_0 x d, tangent_1 x d, ...)`, which is the order
    `direction_sampler_index` records.
    """
    k_mask, k_ma, k_ms, k_v = jax.random.split(key, 4)
    masks = sample_mask(k_mask, **skinline_params)
    optical_params = params[row_spec.mu_sampler.__name__]
    absorption = row_spec.mu_sampler(k_ma, masks, optical_params["absorption"])
    scattering = row_spec.mu_sampler(k_ms, masks, optical_params["scattering"])
    anisotropy = get_anisotropy(masks)

    mu = jnp.float32(jnp.stack((absorption, scattering)))
    v = []
    for (j, tangent) in enumerate(row_spec.v_samplers):
        tangent_params = params[tangent.__name__]
        # fold_in on (sampler, direction) rather than a running split, so adding a
        # tangent sampler leaves the other samplers' directions untouched
        for d in range(directions_per_tangent):
            k = jax.random.fold_in(jax.random.fold_in(k_v, j), d)
            v.append(normalize_tangent(tangent(k, masks, mu, tangent_params), mu))

    masks = jnp.float32(jnp.stack([m for m in masks.values()]))
    v = jnp.float32(jnp.stack(v))
    anisotropy = jnp.float32(anisotropy)
    return (masks, anisotropy, mu, v)


####################################################################################################
# Dataset definition
####################################################################################################
# Each row of a dataset is one tissue sample carrying every tangent sampler's
# directions; the sampler axis lives *within* a sample, not across rows. A file
# with more than one RowSpec is a round-robin over them (row `i` uses spec
# `i % len(specs)`), which is how the test file mixes in-distribution and
# out-of-distribution primal fields. All specs in a file must list the same
# number of tangent samplers, so `v` stays rectangular.
#
# Both layouts are recorded in the file: `sampler_index` (per row, which primal)
# and `direction_sampler_index` (per direction, which tangent sampler).
RowSpec = namedtuple("RowSpec", "name mu_sampler v_samplers")

V_TISSUE = independent_tangent(tissue_sample)
V_RANDOM_TISSUE = independent_tangent(random_tissue_sample)

# The tangent samplers every sample carries, in storage order.
TANGENT_SAMPLERS = (
    ("ID_1", V_TISSUE),
    ("ID_2", V_RANDOM_TISSUE),
    ("SIGN", scaled_sign_tangent),
)
TANGENT_NAMES = tuple(name for name, _ in TANGENT_SAMPLERS)
_TANGENTS = tuple(sampler for _, sampler in TANGENT_SAMPLERS)

ID_ROW = RowSpec("ID", tissue_sample, _TANGENTS)
OOD_ROW = RowSpec("OOD", random_tissue_sample, _TANGENTS)

# rows per RowSpec; a file holds that many times len(specs)
DATASET_SPECS = {
    "train": (2048, 2, (ID_ROW,)),
    "val": (256, 2, (ID_ROW,)),
    "test": (256, 1, (ID_ROW, OOD_ROW)),
}

DEFAULT_SEED = 4196180089


def split_key(seed, split):
    """Independent root key per split, derived from one seed.

    Keyed by name rather than by position, so adding a split cannot shift the
    randomness of the existing ones.
    """
    return jax.random.fold_in(jax.random.key(seed),
                              int.from_bytes(split.encode(), "little") % (2**31))


def _create_dataset_file(filename, n_rows, specs, directions_per_tangent, attrs,
                         row_offset=0):
    n_tangents = len(specs[0].v_samplers)
    n_directions = n_tangents * directions_per_tangent
    with h5.File(filename, "x") as f:
        f.create_dataset("masks", (n_rows, 4, X_PIX, Z_PIX), dtype="f4")
        f.create_dataset("g", (n_rows, 1, X_PIX, Z_PIX), dtype="f4")
        f.create_dataset("mu", (n_rows, 2, X_PIX, Z_PIX), dtype="f4")
        f.create_dataset("v", (n_rows, n_directions, 2, X_PIX, Z_PIX), dtype="f4")
        # which primal produced each row, and the resume marker. Keyed off the
        # global row index so a shard records the specs it actually holds.
        f.create_dataset("sampler_index",
                         data=(np.arange(n_rows, dtype="i4") + row_offset)
                         % len(specs))
        f.create_dataset("done", (n_rows,), dtype="u1")
        # which tangent sampler produced each direction (same for every row)
        f.create_dataset("direction_sampler_index",
                         data=np.repeat(np.arange(n_tangents, dtype="i4"),
                                        directions_per_tangent))
        f.attrs["sampler_names"] = [s.name for s in specs]
        f.attrs["mu_samplers"] = [s.mu_sampler.__name__ for s in specs]
        f.attrs["direction_sampler_names"] = list(TANGENT_NAMES[:n_tangents])
        f.attrs["directions_per_tangent"] = directions_per_tangent
        f.attrs["n_directions"] = n_directions
        f.attrs["width_cm"] = WIDTH
        f.attrs["depth_cm"] = DEPTH
        f.attrs["pixel_size_cm"] = DIST
        f.attrs["mean_skin_depth_cm"] = MEAN_SKIN_DEPTH
        for key, value in attrs.items():
            f.attrs[key] = value


def compute_dataset(key, n_rows_per_spec, specs, filename,
                    directions_per_tangent=DIRECTIONS_PER_TANGENT,
                    resume=True, attrs=None, row_offset=0, n_rows=None):
    """Write tissue samples to `filename`.

    Every row carries `len(specs[0].v_samplers) * directions_per_tangent`
    derivative directions, all at the same primal point.

    Rows are written one at a time and flagged in `done`, so an interrupted run
    resumes where it stopped (`resume=True`, the default). Each row's randomness
    comes from `fold_in(key, row)` rather than from a running split, which makes
    a row's content independent of how many rows preceded it in this process --
    a resumed file is bit-identical to one written in a single pass.

    `row_offset` and `n_rows` carve out a shard: the file gets `n_rows` rows
    holding global rows `row_offset .. row_offset + n_rows - 1`. Both the key and
    the row spec are chosen from the *global* index, so concatenating the shards
    in order reproduces the unsharded dataset -- sharding is a scheduling
    decision, not a different dataset. It also keeps the round-robin over row
    specs intact across a shard boundary, which is what stops a shard of the test
    split from being all-ID or all-OOD.

    "Reproduces" holds to float32 rounding, not bit-for-bit: `masks` and `g` come
    out identical, while `mu` and `v` differ by ~1e-6 relative because XLA does
    not fix reduction order across differently-shaped batches. That is the same
    nondeterminism two identical un-sharded runs show, and is four orders of
    magnitude below the Monte Carlo noise on the labels.

    `n_rows` defaults to the whole split, `n_rows_per_spec * len(specs)`.
    """
    from tissue_config import skinline_params, params

    n_tangents = {len(s.v_samplers) for s in specs}
    if len(n_tangents) != 1:
        raise ValueError("all row specs of a file must list the same number of "
                         f"tangent samplers, got {sorted(n_tangents)}")
    n_total = n_rows_per_spec * len(specs)
    n_rows = n_total if n_rows is None else n_rows
    if not 0 <= row_offset and row_offset + n_rows <= n_total:
        raise ValueError(f"rows {row_offset}..{row_offset + n_rows} fall outside "
                         f"the split's {n_total} rows")
    n_directions = n_tangents.pop() * directions_per_tangent
    if not os.path.exists(filename):
        _create_dataset_file(filename, n_rows, specs, directions_per_tangent,
                             {"row_offset": row_offset, "n_rows_total": n_total,
                              **(attrs or {})}, row_offset=row_offset)
    elif not resume:
        raise FileExistsError(f"{filename} exists; pass resume=True to continue it")

    with h5.File(filename, "r") as f:
        done = np.array(f["done"], dtype=bool)
        if f["v"].shape[0] != n_rows or f["v"].shape[1] != n_directions:
            raise ValueError(f"{filename} was created with a different shape "
                             f"({f['v'].shape}); delete it or fix the arguments")
    todo = np.flatnonzero(~done)
    print(f"{filename}: {len(todo)} of {n_rows} rows to write, "
          f"{n_directions} directions each")

    t_start = time.time()
    for n, row in enumerate(todo):
        # the global row index is what selects the spec and seeds the draw, so a
        # shard's contents do not depend on the shard's own position
        g_row = int(row) + row_offset
        spec = specs[g_row % len(specs)]
        _mask, _g, _mu, _v = sampler(jax.random.fold_in(key, g_row), spec,
                                     skinline_params, params,
                                     directions_per_tangent)
        with h5.File(filename, "r+") as f:
            f["masks"][row] = _mask
            f["g"][row] = _g
            f["mu"][row] = _mu
            f["v"][row] = _v
            f["done"][row] = 1
        elapsed = time.time() - t_start
        eta = elapsed / (n + 1) * (len(todo) - n - 1)
        print(f"  row {row + 1}/{n_rows} ({spec.name})"
              f"\t{elapsed / (n + 1):.1f} s/sample\tETA {eta / 60:.1f} min",
              flush=True)


def compute_all_datasets(base_path, seed=DEFAULT_SEED, splits=tuple(DATASET_SPECS),
                         resume=True, attrs=None):
    for split in splits:
        n_rows_per_spec, directions_per_tangent, specs = DATASET_SPECS[split]
        compute_dataset(split_key(seed, split), n_rows_per_spec, specs,
                        f"{base_path}/{split}_data.h5",
                        directions_per_tangent=directions_per_tangent,
                        resume=resume,
                        attrs={"split": split, "seed": seed, **(attrs or {})})
