import jax
import jax.numpy as jnp
from functools import partial
import nifty8.re as jft
import h5py as h5

WIDTH = 4.
DEPTH = 2.72
DIST = 0.01
X_PIX = int(WIDTH/DIST)
Z_PIX = int(DEPTH/DIST)


def sample_skin_line(key, fluctuations, loglogavgslope):
    cfm = jft.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean=0.5, offset_std=(0.05, 0.001))
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
    skin_line = sample_skin_line(ks, (0.075, 0.03), (-4, 0.1))
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


def random_sample(key, _, params):
    tissue = cf_maker(key, **params)
    return tissue


def sampler(key, primal_sampler, tangent_sampler, skinline_params, optical_params, tangent_params):
    k_mask, k_ma, k_ms, k_va, k_vs = jax.random.split(key, 5)
    masks = sample_mask(k_mask, **skinline_params)
    absorption = primal_sampler(k_ma, masks, optical_params["absorption"])
    scattering = primal_sampler(k_ms, masks, optical_params["scattering"])
    anisotropy = get_anisotropy(masks)

    v_absorption = tangent_sampler(k_va, masks, tangent_params["v_absorption"])
    v_scattering = tangent_sampler(k_vs, masks, tangent_params["v_scattering"])

    masks = jnp.float32(jnp.stack([m for m in masks.values()]))
    mu = jnp.float32(jnp.stack((absorption, scattering)))
    v = jnp.float32(jnp.stack((v_absorption, v_scattering)))
    anisotropy = jnp.float32(anisotropy)
    return (masks, anisotropy, mu, v)


def compute_dataset(key, N_samples_per_sampler, samplers, filename):
    from tissue_config import skinline_params, params

    with h5.File(filename, "x") as f:
        N_total_samples = N_samples_per_sampler * len(samplers)
        f["masks"] = jnp.zeros((N_total_samples, 4, X_PIX, Z_PIX), dtype=jnp.float32)
        f["g"] = jnp.zeros((N_total_samples, 1, X_PIX, Z_PIX), dtype=jnp.float32)
        f["mu"] = jnp.zeros((N_total_samples, 2, X_PIX, Z_PIX), dtype=jnp.float32)
        f["v"] = jnp.zeros((N_total_samples, 2, X_PIX, Z_PIX), dtype=jnp.float32)

    for i in range(N_samples_per_sampler):
        print(i)
        for (j, (mu_sampler, v_sampler)) in enumerate(samplers):
            optical_params = params[mu_sampler.__name__]
            tangent_params = params[v_sampler.__name__]
            key, k_sample = jax.random.split(key)
            _mask, _g, _mu, _v = sampler(k_sample,
                                         mu_sampler, v_sampler,
                                         skinline_params, optical_params, tangent_params)

            idx = (i * len(samplers) + j, Ellipsis)
            with h5.File(filename, "r+") as f:
                f["masks"][idx] = _mask
                f["g"][idx] = _g
                f["mu"][idx] = _mu
                f["v"][idx] = _v


def compute_all_datasets(base_path):
    key = jax.random.key(4196180089)
    k_train, k_val, k_test = jax.random.split(key, 3)
    compute_dataset(k_train, 1024,
                    ((tissue_sample, tissue_sample), (tissue_sample, random_tissue_sample)),
                    f"{base_path}/train_data.h5")
    compute_dataset(k_val, 128,
                    ((tissue_sample, tissue_sample), (tissue_sample, random_tissue_sample)),
                    f"{base_path}/val_data.h5")
    compute_dataset(k_test, 128,
                    ((tissue_sample, tissue_sample),
                     (tissue_sample, random_tissue_sample),
                     (random_tissue_sample, random_tissue_sample),
                     (random_sample, random_sample)),
                    f"{base_path}/test_data.h5")
