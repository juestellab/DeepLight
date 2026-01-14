import pickle
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable
import pandas as pd
import os
import deeplight as dl
import flax.linen as nn
import h5py as h5

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_autotune_level=0 '
)

E_SCALE = 5e-4
dE_SCALE = 1e-2


def nl(x, c):
    return jnp.sign(x) * (jnp.log((jnp.exp(c) - jnp.exp(-jnp.abs(x)))/(jnp.exp(c) - 1)) + jnp.abs(x))


def load_network(path, device):
    with open(path, "rb") as f:
        state = pickle.load(f)
        if "val_loss" in state.keys():
            state.pop("val_loss")
    state = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), state)
    model = dl.UNet((16, 32, 64, 128, 256, 256),
                    (3, 3), nn.celu, nn.softplus)

    @jax.jit
    def apply_model(x):
        return model.apply(state, x, train=False)
    return lambda x: apply_model(x)


def get_optimal_runs(dir):
    dir_list = [x.name for x in os.scandir(dir)]
    optimal_sobolev = ""
    optimal_baseline = ""
    min_sobolev_val = jnp.inf
    min_base_val = jnp.inf
    for d in dir_list:
        if d[-4:] == ".zip":
            continue
        sobolev = float(d.split("jac=")[1].split("_")[0]) != 0
        with open(f"{dir}/{d}/min_val.pkl", "rb") as f:
            val_loss = pickle.load(f)["val_loss"]
            if sobolev and val_loss < min_sobolev_val:
                min_sobolev_val = val_loss
                optimal_sobolev = f"{dir}/{d}"
            elif not sobolev and val_loss < min_base_val:
                min_base_val = val_loss
                optimal_baseline = f"{dir}/{d}"
    return optimal_sobolev, optimal_baseline


def compute_loss(test, models):
    loss_arr = []
    loss_deriv_arr = []
    for (i, fn) in enumerate(models):
        print("Computing Loss")
        k = "Baseline" if i == 1 else "Sobolev"

        loss = [compute_diff(t, fn, lambda x: x, relative=True) for t in test]
        loss_arr += [
            {"type": k,
             "dist": "In Distribution",
             "Rel. RMSE": np.concat((np.array(loss[0][0]), np.array(loss[1][0])))},
            {"type": k,
             "dist": "Out-of-Distribution",
             "Rel. RMSE": np.array(loss[2][0])},
        ]

        loss_deriv_arr += [
            {"type": k,
             "dist": "In Distribution_1",
             "Rel. RMSE": np.array(loss[0][1])},
            {"type": k,
             "dist": "In Distribution_2",
             "Rel. RMSE": np.array(loss[1][1])},
            {"type": k,
             "dist": "Out-of-Distribution",
             "Rel. RMSE": np.array(loss[2][1])},
        ]
    loss_df = pd.DataFrame(loss_arr).explode("Rel. RMSE").reset_index(drop=True)
    deriv_loss_df = pd.DataFrame(loss_deriv_arr).explode("Rel. RMSE").reset_index(drop=True)
    return loss_df, deriv_loss_df


def compute_gain(test, models):
    gain_arr = []
    gain_deriv_arr = []
    loss_base = [jnp.zeros(d.data.data[0].shape[0]) for d in test]
    loss_sobolev = [jnp.zeros(d.data.data[0].shape[0]) for d in test]
    deriv_loss_base = [jnp.zeros(d.data.data[0].shape[0]) for d in test]
    deriv_loss_sobolev = [jnp.zeros(d.data.data[0].shape[0]) for d in test]

    for (i, fn) in enumerate(models):
        print("Compute Gain")
        print("Computing Loss")
        sobolev = (i == 0)
        loss = [compute_diff(t, fn, lambda x: x) for t in test]

        op_loss = [_loss[0] for _loss in loss]
        deriv_loss = [_loss[1] for _loss in loss]
        if sobolev:
            loss_sobolev = [l_sum + _loss for (l_sum, _loss) in zip(loss_sobolev, op_loss)]
            deriv_loss_sobolev = [l_sum + _loss
                                  for (l_sum, _loss) in zip(deriv_loss_sobolev, deriv_loss)]
        else:
            loss_base = [l_sum + _loss for (l_sum, _loss) in zip(loss_base, op_loss)]
            deriv_loss_base = [l_sum + _loss for (l_sum, _loss) in zip(deriv_loss_base, deriv_loss)]

    rel_op_gain = [(l_base - l_sob)/l_base for (l_base, l_sob) in zip(loss_base, loss_sobolev)]
    rel_deriv_gain = [(l_base - l_sob)/l_base
                      for (l_base, l_sob) in zip(deriv_loss_base, deriv_loss_sobolev)]
    gain_arr += [
        {"dist": "In Distribution",
         "Gain": np.concat((np.array(rel_op_gain[0]), np.array(rel_op_gain[1])))},
        {"dist": "Out-of-Distribution",
         "Gain": np.array(rel_op_gain[2])},
    ]

    gain_deriv_arr += [
        {"dist": "In Distribution_1",
         "Gain": np.array(rel_deriv_gain[0])},
        {"dist": "In Distribution_2",
         "Gain": np.array(rel_deriv_gain[1])},
        {"dist": "Out-of-Distribution",
         "Gain": np.array(rel_deriv_gain[2])},
    ]
    gain_df = pd.DataFrame(gain_arr).explode("Gain").reset_index(drop=True)
    deriv_gain_df = pd.DataFrame(gain_deriv_arr).explode("Gain").reset_index(drop=True)
    return gain_df, deriv_gain_df


def sample_error(nl_scale):
    def _sample_error_instance(E_pred, E_gt):
        squared_diff = (nl(E_pred, nl_scale) - nl(E_gt, nl_scale))**2
        return jnp.sqrt(jnp.sum(squared_diff))
    return _sample_error_instance


def compute_diff(ds, model, fn: Callable, relative=False):
    res, dres = jnp.array([]), jnp.array([])
    for batch in ds:
        mu, dmu, E, dE = batch
        E_pred, dE_pred = jax.jvp(lambda x: model(x), (mu,), (dmu,))

        diff_E = jax.vmap(sample_error(E_SCALE))(E_pred, E)
        diff_dE = jax.vmap(sample_error(dE_SCALE))(dE_pred, dE)

        if relative:
            diff_E = diff_E/jnp.sqrt(jnp.sum(nl(E, E_SCALE)**2, axis=(1, 2, 3)))
            diff_dE = diff_dE/jnp.sqrt(jnp.sum(jnp.abs(nl(dE, dE_SCALE)**2), axis=(1, 2, 3)))
        res = jnp.append(res, diff_E)
        dres = jnp.append(dres, diff_dE)
    return np.array(res), np.array(dres)


def compute_diff_depth_resolved(ds, models, averaged=True):
    shape = ds.data.data[0].shape[1:3]
    res = [jnp.zeros(shape[1]) for _ in models]
    dres = [jnp.zeros(shape[1]) for _ in models]
    print(res[0].shape)

    total_E, total_dE, rel_dE, count = jnp.zeros(shape[1]), jnp.zeros(shape[1]), jnp.zeros(shape[1]), jnp.zeros(shape[1])
    for (b_idx, batch) in enumerate(ds):
        print(b_idx)
        mu, dmu, E, dE = batch
        E_nl = nl(E, E_SCALE)
        dE_nl = nl(dE, dE_SCALE)
        preds = [jax.jvp(lambda x: model(x), (mu,), (dmu,)) for model in models]

        diff_E = [jnp.abs(E_nl - nl(pred[0], E_SCALE)) for pred in preds]
        diff_dE = [jnp.abs(dE_nl - nl(pred[1], dE_SCALE)) for pred in preds]
        # diff_E = [jnp.sum(jnp.abs(nl(E, E_SCALE) - nl(model(mu), E_SCALE)), axis=(0, 1, 3)) for model in models]
        for s in range(mu.shape[0]):
            idx = jax.vmap(lambda a: jnp.argmax(a > 10))(mu[s, :, :, 1])
            for i in range(shape[0]):
                res = [r.at[0:(shape[1]-idx[i])].add(d.at[s, i, idx[i]:shape[1], 0].get()) for (r, d) in zip(res, diff_E)]
                dres = [r.at[0:(shape[1]-idx[i])].add(d.at[s, i, idx[i]:shape[1], 0].get()) for (r, d) in zip(dres, diff_dE)]
                total_E = total_E.at[0:(shape[1] - idx[i])].add(E_nl.at[s, i, idx[i]:shape[1], 0].get())
                total_dE = total_dE.at[0:(shape[1] - idx[i])].add(jnp.abs(dE_nl.at[s, i, idx[i]:shape[1], 0].get()))
                # rel_dE = rel_dE.at[0:(shape[1] - idx[i])].add(jnp.abs(dE_nl.at[s, i, idx[i]:shape[1], 0].get())/E_nl.at[s, i, idx[i]:shape[1], 0].get())
                count = count.at[0:(shape[1] - idx[i])].add(1)

    return res, dres, total_E, total_dE, rel_dE, count


def depth_wise_error(dss, models, path, key):
    for (i, ds) in enumerate(dss):
        err, derr, E, dE, rel_dE, c = compute_diff_depth_resolved(ds, models)
        if i == 0:
            error_depth = err
            diff_error_depth = derr
            total_E = E
            total_dE = dE
            count_depth = c
        else:
            error_depth = [e + e_new for (e, e_new) in zip(error_depth, err)]
            diff_error_depth = [e + e_new for (e, e_new) in zip(diff_error_depth, derr)]
        total_E += E
        total_dE += dE
        count_depth += c
    # return error_depth, diff_error_depth, total_E, total_dE, rel_dE, count_depth
    store_depth_error(path, key, error_depth, diff_error_depth, total_E, count_depth)


def store_depth_error(path, key, e, de, E, c):
    with h5.File(path, "a") as f:
        g = f.create_group(key)
        g["Error"] = e
        g["Diff_Error"] = de
        g["Energy"] = E
        g["Count"] = c


if __name__ == "__main__":
    paths = dl.load_paths('./paths.toml')
    res_dir = paths['network_weights_dir']
    n_pixels = (400, 272)

    results_path = os.path.join(paths['results_dir'], "quantitative_results_with_deriv.h5")
    ################################################################################################
    # Initialize dataloader
    data_path = os.path.join(paths['data_dir'], "test_data.h5")
    test_data = dl.load_data(data_path)
    test = []
    for i in range(4):
        data = [arr[i::4] for arr in test_data]
        test_ds = dl.DeepLightDataset(*data,
                                      grid_size=[n/1e2 for n in n_pixels],
                                      n_pixels=n_pixels)
        test += [dl.DataLoader(test_ds, 4)]

    ################################################################################################
    # Peak Performance
    network_paths = get_optimal_runs(res_dir)
    print(network_paths)
    models = [load_network(f"{path}/min_val.pkl", jax.devices()[0]) for path in network_paths]

    err, E, dE, c = depth_wise_error(test[0:2], models, results_path, "ID")
    depth_wise_error(test[0:2], models, results_path, "ID")
    depth_wise_error(test[2:3], models, results_path, "OOD")

    loss_df, deriv_loss_df = compute_loss(test, models)
    gain_df, deriv_gain_df = compute_gain(test, models)

    try:
        loss_df.to_hdf(results_path, key="loss")
        deriv_loss_df.to_hdf(results_path, key="deriv_loss")

        gain_df.to_hdf(results_path, key="gain")
        deriv_gain_df.to_hdf(results_path, key="deriv_gain")
    except Exception as e:
        print(e)


    # TODO linear and log error

