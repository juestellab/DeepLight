import pickle
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import deeplight as dl
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.colors import CenteredNorm

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_autotune_level=0 '
)

mpl.rcParams.update({'font.size': 8,
                     "text.usetex": True,
                     "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}"
                     })

E_SCALE = 5e-4
dE_SCALE = 1e-2


def nl(x, c):
    return jnp.sign(x) * (jnp.log((jnp.exp(c) - jnp.exp(-jnp.abs(x)))/(jnp.exp(c) - 1)) + jnp.abs(x))


def get_optimal_runs(dir):
    import os
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


def get_quantile_samples(ds, apply_sobolev, apply_base, idxs):
    Es, Eps, Epbs = [], [], []
    for idx in idxs:
        print(idx)
        mu, dmu, E, dE = ds.data[idx]

        Ep = apply_sobolev(mu[jnp.newaxis, :, :, :])
        Epb = apply_base(mu[jnp.newaxis, :, :, :])

        Es += [nl(E[:, :, 0], E_SCALE),]
        Eps += [nl(Ep[0, :, :, 0], E_SCALE),]
        Epbs += [nl(Epb[0, :, :, 0], E_SCALE),]
    return Es, Eps, Epbs


def get_quantile_deriv_samples(ds, apply_sobolev, apply_base, idxs):
    dEs, dEps, dEpbs = [], [], []
    for idx in idxs:
        mu, dmu, E, dE = ds.data[idx]

        _, dEp = jax.jvp(apply_sobolev,
                         (mu[jnp.newaxis, :, :, :],),
                         (dmu[jnp.newaxis, :, :, :],))
        _, dEpb = jax.jvp(apply_base,
                          (mu[jnp.newaxis, :, :, :],),
                          (dmu[jnp.newaxis, :, :, :],))

        dEs += [nl(dE, dE_SCALE)[:, :, 0],]
        dEps += [nl(dEp[0, :, :, 0], dE_SCALE),]
        dEpbs += [nl(dEpb[0, :, :, 0], dE_SCALE),]
    return dEs, dEps, dEpbs


def qualitative_samples_sep(gain_df, dist, ds, apply_sobolev, apply_base, quantiles, deriv=False):
    labels = iter("abcdefghijklmnopqrstuvwxyz")

    def loss_str(a, b):
        return np.format_float_scientific(jnp.sqrt(jnp.mean((a - b)**2)), precision=1, exp_digits=1)

    def column_label(ax, text, in_layout=True):
        ax.text(-0.02, 0.5, text,
                horizontalalignment='right',
                verticalalignment='center',
                rotation='vertical',
                in_layout=in_layout,
                transform=ax.transAxes)

    f, axs = plt.subplots(3, 3, figsize=(8, 5), layout="constrained")
    idxs = []
    for quantile in quantiles:
        gain = gain_df[gain_df["dist"] == dist]["Gain"].to_numpy()
        gain = np.array([*gain])
        idxs += [np.argsort(gain)[int(gain.size * quantile) - 1]]

    if deriv:
        gt, sob, base = get_quantile_deriv_samples(ds, apply_sobolev, apply_base, idxs)
    else:
        gt, sob, base = get_quantile_samples(ds, apply_sobolev, apply_base, idxs)

    for (i, q) in enumerate(quantiles):
        if deriv:
            hm = axs[i, 0].matshow(gt[i].T, cmap="coolwarm", norm=CenteredNorm())
        else:
            hm = axs[i, 0].matshow(gt[i].T, cmap="inferno", norm="linear", vmin=0)
        dnorm = CenteredNorm(0, jnp.maximum(jnp.percentile(jnp.abs(gt[i] - sob[i]), 99.9),
                                            jnp.percentile(jnp.abs(gt[i] - base[i]), 99.9)))
        axs[i, 1].matshow((gt[i] - base[i]).T, cmap="RdBu_r", norm=dnorm)
        axs[i, 2].matshow((gt[i] - sob[i]).T, cmap="RdBu_r", norm=dnorm)
        for j in range(3):
            axs[i, j].set_axis_off()
            axs[i, j].text(0.05, 0.9,
                           next(labels),
                           transform=axs[i, j].transAxes,
                           size=12,
                           color="white" if (j == 0 and not deriv) else "black")
        axs[i, 1].text(10, 260, f"RMSE={loss_str(gt[i], base[i])}", color="black")
        axs[i, 2].text(10, 260, f"RMSE={loss_str(gt[i], sob[i])}", color="black")
        f.colorbar(hm, shrink=0.99)
        f.colorbar(axs[i, 2].images[0], shrink=0.99)

    axs[0, 0].set_title(r"$\sigma(\nabla_v E)$" if deriv else r"$\sigma(E)$")
    axs[0, 1].set_title(r"$\sigma(\nabla_v E) - \sigma(\nabla_v \Phi_B(\mu))$" if deriv else r"$\sigma(E) - \sigma(\Phi_B(\mu))$")
    axs[0, 2].set_title(r"$\sigma(\nabla_v E) - \sigma(\nabla_v \Phi_D(\mu))$" if deriv else r"$\sigma(E) - \sigma(\Phi_D(\mu))$")

    column_label(axs[0, 0], r"$25^{\mathrm{th}}$ Percentile Gain")
    column_label(axs[1, 0], r"$50^{\mathrm{th}}$ Percentile Gain")
    column_label(axs[2, 0], r"$75^{\mathrm{th}}$ Percentile Gain")
    return f


def qualitative_samples_combined(gain_df, deriv_gain_df, ds_id, ds_ood, apply_sobolev, apply_base):
    labels = iter("abcdefghijklmnopqrstuvwxyz")

    def loss_str(a, b):
        return np.format_float_scientific(jnp.sqrt(jnp.mean((a - b)**2)), precision=1, exp_digits=1)

    def get_idx(df, dist):
        gain = df[df["dist"] == dist]["Gain"].to_numpy()
        gain = np.array([*gain])
        idx = np.argsort(gain)[int(gain.size * 0.5) - 1]
        return idx

    def plot_row(ax, gt, sob, base, deriv):
        if deriv:
            hm = ax[0].matshow(gt.T, cmap="coolwarm", norm=CenteredNorm())
        else:
            hm = ax[0].matshow(gt.T, cmap="inferno", norm="linear", vmin=0)
        dnorm = CenteredNorm(0, jnp.maximum(jnp.percentile(jnp.abs(gt - sob), 99.9),
                                            jnp.percentile(jnp.abs(gt - base), 99.9)))
        ax[1].matshow((gt - base).T, cmap="RdBu_r", norm=dnorm)
        ax[2].matshow((gt - sob).T, cmap="RdBu_r", norm=dnorm)
        for j in range(3):
            ax[j].set_axis_off()
            ax[j].text(0.05, 0.8,
                       next(labels),
                       transform=ax[j].transAxes,
                       size=12,
                       color="white" if (j == 0 and not deriv) else "black")
        ax[1].text(10, 260, f"RMSE={loss_str(gt, base)}", color="black")
        ax[2].text(10, 260, f"RMSE={loss_str(gt, sob)}", color="black")
        f.colorbar(hm, shrink=0.8)
        f.colorbar(ax[2].images[0], shrink=0.82)

    f, axs = plt.subplots(5, 3, figsize=(4.2, 4.8), layout="constrained")

    params = (("In Distribution", False),
              ("Out-of-Distribution", False),
              ("In Distribution_1", True),
              ("In Distribution_2", True),
              ("Out-of-Distribution", True))
    for (i, param) in enumerate(params):
        dist, deriv = param
        ds = ds_ood if dist == "Out-of-Distribution" else ds_id
        df = deriv_gain_df if deriv else gain_df
        idx = get_idx(df, dist)
        if deriv:
            gt, sob, base = get_quantile_deriv_samples(ds, apply_sobolev, apply_base, [idx,])
            gt, sob, base = gt[0], sob[0], base[0]
            gt_scale = np.max(jnp.abs(gt))
            gt, sob, base = [a/gt_scale for a in (gt, sob, base)]
        else:
            gt, sob, base = get_quantile_samples(ds, apply_sobolev, apply_base, [idx,])
            gt, sob, base = gt[0], sob[0], base[0]
        plot_row(axs[i, :], gt, sob, base, deriv)

    axs[0, 0].set_title(r"$\sigma(E)$", y=1.01)
    axs[0, 1].set_title("Baseline Error", y=1.01)
    axs[0, 2].set_title("DeepLight Error", y=1.01)

    axs[2, 0].set_title(r"$\sigma(\nabla_v E)$", y=1.01)
    axs[2, 1].set_title("Baseline Error", y=1.01)
    axs[2, 2].set_title("DeepLight Error", y=1.01)

    # column_label(axs[0, 0], r"$25^{\mathrm{th}}$ Percentile Gain")
    # column_label(axs[1, 0], r"$50^{\mathrm{th}}$ Percentile Gain")
    # column_label(axs[2, 0], r"$75^{\mathrm{th}}$ Percentile Gain")
    return f


if __name__ == "__main__":
    paths = dl.load_paths('./paths.toml')
    path = os.path.join(paths['results_dir'], "quantitative_results_with_deriv.h5")
    loss_df = pd.read_hdf(path, "loss")
    deriv_loss_df = pd.read_hdf(path, "deriv_loss")
    gain_df = pd.read_hdf(path, "gain")
    deriv_gain_df = pd.read_hdf(path, "deriv_gain")

    res_dir = paths['network_weights_dir']
    paths = get_optimal_runs(res_dir)
    models = [load_network(f"{path}/min_val.pkl", jax.devices()[0]) for path in paths]

    data_path = os.path.join(paths['data_dir'], "test_data.h5")
    test_data = dl.load_data(data_path)
    test = []
    n_pixels = (400, 272)
    data_id = [np.concat(_arr) for _arr in zip([arr[0::4] for arr in test_data], [arr[1::4] for arr in test_data])]
    test_id = dl.DataLoader(dl.DeepLightDataset(*data_id, grid_size=[n/1e2 for n in n_pixels], n_pixels=n_pixels), 4)

    data_ood = [arr[2::4] for arr in test_data]
    test_ood = dl.DataLoader(dl.DeepLightDataset(*data_ood, grid_size=[n/1e2 for n in n_pixels], n_pixels=n_pixels), 4)

    # f = qualitative_samples_sep(gain_df, "In Distribution", test_id, *models, (0.25, 0.5, 0.75), deriv=False)
    # f.savefig("../paper/Resources/qualitative_ID.png", dpi=600)

    # f = qualitative_samples_sep(gain_df, "Out-of-Distribution", test_ood, *models, (0.25, 0.5, 0.75), deriv=False)
    # f.savefig("../paper/Resources/qualitative_OOD.png", dpi=600)

    # f = qualitative_samples_sep(deriv_gain_df, "In Distribution_1", test_id, *models, (0.25, 0.5, 0.75), deriv=True)
    # f.savefig("../paper/Resources/qualitative_ID1_deriv.png", dpi=600)

    # f = qualitative_samples_sep(deriv_gain_df, "In Distribution_2", test_id, *models, (0.25, 0.5, 0.75), deriv=True)
    # f.savefig("../paper/Resources/qualitative_ID2_deriv.png", dpi=600)

    # f = qualitative_samples_sep(deriv_gain_df, "Out-of-Distribution", test_ood, *models, (0.25, 0.5, 0.75), deriv=True)
    # f.savefig("../paper/Resources/qualitative_OOD_deriv.png", dpi=600)


    f = qualitative_samples_combined(gain_df, deriv_gain_df, test_id, test_ood, *models)
    f.savefig("../paper/Resources/qualitative_combined.png", dpi=600)
