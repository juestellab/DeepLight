import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from .utils import nl, DEFAULT_A_E, DEFAULT_C_E, DEFAULT_A_DE, DEFAULT_C_DE
# plt.style.use("./heatmap.mplstyle")


def log_figures(state, data, writer, name, iterations):
    mu, dmu, E, dE = data
    res, f_lin = jax.linearize(
        lambda x: state.apply_fn({"params": state.params,
                                  "batch_stats": state.batch_stats},
                                 x, train=False), mu)
    dres = jax.vmap(f_lin, in_axes=1, out_axes=1)(dmu)

    writer.add_figure(f"Energy/{name}",
                      make_plot(mu, res, E, lambda x: nl(x, DEFAULT_A_E, DEFAULT_C_E)),
                      iterations)
    s = jnp.arange(dmu.shape[0])
    d = s % dmu.shape[1]
    writer.add_figure(f"Sensitivity/{name}",
                      make_plot(dmu[s, d], dres[s, d], dE[s, d],
                                lambda x: nl(x, DEFAULT_A_DE, DEFAULT_C_DE)),
                      iterations)


def log_intermediate(iterations, state,
                     train_loss,
                     val_loss,
                     train_ds,
                     val_ds,
                     test_ds,
                     writer,
                     fig_every_n=100):
    writer.add_scalar("Loss/Operator/Training", train_loss[0], iterations)
    writer.add_scalar("Loss/Jacobian/Training", train_loss[1], iterations)
    writer.add_scalar("Loss/Operator/Validation", val_loss[0], iterations)
    writer.add_scalar("Loss/Jacobian/Validation", val_loss[1], iterations)

    if iterations % fig_every_n == 0:
        log_figures(state, train_ds.data[0:6], writer, "Training", iterations)
        log_figures(state, val_ds.data[0:6], writer, "Validation", iterations)
        # for i in range(4):
        #     log_figures(state, test_ds.data[i:(4*5+i):4], writer, f"Test_{i+1}", iterations)


def make_plot(inputs, output, ground_truth, f):
    n_samples, n_in = inputs.shape[0], inputs.shape[-1]
    fig, ax = plt.subplots(n_samples, n_in + 3, squeeze=False,
                           figsize=(3*(n_in + 3), 2.05*n_samples))
    o = f(output)
    gt = f(ground_truth)
    for i in range(n_samples):
        ims = [ax[i][c].matshow(jnp.cbrt(inputs[i, ..., c]).T, cmap="inferno")
               for c in range(n_in)]

        c_min, c_max = jnp.min(gt[i, ..., 0]) / 1.1, jnp.max(gt[i, ..., 0]) * 1.1
        ims += [ax[i][n_in].matshow(gt[i, ..., 0].T, cmap="inferno",
                                    vmin=c_min, vmax=c_max),
                ax[i][n_in + 1].matshow(o[i, ..., 0].T, cmap="inferno",
                                        vmin=c_min, vmax=c_max),
                ax[i][n_in + 2].matshow((gt[i, ..., 0] - o[i, ..., 0]).T,
                                        cmap="coolwarm", norm=CenteredNorm())]

        for a, im in zip(ax[i], ims):
            a.axis("off")
            fig.colorbar(im, ax=a, fraction=0.04, pad=0.02, shrink=0.8)
    fig.tight_layout()
    return fig
