import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import orbax.checkpoint as ocp
import pickle
# plt.style.use("./heatmap.mplstyle")


def nl(x, c):
    return jnp.sign(x) * jnp.log((jnp.abs(x) + c)/c)


def nl_deriv(x, c):
    return jnp.arcsinh(x/c)


def log_figures(state, data, writer, name, iterations):
    mu, dmu, E, dE = data
    res, dres = jax.jvp(lambda x: state.apply_fn({"params": state.params,
                                                  "batch_stats": state.batch_stats},
                                                 x, train=False),
                        (mu,), (dmu,))
    writer.add_figure(f"Energy/{name}",
                      make_plot(mu, res, E, 5, lambda x: nl(x, 1e-3)),
                      iterations)
    writer.add_figure(f"Sensitivity/{name}",
                      make_plot(dmu, dres, dE, 5, lambda x: nl_deriv(x, 1)),
                      iterations)


def log_intermediate(iterations, state,
                     train_loss,
                     val_loss,
                     test_losses,
                     train_ds, test_ds,
                     writer, checkpointer,
                     fig_every_n=50):
    writer.add_scalar("Loss/Operator/Training", train_loss[0], iterations)
    writer.add_scalar("Loss/Jacobian/Training", train_loss[1], iterations)
    writer.add_scalar("Loss/Operator/Validation", val_loss[0], iterations)
    writer.add_scalar("Loss/Jacobian/Validation", val_loss[1], iterations)
    for (i, test_loss) in enumerate(test_losses):
        writer.add_scalar(f"Loss/Operator/Test_{i+1}", test_loss[0], iterations)
        writer.add_scalar(f"Loss/Jacobian/Test_{i+1}", test_loss[1], iterations)

    if iterations % fig_every_n == 0:
        checkpointer.save(iterations, args=ocp.args.Composite(
                          params=ocp.args.StandardSave(state.params),
                          batch_stats=ocp.args.StandardSave(state.batch_stats))
                          )
        checkpointer.wait_until_finished()
        log_figures(state, train_ds.data[0:5], writer, "Training", iterations)
        for i in range(4):
            log_figures(state, test_ds.data[i:(4*5+i):4], writer, f"Test_{i+1}", iterations)

    if iterations % 100 == 0:
        with open(f"{writer.logdir}state_{iterations}.pkl", "wb") as f:
            pickle.dump(dict(params=state.params, batch_stats=state.batch_stats), f)


def make_plot(input, output, ground_truth, n_samples, f):
    fig, ax = plt.subplots(5, n_samples, figsize=(15, 2.05*n_samples))
    o = f(output)
    gt = f(ground_truth)
    for i in range(n_samples):
        ax[i][0].matshow(jnp.cbrt(input[i, :, :, 0]).T, cmap="inferno")
        ax[i][1].matshow(jnp.cbrt(input[i, :, :, 1]).T, cmap="inferno")
        ax[i][2].matshow(gt[i, :, :, 0].T,
                         cmap="inferno",
                         vmin=jnp.min(gt[i, :, :, 0]) / 1.1,
                         vmax=jnp.max(gt[i, :, :, 0]) * 1.1)
        ax[i][3].matshow(o[i, :, :, 0].T,
                         cmap="inferno",
                         vmin=jnp.min(gt[i, :, :, 0]) / 1.1,
                         vmax=jnp.max(gt[i, :, :, 0]) * 1.1)
        for j in range(5):
            ax[i][j].axis("off")

        residual = ground_truth[i, :, :, 0].T - output[i, :, :, 0].T
        lim = jnp.max(jnp.abs(residual))
        ax[i][4].matshow(residual, cmap="coolwarm",
                         vmin=-lim, vmax=lim)
    fig.tight_layout()
    return fig
