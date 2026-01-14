import pickle
import jax
import jax.numpy as jnp
import optax
from functools import partial
# import mlcollections

from .logging import log_intermediate
E_SCALE = 5e-4
dE_SCALE = 1e-2


def nl(x, c):
    return jnp.sign(x) * jnp.log((jnp.exp(c) - jnp.exp(-jnp.abs(x)))/(jnp.exp(c) - 1)) + x


def compute_loss(x, y, nl, c):
    return optax.l2_loss(nl(x, c), nl(y, c)).mean()


@jax.jit
def train_loss(state, params, batch):
    mu, _, E, _ = batch
    res, updates = state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                  mu, train=True, mutable=["batch_stats"])

    loss = compute_loss(res, E, nl, E_SCALE)
    return loss, (updates, loss, 0)


@jax.jit
def _train_step(state, batch):
    grad_fn = jax.value_and_grad(train_loss,
                                 argnums=1,
                                 has_aux=True)
    (_, (updates, op_loss, jac_loss)), grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, jnp.array([op_loss, jac_loss])


@jax.jit
def train_loss_sobolev(state, params, batch, alpha):
    mu, dmu, E, dE = batch
    (res, updates), (dres, _) = jax.jvp(
        lambda x: state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                 x, train=True, mutable=["batch_stats"]),
        (mu,), (dmu,))

    op_loss = compute_loss(res, E, nl, E_SCALE)
    jac_loss = compute_loss(dres, dE, nl, dE_SCALE)
    loss = op_loss + alpha * jac_loss
    return loss, (updates, op_loss, jac_loss)


@jax.jit
def _train_step_sobolev(state, batch, alpha):
    grad_fn = jax.value_and_grad(train_loss_sobolev,
                                 argnums=1,
                                 has_aux=True)
    (_, (updates, op_loss, jac_loss)), grads = grad_fn(state, state.params, batch, alpha)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, jnp.array([op_loss, jac_loss])


@jax.jit
def eval_loss(state, params, batch):
    mu, dmu, E, dE = batch
    res, dres = jax.jvp(
        lambda x: state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                 x, train=False),
        (mu,), (dmu,))

    op_loss = compute_loss(res, E, nl, E_SCALE)
    jac_loss = compute_loss(dres, dE, nl, dE_SCALE)
    return op_loss, jac_loss


def eval_dataset(state, ds):
    loss_val = jnp.array([0, 0])
    for batch in ds:
        op_loss, jac_loss = eval_loss(state, state.params, batch)
        loss_val += jnp.array([op_loss, jac_loss])/len(ds)
    return loss_val


def train_epochs(state, num_epochs, train_ds, val_ds, test_ds, alpha, writer, checkpointer):
    if alpha == 0:
        train_step = _train_step
    else:
        train_step = partial(_train_step_sobolev, alpha=alpha)

    val_loss_min = jnp.inf
    for epoch in range(num_epochs):
        train_loss = jnp.array([0, 0])
        for batch in train_ds:
            state, batch_loss = train_step(state, batch)
            train_loss += batch_loss/len(train_ds)

        val_loss = eval_dataset(state, val_ds)
        if val_loss[0] < val_loss_min:
            val_loss_min = val_loss[0]
            with open(f"{writer.logdir}min_val.pkl", "wb") as f:
                pickle.dump(dict(params=state.params,
                                 batch_stats=state.batch_stats,
                                 val_loss=val_loss_min),
                            f)

        test_losses = []
        test_losses.append(eval_dataset(state, test_ds))
        # for test in test_ds:
        #     test_losses.append(eval_dataset(state, test))
        print(epoch, ": ", train_loss)
        log_intermediate(epoch+1, state,
                         train_loss, val_loss,
                         test_losses,
                         train_ds, test_ds,
                         writer, checkpointer, 25)
    # Final logging
    log_intermediate(epoch+1, state,
                     train_loss, val_loss,
                     test_losses,
                     train_ds, test_ds,
                     writer, checkpointer, 1)
    return state
