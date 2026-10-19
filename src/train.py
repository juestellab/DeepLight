import functools

import jax
import jax.numpy as jnp
import optax

from .logging import log_intermediate
from .serialization import model_config_from, save_network
from .utils import nl


def nl_transform(a, c):
    """`nl` with its parameters bound, as a one-argument function.

    The parameters are bound here, ahead of jit, rather than handed to the
    jitted step as arguments. `nl` picks its regime with a Python branch, so
    bound parameters are compile-time constants: the regimes not taken emit no
    code, and a limiting one costs nothing -- `nl_transform(1., LINEAR_C)` folds
    away to the identity. Passed as jit arguments they would be traced instead,
    the branch could not be taken at all, and every element would pay for the
    exp/log chain that the limit exists to avoid.
    """
    return functools.partial(nl, a=a, c=c)


def compute_loss(x, y, transform):
    return optax.l2_loss(transform(x), transform(y)).mean()


def _directional_derivatives(f, mu, dmu):
    """Primal and all K directional derivatives of `f` at `mu`.
    Axis 1 of dmu contains the different derivative directions.

    Returns (primal_out, dres) with dres stacking the different
    directional derivatives along axis 1.
    """
    res, f_lin = jax.linearize(f, mu)
    dres = jax.vmap(f_lin, in_axes=1, out_axes=1)(dmu)
    return res, dres


def operator_loss(nl_E):
    """The operator loss under the `nl` transform `nl_E`."""
    def loss_fn(state, params, batch):
        mu, _, E, _ = batch
        res, updates = state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                      mu, train=True, mutable=["batch_stats"])

        loss = compute_loss(res, E, nl_E)
        return loss, (updates, loss, 0)
    return loss_fn


def sobolev_loss(nl_E, nl_dE):
    """Sobolev loss: op_loss in train mode, jac_loss in *eval* mode.

    The Jacobian is taken at train=False on purpose. With BatchNorm, train mode
    normalizes by the current minibatch's statistics, so its JVP differentiates terms
    that are constant in eval mode. As the eval mode JVP is the optimization target,
    the JVP must also be evaluated in that mode.

    GroupNorm keeps no state and behaves identically in both modes, so no additional
    evaluation is required.

    `nl_E`, `nl_dE`: the bound `nl` transforms of the operator and Jacobian terms.
    """
    def loss_fn(state, params, batch, alpha):
        mu, dmu, E, dE = batch
        variables = {"params": params, "batch_stats": state.batch_stats}
        train_fn = lambda x: state.apply_fn(variables, x, train=True, mutable=["batch_stats"])

        # dict emptiness is pytree structure, so this branches at trace time
        if state.batch_stats:
            res, updates = train_fn(mu)
            eval_fn = lambda x: state.apply_fn(variables, x, train=False)
            _, dres = _directional_derivatives(eval_fn, mu, dmu)
        else:
            (res, updates), (dres, _) = _directional_derivatives(train_fn, mu, dmu)

        op_loss = compute_loss(res, E, nl_E)
        jac_loss = compute_loss(dres, dE, nl_dE)
        loss = op_loss + alpha * jac_loss
        return loss, (updates, op_loss, jac_loss)
    return loss_fn


def jacobian_loss(nl_dE):
    """The Jacobian term alone, for fine-tuning a trained operator network.

    `sobolev_loss` without its operator term, and not a special case of it: at
    alpha -> inf the operator term would still be differentiated for nothing.
    One term means the loss scale is again irrelevant -- Adam is invariant under
    a global rescale -- so the `nl` regimes are comparable here without the
    weighting they need against an operator term.

    E is then unconstrained wherever the sampled tangents do not reach, and will
    drift; the network this trains is a probe of what the Jacobian loss teaches,
    not one to ship.
    """
    def loss_fn(state, params, batch):
        mu, dmu, _E, dE = batch
        variables = {"params": params, "batch_stats": state.batch_stats}
        train_fn = lambda x: state.apply_fn(variables, x, train=True, mutable=["batch_stats"])

        # the eval-mode JVP is the target under BatchNorm; see sobolev_loss
        if state.batch_stats:
            _res, updates = train_fn(mu)
            eval_fn = lambda x: state.apply_fn(variables, x, train=False)
            _, dres = _directional_derivatives(eval_fn, mu, dmu)
        else:
            (_res, updates), (dres, _) = _directional_derivatives(train_fn, mu, dmu)

        loss = compute_loss(dres, dE, nl_dE)
        return loss, (updates, 0, loss)
    return loss_fn


def _train_step(loss_fn):
    """The jitted update step for `loss_fn`"""
    @jax.jit
    def step(state, batch, *loss_args):
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        (_, (updates, op_loss, jac_loss)), grads = grad_fn(state, state.params,
                                                           batch, *loss_args)
        state = state.apply_gradients(grads=grads)
        # .get: GroupNorm models have no batch_stats collection to update
        state = state.replace(batch_stats=updates.get("batch_stats", {}))
        return state, jnp.array([op_loss, jac_loss])
    return step


# The steps close over their nl parameters, so they are built per parameter set
# rather than at import. Cached on those parameters (plain floats, hence
# hashable) so a sweep of many runs sharing them compiles each step once
# instead of once per call to `train_epochs`.
@functools.cache
def _operator_step(a_E, c_E):
    return _train_step(operator_loss(nl_transform(a_E, c_E)))


@functools.cache
def _sobolev_step(a_E, c_E, a_dE, c_dE):
    return _train_step(sobolev_loss(nl_transform(a_E, c_E),
                                    nl_transform(a_dE, c_dE)))


@functools.cache
def _jacobian_step(a_dE, c_dE):
    return _train_step(jacobian_loss(nl_transform(a_dE, c_dE)))


@functools.cache
def eval_loss(a_E, c_E, a_dE, c_dE):
    """The jitted `(op_loss, jac_loss)` of one batch, under these nl parameters."""
    nl_E, nl_dE = nl_transform(a_E, c_E), nl_transform(a_dE, c_dE)

    @jax.jit
    def loss_fn(state, params, batch):
        mu, dmu, E, dE = batch
        res, dres = _directional_derivatives(
            lambda x: state.apply_fn({"params": params, "batch_stats": state.batch_stats},
                                     x, train=False),
            mu, dmu)

        op_loss = compute_loss(res, E, nl_E)
        jac_loss = compute_loss(dres, dE, nl_dE)
        return op_loss, jac_loss
    return loss_fn


def eval_dataset(state, ds, a_E, c_E, a_dE, c_dE):
    # float(): the nl parameters are cache keys, and a jnp scalar is unhashable
    batch_loss = eval_loss(float(a_E), float(c_E), float(a_dE), float(c_dE))
    loss_val = jnp.array([0, 0])
    for batch in ds:
        op_loss, jac_loss = batch_loss(state, state.params, batch)
        loss_val += jnp.array([op_loss, jac_loss])/len(ds)
    return loss_val


def train_epochs(state, model, num_epochs, train_ds, val_ds, test_ds, alpha,
                 writer, a_E, c_E, a_dE, c_dE, metadata=None):
    """Train for `num_epochs`, saving the best validation state.

    a_E, c_E, a_dE, c_dE: the `nl` parameters of the operator and Jacobian
    terms. They are bound into the loss ahead of jit rather than passed to it,
    so each regime -- including the linear (c=inf) and log (a=inf) limits --
    compiles to exactly the arithmetic it needs. See `nl_transform`.

    metadata: run identity (alpha, seed, c, depth_channel, oop_slices, ...)
    written into every checkpoint as HDF5 attributes.
    """
    metadata = dict(metadata or {})
    # float(): the nl parameters are cache keys, and a jnp scalar is unhashable
    a_E, c_E = float(a_E), float(c_E)
    a_dE, c_dE = float(a_dE), float(c_dE)
    # alpha=0 takes the operator loss, which needs no JVP at all -- the
    # no-derivative arm must not pay for a Jacobian it then multiplies by zero.
    # alpha=None is the mirror image: the Jacobian term with no operator term,
    # which is fine-tuning an already trained network (see `jacobian_loss`).
    if alpha is None:
        train_step, loss_args = _jacobian_step(a_dE, c_dE), ()
    elif alpha == 0:
        train_step, loss_args = _operator_step(a_E, c_E), ()
    else:
        train_step, loss_args = _sobolev_step(a_E, c_E, a_dE, c_dE), (alpha,)

    # Both the operator-only best network and the best network under the full Sobolev loss
    # are stored.
    best = {"min_operator_val": (jnp.inf, -1, -1, 0, None, None),
            "min_total_val": (jnp.inf, -1, -1, 0, None, None)}
    for epoch in range(num_epochs):
        train_loss = jnp.array([0, 0])
        for batch in train_ds:
            state, batch_loss = train_step(state, batch, *loss_args)
            train_loss += batch_loss/len(train_ds)

        val_loss = eval_dataset(state, val_ds, a_E, c_E, a_dE, c_dE)
        print(epoch, ": ", val_loss)

        # with no operator term the total is the Jacobian term, and the operator
        # checkpoint tracks a quantity nothing is optimizing -- kept anyway, as
        # how far E drifted is the diagnostic of that run
        total_val = (val_loss[1] if alpha is None
                     else val_loss[0] + alpha*val_loss[1])
        for name, loss in (("min_operator_val", val_loss[0]),
                           ("min_total_val", total_val)):
            if loss < best[name][0]:
                params, batch_stats = jax.device_get((state.params,
                                                      state.batch_stats))
                best[name] = (float(loss), float(val_loss[0]),
                              float(val_loss[1]), epoch, params, batch_stats)

        log_intermediate(epoch+1, state,
                         train_loss, val_loss,
                         train_ds, val_ds, test_ds,
                         writer)

    # Write best performing network params to disk
    for name, (loss, op_loss, jac_loss, best_epoch, params, batch_stats) in best.items():
        save_network(f"{writer.logdir}/{name}.h5", params, batch_stats,
                     model_config=model_config_from(model),
                     val_loss=loss, val_op_loss=op_loss, val_jac_loss=jac_loss,
                     epoch=best_epoch, **metadata)

    # Final logging, with guaranteed figures
    log_intermediate(epoch+1, state,
                     train_loss, val_loss,
                     train_ds, val_ds, test_ds,
                     writer, 1)
    return state
