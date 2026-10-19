import os
import optax
import jax.numpy as jnp
import jax
from flax.training import train_state
from tensorboardX import SummaryWriter
from typing import Any

from .network import UNet, N_MU_CHANNELS


class UNetTrainState(train_state.TrainState):
    batch_stats: Any


def initialize_optimizer(lr, iteration_length, n_cycles, weight_decay, steps_per_epoch):
    """Back-to-back cosine decay cycles of doubling length"""
    schedules, boundaries = [], [0]
    for n in range(n_cycles):
        steps = iteration_length * steps_per_epoch * 2**n
        schedules += [optax.cosine_decay_schedule(lr, steps, alpha=lr/200),]
        boundaries += [boundaries[-1] + steps,]
    total_steps = boundaries[-1]
    lr_schedule = optax.join_schedules(schedules, boundaries=boundaries[1:-1])
    tx = optax.adamw(learning_rate=lr_schedule,
                     weight_decay=weight_decay / jnp.sqrt(total_steps))

    assert total_steps % steps_per_epoch == 0
    return tx, total_steps//steps_per_epoch


def initialize(rng_key, unet_params, optimizer_config, train_batches, name, logdir):
    """Build the model, its train state, and a tensorboard writer"""
    key_param, key_x = jax.random.split(rng_key, 2)
    model = UNet(**unet_params)
    input_shape = (1, *model.n_pixels, N_MU_CHANNELS)

    variables = model.init(key_param, jax.random.normal(key_x, input_shape), train=False)
    params = variables["params"]
    # empty when the model normalizes with GroupNorm, which keeps no state
    batch_stats = variables.get("batch_stats", {})

    optimizer, total_steps = initialize_optimizer(**optimizer_config, steps_per_epoch=train_batches)

    model_state = UNetTrainState.create(apply_fn=model.apply,
                                        params=params,
                                        batch_stats=batch_stats,
                                        tx=optimizer
                                        )

    writer = SummaryWriter(logdir=os.path.join(logdir, name))

    return model, model_state, writer, total_steps
