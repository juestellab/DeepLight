import os
import optax
import jax.numpy as jnp
import jax
from flax.training import train_state
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
from datetime import datetime
from typing import Any

from .network import UNet, UNet_basic


class UNetTrainState(train_state.TrainState):
    batch_stats: Any


def initialize_optimizer(lr, iteration_length, num_epochs, warmup, steps_per_epoch, weight_decay):
    schedules = [optax.linear_schedule(init_value=lr/20, end_value=lr,
                                       transition_steps=warmup * steps_per_epoch)]
    boundaries = [warmup * steps_per_epoch,]
    for n in range(num_epochs):
        steps = iteration_length * steps_per_epoch * 2**n
        schedules += [optax.cosine_decay_schedule(lr, steps, alpha=lr/200),]
        boundaries += [boundaries[-1] + steps,]
    lr_schedule = optax.join_schedules(schedules, boundaries=boundaries[:-1])
    tx = optax.adamw(learning_rate=lr_schedule,
                     weight_decay=weight_decay / jnp.sqrt(boundaries[-1]))

    assert boundaries[-1] % steps_per_epoch == 0
    return tx, boundaries[-1]//steps_per_epoch


def initialize(rng_key, model_type, unet_params, input_shape, optimizer_config, name, logdir):
    key_param, key_x = jax.random.split(rng_key, 2)
    if model_type == "basic":
        model = UNet_basic(*unet_params)
    elif model_type == "full":
        model = UNet(*unet_params)

    variables = model.init(key_param, jax.random.normal(key_x, input_shape), train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    optimizer, total_steps = initialize_optimizer(*optimizer_config)

    model_state = UNetTrainState.create(apply_fn=model.apply,
                                        params=params,
                                        batch_stats=batch_stats,
                                        tx=optimizer
                                        )

    logdir = os.path.join(logdir, f"{name}_{datetime.now()}")
    writer = SummaryWriter(logdir=logdir)

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    checkpoint_manager = ocp.CheckpointManager(logdir,
                                               item_names=("params", "batch_stats", "model_conf"),
                                               options=options)
    return model, model_state, writer, checkpoint_manager, total_steps
