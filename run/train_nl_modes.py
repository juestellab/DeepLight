"""Train the non-Sobolev networks of the loss-transform comparison.

One network per `nl` regime, all at alpha=0 with no Jacobian term, so the only
thing that differs between them is how the operator loss weights E's dynamic
range:

    linear   a=1,   c=inf   -- nl is the identity, plain L2 on E
    log      a=inf, c=1e-6  -- sign(x) log1p(|x|/c), the log with the same
                               floor the `log` metric in evaluation.py uses

The third arm of the comparison is the main sweep's own no-derivative run
(`run/train_network.py`, `arm=no_deriv`), trained at the default nl parameters;
it is not retrained here. Everything that defines the network is imported from
that script rather than restated, so the three arms differ in the loss and in
nothing else -- change a feature count there and it applies here too.

Runs land beside the main sweep's, named the same way plus the regime:

    arm=no_deriv_nl=log_seed=0_norm=group_depth=1_oop=65_cfg=5d4e1a07

The `cfg` digest covers the nl parameters, so a regime cannot resume onto
another's checkpoints even though the rest of the configuration is identical.
Resuming, the metadata check and the commit check are `train_network`'s.
"""
import json
import os
import shutil

import jax

import deeplight as dl
import train_network as tn

# one network per regime
SEEDS = (0,)

# (a_E, c_E, a_dE, c_dE) per regime. The Jacobian parameters are carried only so
# the metadata records what the run would have used: at alpha=0 the operator
# loss is the whole objective and no Jacobian is ever taken.
NL_MODES = {
    "nl": (0.1, dl.LOG_C, 0.1, dl.LOG_C),
    "linear": (dl.LINEAR_A, dl.LINEAR_C, dl.LINEAR_A, dl.LINEAR_C),
    # the unfloored log (c=0) is -inf at E=0 and spends its capacity on a tail
    # that reaches 2e-8; the floor is where the `log` metric already puts it
    "log": (dl.LOG_A, dl.LOG_C, dl.LOG_A, dl.LOG_C),
}


def mode_config(data_dir, params):
    """`train_network`'s sweep config with this regime's nl parameters."""
    a_E, c_E, a_dE, c_dE = params
    return dict(tn.sweep_config(data_dir),
                a_E=a_E, c_E=c_E, a_dE=a_dE, c_dE=c_dE)


def run_name(mode, seed_index, digest):
    """The run's directory name, `train_network.run_name` plus the regime.

    `arm=no_deriv` stays in front so one glob over the weights directory still
    picks up every non-Sobolev run, this script's and the main sweep's alike.
    """
    return (f"arm=no_deriv_nl={mode}_seed={seed_index}"
            f"_norm={tn.NORM}_depth={int(tn.DEPTH_CHANNEL)}"
            f"_oop={tn.OOP_SLICES}_cfg={digest}")


def main():
    paths = dl.load_paths('./paths.toml')
    data_dir = paths['data_dir']
    weights_dir = paths["network_weights_dir"]
    commit, dirty = dl.git_commit(), dl.git_dirty()

    data_val = dl.load_data(f"{data_dir}/val_data.h5", tn.OOP_SLICES,
                            samplers=tn.TRAIN_SAMPLERS)
    val = dl.DataLoader(dl.DeepLightDataset(*data_val), tn.BATCH_SIZE)

    # Decide what is outstanding before loading the training set, as the main
    # sweep does: resuming a finished regime must not pay for its data.
    pending = []
    for (mode, params) in NL_MODES.items():
        config = mode_config(data_dir, params)
        digest = tn.config_digest(config)
        print(f"{mode:6s} a_E={params[0]:g} c_E={params[1]:g}  cfg {digest}")
        for i in SEEDS:
            name = run_name(mode, i, digest)
            run_dir = os.path.join(weights_dir, name)
            expected = dict(direction_arm="no_deriv", alpha=0.0,
                            seed_index=int(i), config_digest=digest,
                            direction_spec="None/None", nl_mode=mode)
            metadata = tn.finished_run(run_dir)
            if metadata is not None:
                tn.check_resumable(run_dir, metadata, expected, commit, dirty)
                print(f"  skip (done)   {name}")
                continue
            if os.path.exists(run_dir):
                print(f"  clearing incomplete {name}")
                shutil.rmtree(run_dir)
            pending.append((mode, params, config, i, name, run_dir, expected))

    if not pending:
        print("\nnothing to do")
        return

    # One direction is kept purely to save memory: the operator loss never
    # reads dmu/dE.
    data_train = dl.load_data(f"{data_dir}/train_data.h5", tn.OOP_SLICES,
                              n_directions=1, per_sampler=False,
                              samplers=tn.TRAIN_SAMPLERS)
    train_ds = dl.DeepLightDataset(*data_train, n_samples=tn.N_TRAIN_SAMPLES)

    for (mode, params, config, i, name, run_dir, expected) in pending:
        print(f"\ntrain {name}")
        a_E, c_E, a_dE, c_dE = params
        key = tn.seed_key(i)
        init_key, train_shuffle = jax.random.split(key, 2)
        train = dl.DataLoader(train_ds, tn.BATCH_SIZE, train_shuffle)

        m, ms, w, total_epochs = dl.initialize(
            init_key, unet_params=tn.UNET_PARAMS, optimizer_config=tn.OPTIM,
            train_batches=len(train), name=name, logdir=weights_dir)

        with open(os.path.join(w.logdir, "config.json"), "w") as f:
            json.dump(dl.jsonify(config), f, sort_keys=True, indent=2)

        metadata = dict(**expected,
                        seed=int(jax.random.key_data(key).sum()),
                        seed_root=int(tn.SEED_KEY),
                        a_E=float(a_E), c_E=float(c_E),
                        a_dE=float(a_dE), c_dE=float(c_dE),
                        depth_channel=bool(tn.DEPTH_CHANNEL),
                        oop_slices=int(tn.OOP_SLICES),
                        data_dir=str(data_dir),
                        n_pixels_x=int(tn.N_PIXELS[0]),
                        n_pixels_y=int(tn.N_PIXELS[1]),
                        n_train_samples=int(len(train_ds)),
                        n_directions=0,
                        direction_mode="none",
                        train_samplers=",".join(map(str, tn.TRAIN_SAMPLERS)),
                        batch_size=int(tn.BATCH_SIZE),
                        lr=float(tn.OPTIM["lr"]),
                        weight_decay=float(tn.OPTIM["weight_decay"]),
                        git_commit=commit, git_dirty=bool(dirty))

        ms = dl.train_epochs(ms, m, total_epochs, train, val, None, 0.0,
                             w, a_E, c_E, a_dE, c_dE, metadata=metadata)


if __name__ == "__main__":
    main()
