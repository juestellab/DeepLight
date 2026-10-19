"""Train the Sobolev networks of the derivative loss-transform comparison.

The `train_nl_modes.py` study, one field along: there it was the operator loss
on E at alpha=0, here it is the Jacobian loss on dE at alpha>0, with E's
transform held at the sweep's own (`tn.A_E`, `tn.C_E`) so the arms differ in
nothing but how the derivative term weights dE's dynamic range.

    linear   a=A_LINEAR, c=inf   -- plain L2 on dE, scaled to match the log
                                    arm's loss magnitude (see below)
    log      a=inf,      c=C_DE  -- sign(x) log1p(|x|/c), the MC noise floor
    nl       a=A_DE,     c=C_DE  -- the main sweep's own run, not retrained here

One thing does not carry over from the E study. There the loss was the whole
objective, so its scale was irrelevant: Adam is invariant under a global rescale
of the loss, and `linear` at a=1 optimized the same thing as `log` at slope
1/c = 1e6. Here `nl_dE` also sets the weight of the Jacobian term *against* the
operator term, which no rescaling undoes -- the identity would put the
derivative term about 1/(c_dE) below the nl arm's and the "linear" arm would be
a no-derivative arm in disguise, measuring alpha rather than the transform.

So the linear arm's `a` is set by matching the *loss magnitude* instead. An
earlier version matched the transforms' slope at the origin, where the bulk of
dE sits; that is wrong by an order of magnitude, because a quadratic loss is set
by where the errors are, not by where the pixels are, and the errors are largest
exactly where the log transform's weight has decayed. The measured cost was an
operator val loss 3.7x the log arm's: the linear arm bought its derivative
accuracy with E.

A_LINEAR below is `sqrt(L_linear(a=1) / L_log)` over the whole validation set at
the operator-only network both arms would start from -- the one point where they
are on equal footing. It is stable across the pretrained networks (0.146, 0.148
at a_E=0.1; 0.160, 0.169, 0.173 at a_E=0.5), which is why two digits is all it
is quoted to.

The match holds at that starting point, not throughout training: measured at the
finished networks the same ratio gives 0.145 and 0.080, so the balance still
drifts by up to 2x as each arm's residuals reshape. No single `a` prevents that.
The check that it worked is the operator val loss -- if the arms are balanced,
the linear arm should land near the log arm's rather than well above it.

The crossing of the linear and log medians is what sets a_dE, so those two arms
are the study; the nl arm is the confirmation panel beside them and comes free
only if the main sweep has a run at this ARM, alpha and digest.

Runs land beside the main sweep's, named the same way plus the regime:

    arm=1dir_alpha=0.01_nl=log_seed=0_norm=group_depth=1_oop=65_cfg=1a2b3c4d

Everything defining the network, and the resume/metadata checks, are
`train_network`'s. The `cfg` digest covers the nl parameters, so a regime cannot
resume onto another's checkpoints.
"""
import json
import os
import shutil

import jax

import deeplight as dl
import train_network as tn

# The arm and Sobolev weight every regime is trained at, held fixed so the
# comparison is of the transform alone. Every stored direction of both training
# tangent samplers (ID_1 and SIGN, two each), which is the arm the paper's
# figure is drawn from -- the `1dir` pair that located the crossover trained on
# a quarter of the derivative signal.
ARM, N_DIRECTIONS, PER_SAMPLER = "2_per_sampler", 2, True
ALPHAS = (1,)
SEEDS = (0,)


# The linear arm's scale, so that its Jacobian loss starts out the same size as
# the log arm's rather than ~15x larger. See the module docstring for how it is
# measured.
A_LINEAR = 0.15

# floor of the log arm
C_LOG = 1e-3

# (a_dE, c_dE) per regime. `a_E`/`c_E` are the sweep's throughout.
NL_MODES = {
    "linear": (A_LINEAR, dl.LINEAR_C),
    "log": (dl.LOG_A, C_LOG),
}


def mode_config(data_dir, params):
    """`train_network`'s sweep config with this regime's Jacobian parameters."""
    a_dE, c_dE = params
    return dict(tn.sweep_config(data_dir), a_dE=a_dE, c_dE=c_dE)


def run_name(mode, alpha, seed_index, digest):
    """`train_network.run_name` plus the regime, arm and alpha in their places.

    `arm=<arm>_alpha=<alpha>` stays in front, so the globs that pick up the
    main sweep's runs of this arm pick these up as well.
    """
    return (f"arm={ARM}_alpha={alpha}_nl={mode}_seed={seed_index}"
            f"_norm={tn.NORM}_depth={int(tn.DEPTH_CHANNEL)}"
            f"_oop={tn.OOP_SLICES}_cfg={digest}")


def main():
    paths = dl.load_paths('./paths.toml')
    data_dir = paths['data_dir']
    weights_dir = paths["network_weights_dir"]
    commit, dirty = dl.git_commit(), dl.git_dirty()
    direction_spec = f"{N_DIRECTIONS}/{PER_SAMPLER}"
    direction_mode = "per_sampler" if PER_SAMPLER else "per_sample"

    # every stored direction of the training samplers, as the main sweep does,
    # so the arms are validated against the same Jacobian
    data_val = dl.load_data(f"{data_dir}/val_data.h5", tn.OOP_SLICES,
                            samplers=tn.TRAIN_SAMPLERS)
    val = dl.DataLoader(dl.DeepLightDataset(*data_val), tn.BATCH_SIZE)

    # Decide what is outstanding before loading the training set: resuming a
    # finished regime must not pay for its data.
    pending = []
    for (mode, params) in NL_MODES.items():
        config = mode_config(data_dir, params)
        digest = tn.config_digest(config)
        print(f"{mode:6s} a_dE={params[0]:g} c_dE={params[1]:g} "
              f"cfg {digest}")
        for alpha in ALPHAS:
            for i in SEEDS:
                name = run_name(mode, alpha, i, digest)
                run_dir = os.path.join(weights_dir, name)
                expected = dict(direction_arm=ARM, alpha=float(alpha),
                                seed_index=int(i), config_digest=digest,
                                direction_spec=direction_spec, nl_mode=mode)
                metadata = tn.finished_run(run_dir)
                if metadata is not None:
                    tn.check_resumable(run_dir, metadata, expected, commit,
                                       dirty)
                    print(f"  skip (done)   {name}")
                    continue
                if os.path.exists(run_dir):
                    print(f"  clearing incomplete {name}")
                    shutil.rmtree(run_dir)
                pending.append((mode, params, config, alpha, i, name, run_dir,
                                expected))

    print(f"\nnl arm (not trained here): "
          f"{tn.run_name(ARM, ALPHAS[0], SEEDS[0], tn.config_digest(tn.sweep_config(data_dir)))}")
    if not pending:
        print("nothing to do")
        return

    data_train = dl.load_data(f"{data_dir}/train_data.h5", tn.OOP_SLICES,
                              n_directions=N_DIRECTIONS,
                              per_sampler=PER_SAMPLER,
                              samplers=tn.TRAIN_SAMPLERS)
    train_ds = dl.DeepLightDataset(*data_train, n_samples=tn.N_TRAIN_SAMPLES)
    print(f"{train_ds.n_directions} training direction(s) per sample")

    for (mode, params, config, alpha, i, name, run_dir, expected) in pending:
        print(f"\ntrain {name}")
        a_dE, c_dE = params
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
                        a_E=float(tn.A_E), c_E=float(tn.C_E),
                        a_dE=float(a_dE), c_dE=float(c_dE),
                        depth_channel=bool(tn.DEPTH_CHANNEL),
                        oop_slices=int(tn.OOP_SLICES),
                        data_dir=str(data_dir),
                        n_pixels_x=int(tn.N_PIXELS[0]),
                        n_pixels_y=int(tn.N_PIXELS[1]),
                        n_train_samples=int(len(train_ds)),
                        n_directions=int(train_ds.n_directions),
                        direction_mode=direction_mode,
                        train_samplers=",".join(map(str, tn.TRAIN_SAMPLERS)),
                        batch_size=int(tn.BATCH_SIZE),
                        lr=float(tn.OPTIM["lr"]),
                        weight_decay=float(tn.OPTIM["weight_decay"]),
                        git_commit=commit, git_dirty=bool(dirty))

        ms = dl.train_epochs(ms, m, total_epochs, train, val, None, float(alpha),
                             w, tn.A_E, tn.C_E, a_dE, c_dE, metadata=metadata)


if __name__ == "__main__":
    main()
