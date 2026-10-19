"""Train the DeepLight networks.

    python train_network.py [sweep|seeds|size|depth_ablation]

The sweep is configured by the constants below; the other run groups of the
paper are overrides of them, see GROUPS. Run
identity (seed, alpha, the nl parameters, whether the depth channel was used,
which out-of-plane slice, which direction arm) is written into every checkpoint
as HDF5 metadata, so the analysis in run/compute_error.py reads it from the file
rather than parsing it back out of the run directory's name.

The sweep is DIRECTION_ARMS x seeds x ALPHAS, except for the no-derivative arm,
which is trained once per seed at alpha=0 -- see DIRECTION_ARMS.

Resuming
--------
The sweep is restartable: a run whose directory already holds finished
checkpoints is skipped, so an interrupted sweep continues where it stopped and
raising N_SEEDS trains only the new seeds. The registry is the weights directory
itself -- `run_name` maps a run's identity to a flat, deterministic directory
name, so a run is found by constructing its name rather than by scanning:

    arm=1dir_alpha=0.1_seed=0_norm=group_depth=0_oop=65_cfg=a3f19c2b

The `cfg` digest covers everything else that defines the network (feature
counts, optimizer, batch size, c_E/c_dE, training-set size, data directory,
SEED_KEY). Changing any of it yields a different directory, so a re-run forks a
new set of runs instead of colliding with the old one; `config.json` beside the
checkpoints records what a digest stands for. Names stay flat because
`compute_error.py` is handed run directories by a shell glob and `run_id`
becomes an HDF5 group name in the error store.

Before skipping, the checkpoint's own metadata is checked against the run about
to be trained -- the directory name is a lookup key, not evidence. A mismatch,
or a different git commit, is an error rather than a silent skip; a dirty
working tree is a warning, since the commit alone cannot tell two edited trees
apart.
"""
import argparse
import hashlib
import json
import os
import shutil

import jax

import deeplight as dl

# ---------------------------------------------------------------- sweep config
N_SEEDS = 2
SEED_KEY = 7864532
ALLOW_COMMIT_MISMATCH = True   # resume across commits; the sweep then mixes code
ALPHAS = (0.01, 0.1, 1, 10)     # Sobolev weights of the derivative arms
DEPTH_CHANNEL = True            # False = depth-channel ablation (plain UNet only)
A_E = 0.1
C_E = dl.DEFAULT_C_E            # nl cutoff parameter for the operator term
# where the linear and log arms of the derivative transform study cross, on the
# 2_per_sampler pair with the linear arm scaled to match the log arm's loss
# magnitude: log weighting wins below it, linear above (see train_nl_modes_dE)
A_DE = 1                        # nl absolute scale parameter for the Jacobian term
C_DE = dl.DEFAULT_C_DE          # nl cutoff parameter for the Jacobian term
OOP_SLICES = 65                 # out-of-plane half-width group in the data file
NORM = 'group'                  # "batch" | "group"; group keeps no batch_stats
N_GROUPS = 8
FEATURE_COUNTS = (16, 24, 32, 48, 64, 64)
N_PIXELS = (400, 272)
BATCH_SIZE = 4
OPTIM = dict(lr=1e-3, iteration_length=100, n_cycles=2, weight_decay=1e-5)
N_TRAIN_SAMPLES = None          # None = all; a small int gives a cheap smoke run

# How much derivative signal the Jacobian loss is trained on -- one arm per
# entry, as (label, n_directions, per_sampler). Each sample stores several
# directions per tangent sampler (ID_1, ID_2, SIGN x 2 by default). With
# per_sampler=True the count is applied within each sampler, so a sample keeps
# that many from every one of them; with per_sampler=False it is the count *per
# sample*, filled round-robin across the samplers so the set stays balanced
# across them without every sample carrying all three. The Jacobian loss
# averages over the kept directions.
#
# n_directions=None is the no-derivative arm. train.operator_loss ignores dmu/dE
# entirely, so at alpha=0 the direction arm cannot make any difference: it is
# one arm trained at alpha=0, not a variant of each of the others. Running it
# per direction arm would train the same network three times.
#
# Only the *training* set is narrowed by the arm. Validation keeps every
# direction of TRAIN_SAMPLERS, so the arms are scored against the same Jacobian
# and their losses stay comparable; the test file keeps all of them.
DIRECTION_ARMS = (
    ("no_deriv", None, None),       # the baseline: alpha=0, no derivative term
    ("1dir", 1, False),             # one direction per sample, rotating samplers
    ("1_per_sampler", 1, True),     # one direction from every tangent sampler
    ("2_per_sampler", 2, True),     # two from every tangent sampler
)

# Tangent samplers a network may ever see, as indices into the data file's
# `direction_sampler_names` (ID_1, ID_2, SIGN). ID_2 (index 1) is held out of
# training and validation entirely, so its directions are an out-of-distribution
# derivative at test time rather than one the network was fitted on.
TRAIN_SAMPLERS = (0, 2)

GRID_SIZE = tuple(n/1e2 for n in N_PIXELS)   # cm, at 100 px/cm

# keywords of network.UNet; dl.initialize builds the model from these. The
# geometry belongs to the model: it builds its own depth channel and is stored
# in the checkpoint's model_config, so an arm cannot be scored on inputs that
# disagree with what it was trained on.
UNET_PARAMS = dict(feature_counts=FEATURE_COUNTS, kernel=(3, 3),
                   activation=jax.nn.celu, final_activation=jax.nn.softplus,
                   n_pixels=tuple(N_PIXELS), grid_size=GRID_SIZE,
                   norm=NORM, n_groups=N_GROUPS,
                   depth_channel=DEPTH_CHANNEL)

# Both files train_epochs writes, at the very end of a run. A directory holding
# both, openable, is a finished run; anything else is a crash to train again.
CHECKPOINTS = ("min_operator_val.h5", "min_total_val.h5")

# The run groups of the paper, as overrides of the sweep constants above; the
# group is the script's argument. `seeds` only adds seeds to two arms of the
# sweep, so it shares the sweep's digest. `size` is the equal simulation budget:
# a quarter of the training samples at 4x the iterations per cycle, so the
# number of optimizer steps is unchanged.
GROUPS = {
    "sweep": {},
    "seeds": dict(arms=("no_deriv", "2_per_sampler"), alphas=(1,), n_seeds=5),
    "size": dict(arms=("2_per_sampler",), alphas=(1,), n_train_samples=512,
                 iteration_length=400),
    "depth_ablation": dict(arms=("no_deriv", "2_per_sampler"), alphas=(1,),
                           depth_channel=False),
}


def group_settings(group):
    """The sweep constants with `GROUPS[group]` applied."""
    g = GROUPS[group]
    arms = tuple(a for a in DIRECTION_ARMS
                 if a[0] in g.get("arms", [a[0] for a in DIRECTION_ARMS]))
    depth_channel = g.get("depth_channel", DEPTH_CHANNEL)
    return dict(arms=arms,
                alphas=g.get("alphas", ALPHAS),
                n_seeds=g.get("n_seeds", N_SEEDS),
                n_train_samples=g.get("n_train_samples", N_TRAIN_SAMPLES),
                depth_channel=depth_channel,
                unet_params=dict(UNET_PARAMS, depth_channel=depth_channel),
                optim=dict(OPTIM, iteration_length=g.get(
                    "iteration_length", OPTIM["iteration_length"])))


# ------------------------------------------------------------------- resuming

def seed_key(index):
    """Training key for seed `index`, independent of how many seeds are run.

    `fold_in` is stable in the index by contract, which is what lets N_SEEDS be
    raised without disturbing the seeds already trained. Under the current
    threefry2x32 PRNG it happens to return exactly `split(root, N)[index]`, so
    this changes no seed the old code would have produced -- but that agreement,
    and split's prefix-stability in N, are properties of the implementation that
    jax does not promise across versions or PRNG impls. Deriving the key from
    the index alone does not depend on either.
    """
    return jax.random.fold_in(jax.random.key(SEED_KEY), index)


def sweep_config(data_dir, group="sweep"):
    """Everything that defines the networks but is not a sweep axis.

    The sweep axes (direction arm, alpha, seed index) name the run directory;
    this is what the `cfg` digest covers. `data_dir` is in here because a
    different dataset is a different network -- moving the data forces a
    retrain, which is the safe direction to err in.
    """
    s = group_settings(group)
    return dict(unet=s["unet_params"], optim=s["optim"], batch_size=BATCH_SIZE,
                a_E = A_E, c_E=C_E, a_dE=A_DE, c_dE=C_DE, oop_slices=OOP_SLICES,
                n_train_samples=s["n_train_samples"], data_dir=str(data_dir),
                train_samplers=list(TRAIN_SAMPLERS), seed_root=SEED_KEY)


def group_runs(group, weights_dir, data_dir):
    """`(arm, alpha, seed_index, run_dir)` of every run `group` trains."""
    s = group_settings(group)
    digest = config_digest(sweep_config(data_dir, group))
    for (arm, n_directions, _) in s["arms"]:
        alphas = (0.0,) if n_directions is None else s["alphas"]
        for i in range(s["n_seeds"]):
            for alpha in alphas:
                name = run_name(arm, alpha, i, digest, s["depth_channel"])
                yield (arm, alpha, i, os.path.join(weights_dir, name))


def config_digest(config):
    """Short stable digest of `config`.

    Hashed over canonical JSON rather than with `hash()`, which is salted per
    process and would give a different answer on every invocation. `dl.jsonify`
    renders the activation functions by name; it also maps tuples to lists, so a
    config digests the same whichever the constants happen to be.
    """
    canonical = json.dumps(dl.jsonify(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def run_name(arm, alpha, seed_index, digest, depth_channel=DEPTH_CHANNEL):
    """The run's directory name: flat, deterministic, and greppable.

    Flat because a shell glob over one level is how `compute_error.py` is given
    its runs, and the directory name becomes the `run_id` the error store uses
    as an HDF5 group name. The named fields are the ones that vary between
    sweeps and are worth globbing on; the digest covers the rest.
    """
    return (f"arm={arm}_alpha={alpha}_seed={seed_index}"
            f"_norm={NORM}_depth={int(depth_channel)}_oop={OOP_SLICES}"
            f"_cfg={digest}")


def finished_run(run_dir):
    """The metadata of a finished run in `run_dir`, or None if it never finished.

    Reads attributes only: identifying a run must not pull every parameter tree
    in the sweep off disk.
    """
    stored = []
    for name in CHECKPOINTS:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            return None
        try:
            # every one is opened: a crash during the final write leaves a
            # truncated file that exists but cannot be read
            stored.append(dl.load_metadata(path))
        except Exception as exc:                        # noqa: BLE001
            print(f"  {run_dir}/{name} is unreadable "
                  f"({type(exc).__name__}: {exc})")
            return None
    return stored[0]["metadata"]


def check_resumable(run_dir, metadata, expected, commit, dirty):
    """Whether the finished run in `run_dir` really is the one about to be run.

    The directory name is a lookup key, not evidence: a name collision, a moved
    checkpoint or a redefined direction arm would all present as "already
    trained". Anything inconsistent is an error, because the alternative is
    quietly reporting someone else's network as this arm's result.
    """
    differing = {k: (metadata.get(k), v) for k, v in expected.items()
                 if metadata.get(k) != v}
    if differing:
        raise SystemExit(
            f"{run_dir}\nholds a finished run that disagrees with the one being "
            "trained:\n" + "\n".join(f"  {k}: stored {s!r}, expected {e!r}"
                                     for k, (s, e) in sorted(differing.items()))
            + "\nMove or delete that directory if it is stale.")

    stored_commit = metadata.get("git_commit")
    if stored_commit != commit:
        if not ALLOW_COMMIT_MISMATCH:
            raise SystemExit(
                f"{run_dir}\nwas trained at commit {stored_commit}, but this "
                f"tree is at {commit}. Check out that commit to extend the "
                "sweep, or set ALLOW_COMMIT_MISMATCH=True to mix code across "
                "runs deliberately.")
        print(f"  WARNING: trained at {stored_commit}, resuming from {commit}")
    if dirty or metadata.get("git_dirty"):
        print(f"  WARNING: {run_dir} involves a dirty working tree; the commit "
              "alone does not pin down the code that produced it")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train one run group of the paper.")
    p.add_argument("group", nargs="?", default="sweep", choices=tuple(GROUPS))
    group = p.parse_args(argv).group
    settings = group_settings(group)

    paths = dl.load_paths('./paths.toml')
    data_dir = paths['data_dir']

    # datasets are generated by run/generate_tissue.py (tissue properties) and
    # run/generate_light_transport.jl (E and dE), in that order

    data_val = dl.load_data(f"{data_dir}/val_data.h5", OOP_SLICES,
                            samplers=TRAIN_SAMPLERS)
    # data_test = dl.load_data(f"{data_dir}/test_data.h5", OOP_SLICES)

    # scored on every stored direction, so the direction arms stay comparable
    val_ds = dl.DeepLightDataset(*data_val)
    # test_ds = dl.DeepLightDataset(*data_test)

    commit, dirty = dl.git_commit(), dl.git_dirty()
    weights_dir = paths["network_weights_dir"]
    config = sweep_config(data_dir, group)
    digest = config_digest(config)
    print(f"{group}: config {digest} at commit {commit}{' (dirty)' if dirty else ''}\n")

    val = dl.DataLoader(val_ds, BATCH_SIZE)
    # test = dl.DataLoader(test_ds, BATCH_SIZE)

    for (arm, n_directions, per_sampler) in settings["arms"]:
        no_deriv = n_directions is None
        alphas = (0.0,) if no_deriv else settings["alphas"]
        # The arm's own definition, so redefining a label is caught rather than
        # resumed onto. Known before any data is read, unlike the resulting
        # direction count.
        direction_spec = f"{n_directions}/{per_sampler}"
        direction_mode = ("none" if no_deriv else
                          "per_sampler" if per_sampler else "per_sample")

        # Decide what is outstanding *before* loading anything: resuming a
        # finished arm must not pay for its training set.
        pending = []
        for i in range(settings["n_seeds"]):
            for alpha in alphas:
                name = run_name(arm, alpha, i, digest,
                                settings["depth_channel"])
                run_dir = os.path.join(weights_dir, name)
                expected = dict(direction_arm=str(arm), alpha=float(alpha),
                                seed_index=int(i), config_digest=digest,
                                direction_spec=direction_spec)
                metadata = finished_run(run_dir)
                if metadata is not None:
                    check_resumable(run_dir, metadata, expected, commit, dirty)
                    print(f"skip (done)   {name}")
                    continue
                if os.path.exists(run_dir):
                    # A crash left a partial run. Its event file would otherwise
                    # overlay the retry's curves in tensorboard.
                    print(f"clearing incomplete {name}")
                    shutil.rmtree(run_dir)
                pending.append((i, alpha, name, run_dir, expected))

        if not pending:
            print(f"{arm}: nothing to do\n")
            continue

        # Re-read per arm: the directions are selected at load time, so only the
        # kept ones are ever held. The no-derivative arm never reads dmu/dE, so
        # it keeps a single direction purely to save that memory.
        data_train = dl.load_data(
            f"{data_dir}/train_data.h5", OOP_SLICES,
            n_directions=1 if no_deriv else n_directions,
            per_sampler=False if no_deriv else per_sampler,
            samplers=TRAIN_SAMPLERS)
        train_ds = dl.DeepLightDataset(*data_train,
                                       n_samples=settings["n_train_samples"])
        if no_deriv:
            print(f"{arm}: no derivative term, alpha=0")
        else:
            print(f"{arm}: {train_ds.n_directions} training direction(s) per "
                  f"sample ({n_directions} per "
                  f"{'tangent sampler' if per_sampler else 'sample, round-robin'})")

        for (i, alpha, name, run_dir, expected) in pending:
            print(f"train         {name}")
            key = seed_key(i)
            init_key, train_shuffle = jax.random.split(key, 2)
            train = dl.DataLoader(train_ds, BATCH_SIZE, train_shuffle)

            m, ms, w, total_epochs = dl.initialize(
                init_key,
                unet_params=settings["unet_params"],
                optimizer_config=settings["optim"],
                train_batches=len(train),
                name=name,
                logdir=weights_dir)

            # what `cfg=<digest>` in the directory name stands for
            with open(os.path.join(w.logdir, "config.json"), "w") as f:
                json.dump(dl.jsonify(config), f,
                          sort_keys=True, indent=2)

            metadata = dict(**expected,
                            seed=int(jax.random.key_data(key).sum()),
                            seed_root=int(SEED_KEY),
                            a_E=float(A_E),
                            c_E=float(C_E),
                            a_dE=float(A_DE),
                            c_dE=float(C_DE),
                            depth_channel=bool(settings["depth_channel"]),
                            oop_slices=int(OOP_SLICES),
                            data_dir=str(data_dir),
                            n_pixels_x=int(N_PIXELS[0]),
                            n_pixels_y=int(N_PIXELS[1]),
                            n_train_samples=int(len(train_ds)),
                            n_directions=(0 if no_deriv
                                          else int(train_ds.n_directions)),
                            direction_mode=direction_mode,
                            # a string, so the metadata stays flat scalars
                            train_samplers=",".join(map(str, TRAIN_SAMPLERS)),
                            batch_size=int(BATCH_SIZE),
                            lr=float(settings["optim"]["lr"]),
                            weight_decay=float(settings["optim"]["weight_decay"]),
                            git_commit=commit,
                            git_dirty=bool(dirty))

            # ms = dl.train_epochs(ms, m, total_epochs, train, val, test, alpha,
            ms = dl.train_epochs(ms, m, total_epochs, train, val, None, alpha,
                                 w, A_E, C_E, A_DE, C_DE, metadata=metadata)


if __name__ == "__main__":
    main()
