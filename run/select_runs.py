"""Print the run directories one scoring step of run/pipeline.sh needs.

    python select_runs.py all            # every run of train_network's groups
    python select_runs.py depth          # best seeds, with and without depth channel
    python select_runs.py arms           # best baseline and DeepLight seed
    python select_runs.py transform_E    # train_nl_modes.py and its sweep reference
    python select_runs.py transform_dE   # train_nl_modes_dE.py and its sweep reference

Runs are found by constructing their names, as the training scripts do, and
every selected run has to be finished. "Best" is the seed with the lowest
operator validation loss, as in `results_figure.best_seed`.
"""
import argparse
import os

import deeplight as dl
import train_network as tn
import train_nl_modes as nlm
import train_nl_modes_dE as nlm_de

ARM = ("2_per_sampler", 1)          # the DeepLight network: arm and alpha
BASELINE = ("no_deriv", 0.0)


def finished(run_dir):
    metadata = tn.finished_run(run_dir)
    if metadata is None:
        raise SystemExit(f"{run_dir} is not a finished run; train it first")
    return metadata


def best(group, arm, alpha, weights_dir, data_dir):
    runs = [r[3] for r in tn.group_runs(group, weights_dir, data_dir)
            if (r[0], r[1]) == (arm, alpha)]
    return min(runs, key=lambda r: finished(r)["val_op_loss"])


def select(name, weights_dir, data_dir):
    if name == "all":
        return list(dict.fromkeys(r[3] for g in tn.GROUPS
                                  for r in tn.group_runs(g, weights_dir, data_dir)))
    if name == "depth":
        return [best(g, *net, weights_dir, data_dir)
                for g in ("seeds", "depth_ablation") for net in (BASELINE, ARM)]
    if name == "arms":
        return [best("seeds", *net, weights_dir, data_dir)
                for net in (BASELINE, ARM)]
    if name == "transform_E":
        runs = [os.path.join(weights_dir, nlm.run_name(
                    mode, i, tn.config_digest(nlm.mode_config(data_dir, params))))
                for (mode, params) in nlm.NL_MODES.items() for i in nlm.SEEDS]
        return runs + [best("seeds", *BASELINE, weights_dir, data_dir)]
    if name == "transform_dE":
        runs = [os.path.join(weights_dir, nlm_de.run_name(
                    mode, alpha, i,
                    tn.config_digest(nlm_de.mode_config(data_dir, params))))
                for (mode, params) in nlm_de.NL_MODES.items()
                for alpha in nlm_de.ALPHAS for i in nlm_de.SEEDS]
        # the sweep's run at the same arm, alpha and seed, so the comparison
        # is paired
        reference = tn.run_name(nlm_de.ARM, nlm_de.ALPHAS[0], nlm_de.SEEDS[0],
                                tn.config_digest(tn.sweep_config(data_dir)))
        return runs + [os.path.join(weights_dir, reference)]
    raise ValueError(name)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("selection", choices=("all", "depth", "arms", "transform_E",
                                         "transform_dE"))
    args = p.parse_args(argv)
    paths = dl.load_paths("./paths.toml")
    runs = select(args.selection, paths["network_weights_dir"], paths["data_dir"])
    for run_dir in runs:
        finished(run_dir)
        print(run_dir)


if __name__ == "__main__":
    main()
