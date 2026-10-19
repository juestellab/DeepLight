"""The "Network Performance" tables of the paper (tab:res / tab:res_metrics).

    python network_performance_table.py                  # csv + tex macros, to results_dir/figures

One row per network configuration in the paper's tables: the Sobolev/depth
ablations (`no_deriv`, at the depth-ablation digest and the main sweep) and
the three derivative arms at the main sweep, plus the depth-ablated `4 Deriv`
counterpart. Each row is the single best seed (by `val_op_loss`, across alpha
too for the swept arms) of its configuration -- one row is one trained
network, same as `results_figure.per_run`.

Each row carries the operator error (`field="E"`) and the derivative error
(`field="dE"`), both broken out by row split (ID/OOD,
`results_figure.SPLIT_LABELS`). The derivative error is broken out further by
which tangent sampler it was taken along, but only the sampler pairing that
matches the row split's own distribution (`SAMPLERS_FOR_SPLIT`, same pairing
as `results_figure.TANGENT_GROUPS`): ID_1 and SIGN for ID rows, ID_2 for OOD
rows. The other two (row split, sampler) combinations exist in the store but
are the tangent family evaluated off its own distribution, which is not part
of this comparison. All of it is `rel_err` (`nl_train` transform) as a
percentage, median over the relevant samples of that one network.

`transform_table` is a second, narrower table: just the two networks the
paper's main comparison turns on (`no_deriv` and the settled `2_per_sampler`
arm, both with the depth channel), scored under the `linear` and `log`
transforms instead of `nl_train` -- i.e. the operator and derivative error as
they would read on an untransformed and a log-compressed scale, rather than
the scale each network was actually trained against. Same operator/derivative
and sampler-pairing columns as `table`.

Writes a CSV (for inspection) and a `.tex` file of `\\newcommand` macros that
`DeepLight.tex` can `\\input` directly, so the table cells never need hand
copying numbers out of a CSV.
"""
import argparse
import os

import pandas as pd

import deeplight as dl
import results_figure as rf
import train_network as tn

# (row label, macro-safe name, selection). `config_digest` pins the row to one
# training configuration; depth_channel and direction_arm say which ablation or
# arm it is. `results_figure.SWEEP` is the main sweep; this is the depth-channel
# ablation, trained with DEPTH_CHANNEL=False.
DEPTH_ABLATION_DIGEST = tn.config_digest(
    tn.sweep_config(rf._DATA_DIR, "depth_ablation"))

ROWS = (
    ("No Sobolev, no depth", "NoSobolevNoDepth",
     dict(direction_arm="no_deriv", depth_channel=False,
          config_digest=DEPTH_ABLATION_DIGEST)),
    ("No Sobolev + depth", "NoSobolevDepth",
     dict(direction_arm="no_deriv", depth_channel=True,
          config_digest=rf.SWEEP)),
    ("1 Deriv", "OneDeriv",
     dict(direction_arm="1dir", depth_channel=True, config_digest=rf.SWEEP)),
    ("2 Deriv", "TwoDeriv",
     dict(direction_arm="1_per_sampler", depth_channel=True,
          config_digest=rf.SWEEP)),
    ("4 Deriv", "FourDeriv",
     dict(direction_arm="2_per_sampler", depth_channel=True,
          config_digest=rf.SWEEP)),
    ("4 Deriv - depth", "FourDerivNoDepth",
     dict(direction_arm="2_per_sampler", depth_channel=False,
          config_digest=DEPTH_ABLATION_DIGEST)),
)

# Derivative columns: one per (row split, tangent sampler), restricted to the
# sampler(s) drawn from that row split's own distribution -- the same pairing
# `results_figure.TANGENT_GROUPS` pools for the density figures. The macro-safe
# names are combined as f"{split}{SAMPLER_MACRO[sampler]}", e.g. "IDSign" or
# "OODIDTwo". LaTeX macro names may only contain letters, so the samplers'
# digits are spelled out.
SAMPLER_MACRO = {"ID_1": "IDOne", "SIGN": "Sign", "ID_2": "IDTwo"}
SPLITS = ("ID", "OOD")
SAMPLERS_FOR_SPLIT = {"ID": ("ID_1", "SIGN"), "OOD": ("ID_2",)}


def table(path):
    """One row per `ROWS` entry: label, macro name, run_id, operator and dE metrics."""
    df = dl.read_errors(path, transform="nl_train")
    op = df[df["field"] == "E"]
    deriv = df[df["field"] == "dE"]
    rows = []
    for (label, macro, spec) in ROWS:
        sub = rf.pick(op, **rf.OPERATOR, **spec)
        if sub.empty:
            raise LookupError(f"{label!r}: no run matches {spec}")
        run_id = rf.best_seed(sub)
        cell = sub[sub["run_id"] == run_id]
        row = {"label": label, "macro": macro, "run_id": run_id}
        for split in SPLITS:
            scell = cell[cell["split"] == split]
            for metric in ("rmse", "mae"):
                row[f"op_{split}_{metric}"] = (scell.loc[scell["metric"] == metric, "rel_err"].median()*1000)

        dcell = deriv[deriv["run_id"] == run_id]
        for split in SPLITS:
            for sampler in SAMPLERS_FOR_SPLIT[split]:
                dsub = dcell[(dcell["split"] == split)
                             & (dcell["sampler"] == sampler)]
                key = f"{split}{SAMPLER_MACRO[sampler]}"
                for metric in ("rmse", "mae"):
                    row[f"dE_{key}_{metric}"] = (dsub.loc[dsub["metric"] == metric, "rel_err"].median()*1000)
        rows.append(row)
    return pd.DataFrame(rows)


TRANSFORMS = ("linear", "log")

# The two networks `transform_table` compares -- labels into `table`'s output,
# so the run picked there (by val_op_loss) is the run scored here too.
TRANSFORM_LABELS = ("No Sobolev + depth", "4 Deriv")


def transform_table(path, main_rows, labels=TRANSFORM_LABELS):
    """Operator and derivative error of `labels`' runs under `linear`/`log`.

    Takes `run_id` from `main_rows` (`table`'s output) rather than re-selecting
    a best seed, so this table can never compare a different pair of networks
    than the main one does. Derivative columns use the same `SAMPLERS_FOR_SPLIT`
    pairing as `table`.

    `src.evaluation.TRANSFORMS` only defines `log` for `E`: `dE` is a signed
    directional derivative, for which a plain log has no well-defined meaning,
    so the store never scored it under `log`. Those cells come back as `None`
    rather than a selection that happens to match nothing, so the gap reads as
    "not computed" rather than a silent NaN.
    """
    df = dl.read_errors(path)
    op = df[df["field"] == "E"]
    deriv = df[df["field"] == "dE"]
    deriv_transforms = set(deriv["transform"].unique())
    rows = []
    for label in labels:
        picked = main_rows.loc[main_rows["label"] == label].iloc[0]
        cell = op[op["run_id"] == picked["run_id"]]
        dcell = deriv[deriv["run_id"] == picked["run_id"]]
        row = {"label": label, "macro": picked["macro"], "run_id": picked["run_id"]}
        for transform in TRANSFORMS:
            for split in SPLITS:
                scell = cell[(cell["transform"] == transform)
                             & (cell["split"] == split)]
                for metric in ("rmse", "mae"):
                    row[f"op_{transform}_{split}_{metric}"] = (
                        scell.loc[scell["metric"] == metric, "rel_err"].median()
                        * 100)
                for sampler in SAMPLERS_FOR_SPLIT[split]:
                    key = f"{split}{SAMPLER_MACRO[sampler]}"
                    if transform not in deriv_transforms:
                        for metric in ("rmse", "mae"):
                            row[f"dE_{transform}_{key}_{metric}"] = None
                        continue
                    dsub = dcell[(dcell["transform"] == transform)
                                 & (dcell["split"] == split)
                                 & (dcell["sampler"] == sampler)]
                    for metric in ("rmse", "mae"):
                        row[f"dE_{transform}_{key}_{metric}"] = (
                            dsub.loc[dsub["metric"] == metric, "rel_err"].median()
                            * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value):
    """`"--"` for a cell the store never scored (`None`/NaN), else 2 decimals."""
    return "--" if value is None or pd.isna(value) else f"{value:.2f}"


def write_transform_macros(rows, path):
    with open(path, "a") as f:
        for row in rows.itertuples():
            for transform in TRANSFORMS:
                cap = transform.capitalize()
                for split in SPLITS:
                    for metric in ("rmse", "mae"):
                        value = getattr(row, f"op_{transform}_{split}_{metric}")
                        name = f"\\NetPerf{row.macro}{cap}Op{split}{metric.capitalize()}"
                        f.write(f"\\newcommand{{{name}}}{{{_fmt(value)}}}\n")
                    for sampler in SAMPLERS_FOR_SPLIT[split]:
                        key = f"{split}{SAMPLER_MACRO[sampler]}"
                        for metric in ("rmse", "mae"):
                            value = getattr(row, f"dE_{transform}_{key}_{metric}")
                            name = (f"\\NetPerf{row.macro}{cap}Deriv{key}"
                                    f"{metric.capitalize()}")
                            f.write(f"\\newcommand{{{name}}}{{{_fmt(value)}}}\n")


def write_macros(rows, path):
    with open(path, "w") as f:
        f.write("% Generated by run/network_performance_table.py -- do not edit.\n")
        for row in rows.itertuples():
            for split in SPLITS:
                rmse = getattr(row, f"op_{split}_rmse")
                mae = getattr(row, f"op_{split}_mae")
                f.write(f"\\newcommand{{\\NetPerf{row.macro}Op{split}Rmse}}{{{rmse:.2f}}}\n")
                f.write(f"\\newcommand{{\\NetPerf{row.macro}Op{split}Mae}}{{{mae:.2f}}}\n")
            for split in SPLITS:
                for sampler in SAMPLERS_FOR_SPLIT[split]:
                    key = f"{split}{SAMPLER_MACRO[sampler]}"
                    rmse = getattr(row, f"dE_{key}_rmse")
                    mae = getattr(row, f"dE_{key}_mae")
                    f.write(f"\\newcommand{{\\NetPerf{row.macro}Deriv{key}Rmse}}{{{rmse:.2f}}}\n")
                    f.write(f"\\newcommand{{\\NetPerf{row.macro}Deriv{key}Mae}}{{{mae:.2f}}}\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="output directory (default: results_dir/figures)")
    p.add_argument("--errors", default=None,
                   help="error .h5 (default: results_dir/errors.h5)")
    args = p.parse_args(argv)

    paths = dl.load_paths("./paths.toml")
    errors = args.errors or os.path.join(paths["results_dir"], "errors.h5")
    args.out = args.out or os.path.join(paths["results_dir"], "figures")
    os.makedirs(args.out, exist_ok=True)

    rows = table(errors)
    csv_path = os.path.join(args.out, "network_performance.csv")
    rows.to_csv(csv_path, index=False)
    tex_path = os.path.join(args.out, "network_performance_values.tex")
    write_macros(rows, tex_path)
    print(f"  wrote {csv_path}  ({rows.shape[0]} rows)")

    trows = transform_table(errors, rows)
    tcsv_path = os.path.join(args.out, "transform_performance.csv")
    trows.to_csv(tcsv_path, index=False)
    write_transform_macros(trows, tex_path)
    print(f"  wrote {tcsv_path}  ({trows.shape[0]} rows)")
    print(f"  wrote {tex_path}")


if __name__ == "__main__":
    main()
