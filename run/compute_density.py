"""Bin a field's error against its own magnitude, and write the density store.

    python compute_density.py /weights/arm=no_deriv_*        # shell does the sweep
    python compute_density.py /weights/arm=no_deriv_* --limit 8
    python compute_density.py /weights/arm=1dir_alpha=0.1_nl=* --field dE

Runs are the positional arguments, as in `compute_error.py`, and the store is
written the same way: one group per run, appended, so an interrupted sweep
resumes and a rerun run replaces itself.

This is the loss-transform comparison's own store, not another dataset in
errors.h5 -- see `deeplight.evaluation.write_density`. Reading it back and
drawing it is `run/results_figure.py`, whose `density_figure` and the median
panel beside it are the only readers of this store.

Beside the counts, the exact median and mean |error| per |field| bin, taken
over the pixels here rather than read off the counts at plot time: either read
off the error axis can only be a bin centre, and those are 1.3x apart. It is
why this holds every scored pixel of a run in memory for the length of one run.

Both, because on an error this heavy-tailed they are different claims -- the
median is what a typical pixel does, the mean is what the total error is made
of, and the gap between them is the tail. A store written before the means
exist is rejected by `read_density`; rescore it.
"""
import argparse
import os
import time

import deeplight as dl


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+",
                   help="run directories (or checkpoint files) to score")
    p.add_argument("--paths", default="./paths.toml")
    p.add_argument("--criterion", default="operator",
                   choices=("operator", "total"),
                   help="which checkpoint to score inside a run directory")
    p.add_argument("--data", default=None,
                   help="test_data.h5 (default: paths.toml data_dir)")
    p.add_argument("--field", default="E", choices=("E", "dE"),
                   help="field to bin; dE costs one JVP per stored direction")
    p.add_argument("--out", default=None,
                   help="output .h5 (default: results_dir/density[_dE].h5)")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N rows of the test file")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-pixels", type=int, default=dl.MEDIAN_MIN_PIXELS,
                   help="pixels a |field| bin needs before it reports a median")
    p.add_argument("--tangent-groups", default=None,
                   help="dE only: merge tangent samplers into named splits, as "
                        "'ID=ID_1+SIGN,OOD=ID_2'. Every stored sampler must "
                        "appear. Merging has to happen here and not at plot "
                        "time -- a pooled median is taken over the pooled "
                        "pixels, not averaged from the separate medians")
    return p.parse_args(argv)


def parse_tangent_groups(spec):
    """`'ID=ID_1+SIGN,OOD=ID_2'` -> `{"ID": ("ID_1", "SIGN"), "OOD": ("ID_2",)}`."""
    if spec is None:
        return None
    groups = {}
    for part in spec.split(","):
        name, _, members = part.partition("=")
        if not members:
            raise SystemExit(f"--tangent-groups: {part!r} is not 'name=a+b'")
        groups[name.strip()] = tuple(m.strip() for m in members.split("+"))
    return groups


def main(argv=None):
    args = parse_args(argv)
    paths = dl.load_paths(args.paths)
    data_path = args.data or os.path.join(paths["data_dir"], "test_data.h5")
    # one file per field: the bin edges and the splits are the field's own
    default_out = "density.h5" if args.field == "E" else f"density_{args.field}.h5"
    out_path = args.out or os.path.join(paths["results_dir"], default_out)
    groups = parse_tangent_groups(args.tangent_groups)
    print(f"{len(args.runs)} run(s)\nfield  {args.field}\n"
          f"data   {data_path}\nout    {out_path}\n"
          + (f"groups {groups}\n" if groups else ""))

    for (i, path) in enumerate(args.runs, 1):
        start = time.time()
        run = dl.load_run(path, criterion=args.criterion)
        print(f"[{i}/{len(args.runs)}] {run['run_id']}", flush=True)
        density = dl.score_density(run, data_path, args.field,
                                   batch_size=args.batch_size,
                                   n_samples=args.limit,
                                   min_pixels=args.min_pixels,
                                   tangent_groups=groups)
        dl.write_density(out_path, run, density, data_path, args.field,
                         tangent_groups=groups)
        print(f"    {time.time() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
