"""Score trained networks on the test set and write the error store.

    python compute_error.py /results/arm=*                  # shell does the sweep
    python compute_error.py /results/* --limit 8            # cheap smoke run

Runs are the positional arguments, as run directories. Nothing is discovered
and no "best run" is picked here; that decision belongs to the figures, which
make it from the store. Passing several paths at once is worth it: they share
one read of the test file.

One group per run, appended, so an interrupted sweep resumes and a rerun run
replaces itself. Only errors are stored, in the test file's own row order;
splits and gains are derived at plot time (see deeplight.results).
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
    p.add_argument("--out", default=None,
                   help="output .h5 (default: paths.toml results_dir/errors.h5)")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N rows of the test file")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--depth-above", type=int, default=0,
                   help="pixels of depth profile to score above the skin "
                        "line, into the coupling medium (0.1mm/px)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = dl.load_paths(args.paths)
    data_path = args.data or os.path.join(paths["data_dir"], "test_data.h5")
    out_path = args.out or os.path.join(paths["results_dir"], "errors.h5")
    print(f"{len(args.runs)} run(s)\ndata   {data_path}\nout    {out_path}\n")

    for (i, path) in enumerate(args.runs, 1):
        start = time.time()
        run = dl.load_run(path, criterion=args.criterion)
        print(f"[{i}/{len(args.runs)}] {run['run_id']}", flush=True)
        scores = dl.score_run(run, data_path, batch_size=args.batch_size,
                              n_samples=args.limit,
                              depth_above=args.depth_above)
        dl.write_run(out_path, run, scores, data_path)
        print(f"    {time.time() - start:.1f} s", flush=True)

    print(f"\n{len(dl.run_table(out_path))} run(s) in {out_path}")


if __name__ == "__main__":
    main()
