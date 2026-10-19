"""Stage 1 of dataset generation: sample the tissue optical properties.

Writes one HDF5 file per split (`train_data.h5`, `val_data.h5`, `test_data.h5`)
containing the primal fields, the derivative directions and the sampler
provenance -- everything except the light transport, which is stage 2
(`generate_light_transport.jl`, run afterwards on the same files).

    python run/generate_tissue.py                       # all splits, paths.toml
    python run/generate_tissue.py --splits train        # one split
    python run/generate_tissue.py --n-samples 8 --data-dir /tmp/smoke

Interrupting is safe: every row is flagged as it is written and a re-run picks
up the unwritten ones. Rows are keyed by index, so a resumed file is identical
to one written in a single pass.

Every sample carries the directions of *all* tangent samplers at the same primal
point (2 each by default: ID_1, ID_2, SIGN -> K=6), so the tangent samplers are
balanced within every single sample and share one ray ensemble downstream.

File layout (N = n_rows_per_spec * n_row_specs, K = n_tangents * per_tangent):

    masks                    (N, 4, X, Z)   coupling/epidermis/dermis/tissue
    g                        (N, 1, X, Z)   scattering anisotropy
    mu                       (N, 2, X, Z)   absorption, scattering
    v                        (N, K, 2, X, Z)  K derivative directions per sample
    sampler_index            (N,)           which primal sampler made each row
    direction_sampler_index  (K,)           which tangent sampler made each direction
    done                     (N,)           resume marker
    attrs: sampler_names, mu_samplers, direction_sampler_names,
           directions_per_tangent, n_directions, split, seed, width_cm,
           depth_cm, pixel_size_cm, mean_skin_depth_cm, git_commit

`mean_skin_depth_cm` is where the sampled skin line sits on average; the
transport stage reads it to aim the tilted beam at the tissue surface rather
than at the top of the grid.
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "datagen"))

import tomli  # noqa: E402
import tissue_prior as tp  # noqa: E402


def shard_rows(n_total, n_shards, shard):
    """(row_offset, n_rows) of `shard` (1-based) when splitting `n_total` rows.

    The remainder is spread over the leading shards rather than dumped on the
    last one, so shard sizes differ by at most one row and the machines finish
    together.
    """
    base, extra = divmod(n_total, n_shards)
    i = shard - 1
    n_rows = base + (1 if i < extra else 0)
    offset = i * base + min(i, extra)
    return offset, n_rows


# Deliberately duplicated from deeplight.utils rather than imported: this
# script only needs numpy/h5py (it reaches straight into src/datagen above),
# and `import deeplight` would drag in jax, flax and pandas on the data
# generation nodes just for this one line.
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=pathlib.Path(__file__).parent,
                                       text=True).strip()
    except Exception:
        return "unknown"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None,
                   help="name of the output file (default: {split}_data.h5)")
    p.add_argument("--data-dir", default=None,
                   help="output directory (default: data_dir from paths.toml)")
    p.add_argument("--paths", default=str(pathlib.Path(__file__).parent / "paths.toml"))
    p.add_argument("--splits", default=",".join(tp.DATASET_SPECS),
                   help=f"comma separated, from {', '.join(tp.DATASET_SPECS)}")
    p.add_argument("--n-samples", type=int, default=None,
                   help="rows per row spec; default is the per-split size in "
                        "tissue_prior.DATASET_SPECS")
    p.add_argument("--seed", type=int, default=tp.DEFAULT_SEED)
    p.add_argument("--shards", type=int, default=1,
                   help="split each file into this many shards, so they can be "
                        "generated on separate machines or GPUs. HDF5 has no "
                        "safe concurrent-writer mode, so shards are separate "
                        "files, merged afterwards by run/merge_shards.py")
    p.add_argument("--shard", type=int, default=None,
                   help="which shard to write, 1-based. Required when --shards "
                        "> 1; the output is <split>_data.shard<i>of<n>.h5")
    p.add_argument("--no-resume", action="store_true",
                   help="fail instead of continuing an existing file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    data_dir = args.data_dir
    if data_dir is None:
        with open(args.paths, "rb") as f:
            data_dir = tomli.load(f)["paths"]["data_dir"]
    out = pathlib.Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - set(tp.DATASET_SPECS)
    if unknown:
        raise SystemExit(f"unknown split(s): {', '.join(sorted(unknown))}")

    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")
    if args.shards > 1 and args.shard is None:
        raise SystemExit("--shards > 1 requires --shard")
    if args.shard is not None and not 1 <= args.shard <= args.shards:
        raise SystemExit(f"--shard must be in 1..{args.shards}")

    attrs = {"git_commit": git_commit()}
    for split in splits:
        n_per_spec, directions_per_tangent, specs = tp.DATASET_SPECS[split]
        if args.n_samples is not None:
            n_per_spec = args.n_samples
        n_directions = len(specs[0].v_samplers) * directions_per_tangent
        offset, n_rows = shard_rows(n_per_spec * len(specs),
                                    args.shards, args.shard or 1)
        prefix = split if args.name is None else args.name
        name = (f"{prefix}_data.h5" if args.shards == 1 else
                f"{prefix}_data.shard{args.shard}of{args.shards}.h5")
        print(f"\n=== {split}: {n_per_spec} x {len(specs)} row spec(s) "
              f"({', '.join(s.name for s in specs)}), "
              f"{n_directions} directions each "
              f"({directions_per_tangent} x {', '.join(tp.TANGENT_NAMES)})"
              + (f", rows {offset}..{offset + n_rows - 1}"
                 if args.shards > 1 else "")
              + f" -> {out}/{name}")
        tp.compute_dataset(tp.split_key(args.seed, split), n_per_spec, specs,
                           str(out / name),
                           directions_per_tangent=directions_per_tangent,
                           resume=not args.no_resume,
                           attrs={"split": split, "seed": args.seed, **attrs},
                           row_offset=offset, n_rows=n_rows)


if __name__ == "__main__":
    main()
