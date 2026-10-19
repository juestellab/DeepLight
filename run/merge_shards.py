"""Concatenate the shards of a split into one `<split>_data.h5`.

HDF5 has no safe concurrent-writer mode without parallel HDF5, so a split
generated on several GPUs or machines is written as separate shard files and
joined afterwards. Because `compute_dataset` seeds every row from its *global*
index, the concatenation in shard order is byte-identical to what a single
unsharded run would have produced -- so sharding is purely a scheduling choice.

    python run/merge_shards.py --data-dir DIR --splits train
    python run/merge_shards.py --data-dir DIR --splits train --keep

Refuses to merge unless every shard is present, complete (tissue *and* light
transport), and tiles the split's rows exactly once. A merged file whose rows
were silently left as preallocated zeros would look like valid zero-energy
labels, so partial input is an error rather than a warning.
"""
import argparse
import pathlib
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "datagen"))

import tissue_prior as tp  # noqa: E402

SHARD_RE = re.compile(r"^(?P<split>.+)_data\.shard(?P<i>\d+)of(?P<n>\d+)\.h5$")
# per-row datasets, concatenated along axis 0; everything else is an attribute
# or a per-file constant and is copied from the first shard
ROW_DATASETS = ("masks", "g", "mu", "v", "sampler_index", "done",
                "light_transport_done")


def find_shards(data_dir, split):
    """Sorted [(index, path)] for `split`, validated to tile it exactly once."""
    found = {}
    n_shards = None
    for path in sorted(pathlib.Path(data_dir).glob(f"{split}_data.shard*.h5")):
        m = SHARD_RE.match(path.name)
        if not m or m.group("split") != split:
            continue
        i, n = int(m.group("i")), int(m.group("n"))
        if n_shards is not None and n != n_shards:
            raise SystemExit(f"{split}: shards disagree on the total "
                             f"({n_shards} vs {n}); remove the stale ones")
        n_shards = n
        found[i] = path
    if n_shards is None:
        raise SystemExit(f"{split}: no shard files in {data_dir}")
    missing = sorted(set(range(1, n_shards + 1)) - set(found))
    if missing:
        raise SystemExit(f"{split}: missing shard(s) {missing} of {n_shards}")
    return [(i, found[i]) for i in range(1, n_shards + 1)]


def check(shards, split):
    """Validate completeness and row coverage; return the total row count."""
    n_per_spec, directions_per_tangent, specs = tp.DATASET_SPECS[split]
    expected = n_per_spec * len(specs)
    offset = 0
    for i, path in shards:
        with h5py.File(path) as f:
            rows = f["mu"].shape[0]
            got = int(f.attrs.get("row_offset", 0))
            if got != offset:
                raise SystemExit(f"{path.name}: row_offset {got}, expected "
                                 f"{offset} -- shards do not tile the split")
            if not np.asarray(f["done"][()]).astype(bool).all():
                raise SystemExit(f"{path.name}: tissue generation incomplete")
            if "light_transport_done" not in f:
                raise SystemExit(f"{path.name}: no light transport results")
            if not np.asarray(f["light_transport_done"][()]).astype(bool).all():
                n_done = int(np.asarray(f["light_transport_done"][()]).sum())
                raise SystemExit(f"{path.name}: light transport incomplete "
                                 f"({n_done}/{rows} rows)")
        offset += rows
    if offset != expected:
        raise SystemExit(f"{split}: shards cover {offset} rows, but "
                         f"DATASET_SPECS says {expected}")
    return offset


def merge(data_dir, split, keep=False):
    shards = find_shards(data_dir, split)
    total = check(shards, split)
    out = pathlib.Path(data_dir) / f"{split}_data.h5"
    if out.exists():
        raise SystemExit(f"{out} already exists; move it aside first")

    print(f"{split}: merging {len(shards)} shards, {total} rows -> {out.name}")
    tmp = out.with_suffix(".h5.partial")
    with h5py.File(shards[0][1]) as first, h5py.File(tmp, "w") as dst:
        for key, value in first.attrs.items():
            dst.attrs[key] = value
        # row_offset described the shard, and would be a lie on the merged file
        for key in ("row_offset", "n_rows_total"):
            dst.attrs.pop(key, None)

        for name in first:
            if name in ROW_DATASETS:
                shape = (total,) + first[name].shape[1:]
                dst.create_dataset(name, shape, dtype=first[name].dtype)
            elif isinstance(first[name], h5py.Group):      # E, dE
                g = dst.create_group(name)
                for dist in first[name]:
                    d = first[name][dist]
                    # Julia writes these as (depth, lateral, K, N), but h5py sees
                    # the axes reversed, so from here the row axis leads exactly
                    # as it does for mu and v.
                    g.create_dataset(dist, (total,) + d.shape[1:], dtype=d.dtype)
            else:
                dst.create_dataset(name, data=first[name][()])

        offset = 0
        for i, path in shards:
            with h5py.File(path) as src:
                n = src["mu"].shape[0]
                for name in ROW_DATASETS:
                    if name in src:
                        dst[name][offset:offset + n] = src[name][()]
                for name in src:
                    if isinstance(src[name], h5py.Group):
                        for dist in src[name]:
                            dst[name][dist][offset:offset + n] = \
                                src[name][dist][()]
                offset += n
            print(f"  shard {i}/{len(shards)}: {n} rows")

    tmp.rename(out)
    print(f"  wrote {out}")
    if not keep:
        for _, path in shards:
            path.unlink()
        print(f"  removed {len(shards)} shard files")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", default=",".join(tp.DATASET_SPECS))
    p.add_argument("--keep", action="store_true",
                   help="keep the shard files instead of deleting them")
    args = p.parse_args(argv)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        merge(args.data_dir, split, keep=args.keep)


if __name__ == "__main__":
    main()
