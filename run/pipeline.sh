#!/usr/bin/env bash
#
# End-to-end DeepLight pipeline: dataset -> training -> analysis.
#
#   run/pipeline.sh                          # all stages
#   run/pipeline.sh --stages dataset         # just the dataset
#   run/pipeline.sh --stages train,analysis  # everything after it
#   run/pipeline.sh --dry-run                # report what would happen
#
# All locations come from run/paths.toml: the dataset goes to data_dir, the
# networks to network_weights_dir, the result stores to results_dir and the
# figures and tables to results_dir/figures.
#
# Splitting the work across machines
# ----------------------------------
# Each split is independent, so pick them off with --splits:
#
#   machine A (2 GPUs):  run/pipeline.sh --splits train --gpus 0,1
#   machine B:           run/pipeline.sh --splits val
#   afterwards:          run/pipeline.sh --splits test
#
# --gpus shards the split over the listed devices, one process each, and merges
# the shards when they finish. HDF5 has no safe concurrent-writer mode, so the
# shards are separate files; rows are seeded from their global index, so the
# merged result is byte-identical to an unsharded run. To drive the shards by
# hand (separate machines rather than separate GPUs on one), use --shards/--shard
# and merge at the end:
#
#   host1: run/pipeline.sh --splits train --shards 2 --shard 1
#   host2: run/pipeline.sh --splits train --shards 2 --shard 2
#   then:  python run/merge_shards.py --data-dir DIR --splits train
#
# The dataset stage is idempotent. Each split is inspected first:
#
#   complete  -> skipped, nothing runs
#   partial   -> both generation stages re-run; each resumes at the first
#                unfinished row, so interrupting is free
#   missing   -> generated from scratch
#   mismatch  -> ERROR. A file whose row count differs from the configured size
#                is not touched: fixing it means deleting data, which is the
#                caller's decision to make, not this script's.
#
# Sizes, ray counts and direction counts are deliberately *not* set here. They
# live in tissue_prior.DATASET_SPECS and generate_light_transport.jl's
# DEFAULT_RAYS, and passing them again would create a second source of truth
# that silently drifts.
#
# Training
# --------
# Trains every run group of the paper: the run groups of train_network.py
# (sweep, seeds, size, depth_ablation) and the loss-transform studies
# (train_nl_modes.py, train_nl_modes_dE.py). Finished runs are skipped, so an
# interrupted stage resumes. It runs on the default device; set
# CUDA_VISIBLE_DEVICES to pick one (--gpus applies to the dataset stage only).
#
# Analysis
# --------
# Scores the trained networks into the result stores (compute_error.py,
# compute_density.py) and draws the figures and tables from them. Which runs
# go into which store is decided by select_runs.py; every one has to be
# finished. Scoring is not incremental: the stage rescores every run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGES="dataset,train,analysis"
DRY_RUN=0
SPLITS=""          # empty = every split in DATASET_SPECS
GPUS=""            # e.g. "0,1": shard over these devices and merge
SHARDS=1           # manual sharding, for splitting across machines
SHARD=""

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages)   STAGES="$2";   shift 2 ;;
        --splits)   SPLITS="$2";   shift 2 ;;
        --gpus)     GPUS="$2";     shift 2 ;;
        --shards)   SHARDS="$2";   shift 2 ;;
        --shard)    SHARD="$2";    shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift ;;
        -h|--help)  usage ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

if [[ -n "$GPUS" && ( "$SHARDS" != 1 || -n "$SHARD" ) ]]; then
    echo "--gpus manages sharding itself; do not combine it with --shards/--shard" >&2
    exit 64
fi
if [[ "$SHARDS" != 1 && -z "$SHARD" ]]; then
    echo "--shards > 1 requires --shard (or use --gpus to run them here)" >&2
    exit 64
fi

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

path_of() {
    python - "$1" <<'PY'
import sys
import tomli
with open("run/paths.toml", "rb") as f:
    print(tomli.load(f)["paths"][sys.argv[1]])
PY
}
DATA_DIR="$(path_of data_dir)"
RESULTS_DIR="$(path_of results_dir)"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

# Splits, read from the same place the generator reads them, so this cannot
# disagree with what gets written.
ALL_SPLITS="$(python - <<'PY'
import sys
sys.path.insert(0, "src/datagen")
import tissue_prior as tp
print(" ".join(tp.DATASET_SPECS))
PY
)"
if [[ -n "$SPLITS" ]]; then
    read -r -a SPLIT_LIST <<<"${SPLITS//,/ }"
    for s in "${SPLIT_LIST[@]}"; do
        [[ " $ALL_SPLITS " == *" $s "* ]] || {
            echo "unknown split '$s' (have: $ALL_SPLITS)" >&2; exit 64; }
    done
else
    read -r -a SPLIT_LIST <<<"$ALL_SPLITS"
fi

# missing | partial | complete | mismatch:<rows>/<expected>
split_status() {
    python - "$DATA_DIR" "$1" <<'PY'
import pathlib
import sys

sys.path.insert(0, "src/datagen")
import numpy as np
import h5py
import tissue_prior as tp

data_dir, split = sys.argv[1], sys.argv[2]
n_per_spec, directions_per_sample, specs = tp.DATASET_SPECS[split]
expected = n_per_spec * len(specs)

path = pathlib.Path(data_dir) / f"{split}_data.h5"
if not path.exists():
    print("missing")
    raise SystemExit

with h5py.File(path) as f:
    rows = f["mu"].shape[0]
    if rows != expected:
        print(f"mismatch:{rows}/{expected}")
        raise SystemExit
    # `done` marks written tissue rows, `light_transport_done` simulated ones.
    # Both must be complete: a file can carry every tissue row and no transport
    # at all, and its E/dE would then be preallocated zeros -- which look like
    # perfectly valid zero-energy labels to anything that does not check.
    tissue_done = np.asarray(f["done"][()]).astype(bool).all()
    lt_done = ("light_transport_done" in f
               and np.asarray(f["light_transport_done"][()]).astype(bool).all())

print("complete" if (tissue_done and lt_done) else "partial")
PY
}

# One shard of one split: tissue then transport, both resuming per row.
# `dev` is the CUDA device to pin the transport to, or "" for the default.
generate_shard() {
    local split="$1" shards="$2" shard="$3" dev="$4"
    local sh_args=()
    [[ "$shards" == 1 ]] || sh_args=(--shards "$shards" --shard "$shard")
    python run/generate_tissue.py --data-dir "$DATA_DIR" --splits "$split" \
        "${sh_args[@]}"
    # Only pin the device when one was actually named. Exporting an *empty*
    # CUDA_VISIBLE_DEVICES does not mean "use the default" -- it hides every GPU,
    # and the run dies with CUDA_ERROR_NO_DEVICE at the first allocation.
    if [[ -n "$dev" ]]; then
        CUDA_VISIBLE_DEVICES="$dev" julia --project=run \
            run/generate_light_transport.jl --data-dir "$DATA_DIR" \
            --splits "$split" "${sh_args[@]}"
    else
        julia --project=run \
            run/generate_light_transport.jl --data-dir "$DATA_DIR" \
            --splits "$split" "${sh_args[@]}"
    fi
}

run_dataset() {
    echo "=== dataset -> $DATA_DIR"
    local todo=()

    for split in "${SPLIT_LIST[@]}"; do
        local status
        status="$(split_status "$split")"
        case "$status" in
            complete) echo "  $split: complete, skipping" ;;
            missing)  echo "  $split: missing, will generate";  todo+=("$split") ;;
            partial)  echo "  $split: partial, will resume";    todo+=("$split") ;;
            mismatch:*)
                echo "  $split: ERROR row count ${status#mismatch:} (have/expected)" >&2
                echo "  Refusing to touch it. Move or delete the file, or change" >&2
                echo "  DATASET_SPECS in src/datagen/tissue_prior.py to match." >&2
                exit 65
                ;;
            *) echo "  $split: unexpected status '$status'" >&2; exit 70 ;;
        esac
    done

    if [[ ${#todo[@]} -eq 0 ]]; then
        echo "  nothing to do"
        return
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would generate: ${todo[*]}"
        return
    fi

    for split in "${todo[@]}"; do
        if [[ -n "$GPUS" ]]; then
            # One process per device, each writing its own shard file, then a
            # merge. Concurrent writers to a single HDF5 file are not safe
            # without parallel HDF5, which is why this cannot just be threads.
            local devs; read -r -a devs <<<"${GPUS//,/ }"
            local n=${#devs[@]} pids=()
            echo "--- $split: ${n} shards over GPU(s) ${GPUS}"
            for ((i = 0; i < n; i++)); do
                generate_shard "$split" "$n" "$((i + 1))" "${devs[i]}" \
                    > "$DATA_DIR/${split}.shard$((i + 1)).log" 2>&1 &
                pids+=($!)
                echo "    shard $((i + 1)) on GPU ${devs[i]} (pid ${pids[-1]}, "\
"log $DATA_DIR/${split}.shard$((i + 1)).log)"
            done
            local failed=0
            for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
            [[ $failed -eq 0 ]] || {
                echo "  $split: a shard failed; see the logs above" >&2; exit 70; }
            python run/merge_shards.py --data-dir "$DATA_DIR" --splits "$split"
        elif [[ "$SHARDS" != 1 ]]; then
            echo "--- $split: shard $SHARD of $SHARDS"
            generate_shard "$split" "$SHARDS" "$SHARD" "${CUDA_VISIBLE_DEVICES:-}"
            echo "  shard written. Run the other shards, then:"
            echo "    python run/merge_shards.py --data-dir $DATA_DIR --splits $split"
            return
        else
            echo "--- $split"
            generate_shard "$split" 1 1 "${CUDA_VISIBLE_DEVICES:-}"
        fi
    done

    for split in "${todo[@]}"; do
        local status
        status="$(split_status "$split")"
        [[ "$status" == "complete" ]] || {
            echo "  $split: still '$status' after generation" >&2
            exit 70
        }
    done
    echo "  dataset complete"
}

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

# Runs a python script in run/, where the scripts find paths.toml.
py() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] python $*"
    else
        (cd run && python "$@")
    fi
}

run_train() {
    echo "=== train"
    for group in sweep seeds size depth_ablation; do
        echo "--- $group"
        py train_network.py "$group"
    done
    echo "--- loss transform, operator"
    py train_nl_modes.py
    echo "--- loss transform, derivative"
    py train_nl_modes_dE.py
}

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

# Run directories of one select_runs.py selection, into the array SELECTED.
select_runs() {
    local out
    out="$(cd run && python select_runs.py "$1")"
    mapfile -t SELECTED <<<"$out"
}

run_analysis() {
    echo "=== analysis -> $RESULTS_DIR"
    local tangents="ID=ID_1+SIGN,OOD=ID_2"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would score the runs of select_runs.py into"
        echo "    errors.h5, errors_depth_ext.h5, density.h5,"
        echo "    density_transform_dE.h5, density_arms.h5, density_arms_dE.h5"
        echo "  and draw the figures and tables into $RESULTS_DIR/figures"
        return
    fi

    echo "--- errors"
    select_runs all
    py compute_error.py "${SELECTED[@]}"
    py compute_error.py "${SELECTED[@]}" --criterion total

    echo "--- errors by depth, depth-channel ablation"
    select_runs depth
    py compute_error.py "${SELECTED[@]}" --depth-above 20 \
        --out "$RESULTS_DIR/errors_depth_ext.h5"

    echo "--- loss transform"
    select_runs transform_E
    py compute_density.py "${SELECTED[@]}" --out "$RESULTS_DIR/density.h5"
    select_runs transform_dE
    py compute_density.py "${SELECTED[@]}" --field dE \
        --tangent-groups "$tangents" \
        --out "$RESULTS_DIR/density_transform_dE.h5"

    echo "--- baseline against DeepLight"
    select_runs arms
    py compute_density.py "${SELECTED[@]}" --out "$RESULTS_DIR/density_arms.h5"
    py compute_density.py "${SELECTED[@]}" --field dE \
        --tangent-groups "$tangents" --out "$RESULTS_DIR/density_arms_dE.h5"

    echo "--- figures and tables"
    py nl_plot.py
    py samples_figure.py
    py results_figure.py
    py network_performance_table.py
    py qualitative_samples.py
}

# --------------------------------------------------------------------------

if has_stage dataset;  then run_dataset;  fi
if has_stage train;    then run_train;    fi
if has_stage analysis; then run_analysis; fi
