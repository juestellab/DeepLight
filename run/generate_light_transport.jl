#=
Stage 2 of dataset generation: Monte Carlo light transport.

Runs on the HDF5 files written by `run/generate_tissue.py` and appends the
absorbed energy `E` and its directional derivatives `dE` to each of them. All
`K` derivative directions of a sample share one ray ensemble, so the cost is
close to that of a plain energy simulation.

    julia --project run/generate_light_transport.jl
    julia --project run/generate_light_transport.jl --splits train --rays 5e8
    julia --project run/generate_light_transport.jl --file /path/to/train_data.h5

Interrupting is safe: finished samples are flagged in `light_transport_done` and
a re-run continues with the rest. Per-sample timing and an ETA go to stdout and
to `<file>.log`.
=#
using TOML
using DeepLight

# Ray counts per split. The validation and test sets are smaller but are what
# models are compared on, so they get the quieter simulation.
#
# Measured on 10 phantoms from 5 independent replicates (scratchpad
# noise_vs_depth.py; the 1/sqrt(N) law verified to within 3%), the whole-image
# per-direction Monte Carlo noise in loss units at c_dE=1e-2 is:
#
#     rays     dE      E
#     1.5e8   5.8%   0.27%     <- train
#     9e8     2.4%   0.11%     <- val/test
#
# dE carries essentially all of it; E is exact for these purposes. The noise is
# far from uniform over the image: ~3-4% at the skin line, ~10-11% in the
# coupling layer and the deep tail, where |dE| is 25x smaller. Raising the ray
# count only buys 1/sqrt(N) and cannot reach the tails, which is why 1.5e8 is
# the deliberate choice for training rather than a budget compromise.
#
# Val and test get 6x the training rays because they are what model error is
# measured against, and label noise adds to that error in quadrature: at 2.4%
# against a ~4% model error the measurement is inflated by ~16%, where 4.5e8
# would inflate it by ~30%. Going further has poor returns -- halving the
# inflation again costs 4x the rays.
const DEFAULT_RAYS = Dict("train" => 150_000_000,
                          "val"   => 900_000_000,
                          "test"  => 900_000_000)
const DEFAULT_SPLITS = ["train", "val", "test"]


function parse_args(args)
    opts = Dict{String,Any}("splits" => DEFAULT_SPLITS, "files" => String[],
                            "rays" => nothing, "resume" => true,
                            "paths" => joinpath(@__DIR__, "paths.toml"),
                            "data_dir" => nothing,
                            "dists" => DEFAULT_DISTANCES,
                            "store_type" => Float32,
                            "oop_halfwidth" => nothing,
                            "max_samples" => nothing,
                            "shards" => 1, "shard" => nothing)
    i = 1
    while i <= length(args)
        arg = args[i]
        next() = (i += 1; i <= length(args) ? args[i] : error("$(arg) needs a value"))
        if arg == "--splits"
            opts["splits"] = split(next(), ",")
        elseif arg == "--file"
            push!(opts["files"], next())
        elseif arg == "--rays"
            opts["rays"] = round(Int, parse(Float64, next()))
        elseif arg == "--data-dir"
            opts["data_dir"] = next()
        elseif arg == "--paths"
            opts["paths"] = next()
        elseif arg == "--dists"
            opts["dists"] = [parse(Int, s) for s in split(next(), ",")]
        elseif arg == "--float64"
            opts["store_type"] = Float64
        elseif arg == "--oop-halfwidth"
            opts["oop_halfwidth"] = parse(Int, next())
        elseif arg == "--max-samples"
            opts["max_samples"] = parse(Int, next())
        elseif arg == "--shards"
            opts["shards"] = parse(Int, next())
        elseif arg == "--shard"
            opts["shard"] = parse(Int, next())
        elseif arg == "--no-resume"
            opts["resume"] = false
        elseif arg in ("-h", "--help")
            println("""
                usage: julia --project run/generate_light_transport.jl [options]
                  --splits a,b      splits to simulate (default: $(join(DEFAULT_SPLITS, ",")))
                  --file PATH       simulate this file instead of the splits (repeatable)
                  --rays N          rays per sample (default: per split, $(DEFAULT_RAYS))
                  --data-dir DIR    directory holding <split>_data.h5 (default: paths.toml)
                  --paths FILE      paths.toml to read (default: run/paths.toml)
                  --dists a,b       out-of-plane half-widths to store
                                    (default: $(join(DEFAULT_DISTANCES, ",")));
                                    E and dE are stored once per distance, so this
                                    multiplies the file size directly
                  --float64         store E/dE as Float64 (default Float32, which
                                    halves the file and stays far below MC noise)
                  --oop-halfwidth N out-of-plane half-extent of the accumulator
                                    grid (default: max(dists) + $(OOP_MARGIN)).
                                    Rays still travel beyond it; only the voxels
                                    that are read back are allocated
                  --max-samples N   simulate only the first N rows of each file.
                                    Any prefix is a balanced subset, and the file
                                    can be extended later without recomputation
                  --shards N        the split was generated in N shards by
                                    generate_tissue.py; simulate one of them
                  --shard I         which shard (1-based), i.e. the file
                                    <split>_data.shard<I>of<N>.h5. Ray counts
                                    still come from the split, so sharding never
                                    changes what is simulated
                  --no-resume       recompute samples that are already finished
                """)
            exit(0)
        else
            error("unknown argument $(arg)")
        end
        i += 1
    end
    return opts
end


function main(args)
    opts = parse_args(args)

    jobs = Tuple{String,Int}[]
    if !isempty(opts["files"])
        isnothing(opts["rays"]) && error("--file requires --rays")
        for file in opts["files"]
            push!(jobs, (file, opts["rays"]))
        end
    else
        data_dir = something(opts["data_dir"],
                             TOML.parsefile(opts["paths"])["paths"]["data_dir"])
        shards, shard = opts["shards"], opts["shard"]
        shards > 1 && isnothing(shard) && error("--shards > 1 requires --shard")
        isnothing(shard) || 1 <= shard <= shards ||
            error("--shard must be in 1..$(shards)")
        for split in opts["splits"]
            rays = something(opts["rays"], get(DEFAULT_RAYS, split) do
                error("no default ray count for split $(split); pass --rays")
            end)
            # the shard is only a different *file*; the ray count is still the
            # split's, so a sharded run is the same simulation as an unsharded one
            name = shards == 1 ? "$(split)_data.h5" :
                   "$(split)_data.shard$(shard)of$(shards).h5"
            push!(jobs, (joinpath(data_dir, name), rays))
        end
    end

    for (file, rays) in jobs
        isfile(file) || error("$(file) does not exist -- run generate_tissue.py first")
    end
    for (file, rays) in jobs
        println("\n=== $(file): $(rays) rays/sample, dists $(opts["dists"]), " *
                "stored as $(opts["store_type"])")
        generate_dataset(file, rays; resume=opts["resume"], dists=opts["dists"],
                         store_type=opts["store_type"],
                         oop_halfwidth=opts["oop_halfwidth"],
                         max_samples=opts["max_samples"])
    end
end

main(ARGS)
