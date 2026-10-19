const newaxis = [CartesianIndex()]

# Out-of-plane half-widths stored per sample. 100 and 150 were dropped: they cost
# a full E and dE array each (dE once per direction) and nothing trains on them.
# 65 is the widest, which also sets the default accumulator extent.
const DEFAULT_DISTANCES = [0, 2, 5, 10, 25, 65]
const BEAM_ANGLE = deg2rad(-32)
const BEAM_DIVERGENCE = deg2rad(1)
# Out-of-plane voxels kept beyond the widest stored slice. Zero: a grid of
# exactly `2 * maximum(dists) + 1` voxels was measured to be indistinguishable
# from the full 545-voxel extrusion at every stored distance (differences sat on
# the run-to-run Monte Carlo noise floor, d=65 included, which is the boundary
# voxel a margin would protect). Rays still travel far beyond the grid; their
# deposits land in edge padding that is discarded.
const OOP_MARGIN = 0


"""
    beam_oop_at_depth(depth; n_rays)

Out-of-plane position of an unshifted beam once it has reached `depth`, in cm.

Measured, not derived: `n_rays` rays are drawn from the sampler with zero shift,
and their mean entry point plus mean direction give
`y(depth) = y0 + (d_y/d_depth) * depth`. `dataset_geometry` subtracts this from
the slice centre to aim the beam. See its docstring for why this is measured
rather than written as `tan(BEAM_ANGLE) * depth`.

The global RNG is seeded and then restored, so the answer is reproducible and the
Monte Carlo stream of the simulation proper is unaffected.
"""
function beam_oop_at_depth(depth; n_rays=400_000)
    rng_state = copy(Random.default_rng())
    try
        Random.seed!(0)
        beam = initialize_beam_sampler(Float32, BEAM_ANGLE, BEAM_DIVERGENCE,
                                       (0., 0., 0.))
        p, d = Moca.next(beam, n_rays)
        colmean(a, i) = sum(@view a[i, :]) / size(a, 2)
        y0 = colmean(p, 2)
        # component 1 is depth, 2 out of plane; the ratio of the means is the
        # walk per unit depth of the beam as a whole
        dy_ddepth = colmean(d, 2) / colmean(d, 1)
        return y0 + dy_ddepth * depth
    finally
        copy!(Random.default_rng(), rng_state)
    end
end


"""
    dataset_geometry(n_depth, n_lateral, pixel_size; oop_halfwidth, skin_depth)

Simulation grid for a 2D tissue sample of `n_depth x n_lateral` pixels.

The sample is extruded out of plane so that rays can spread sideways before
their contribution is projected back onto the slice.

The beam enters at the top of the grid but travels at an angle to the depth axis,
so its out-of-plane position walks as it goes down. It must be *centred where it
meets the tissue*, not where it enters the coupling medium, so the entry point is
backed off by that walk over `skin_depth`. This matters: the beam is only
`sigma ~ 0.49 cm` wide out of plane at the source while the walk over a 0.5 cm
skin depth is 0.31 cm, so getting it wrong pushes the illuminated region most of
a beam width off the slice that is actually stored.

Rather than deriving that offset from `BEAM_ANGLE`, it is *measured* from the
beam sampler: a batch of rays is drawn and their mean entry point and mean
direction give the walk per unit depth directly. Two reasons. First, the sign
convention is not obvious -- the local `angle_sampler` below rotates by
`-BEAM_ANGLE` where MoCA's rotates by `+BEAM_ANGLE`, so the analytic expression
that is correct for one is exactly backwards for the other. Second, the
illumination profile is not centred on its own array (its out-of-plane centroid
sits 0.16 cm off, a third of a beam width); measuring folds that in for free,
where an analytic offset would silently ignore it. The measurement uses a fixed
seed and restores the global RNG, so the geometry is reproducible and the ray
stream of the simulation itself is untouched.

`skin_depth` is a rough typical depth of the skin line, taken from the dataset's
`mean_skin_depth_cm` attribute; it does not need to match any individual sample,
whose surface fluctuates around it by far less than the beam width.

`oop_halfwidth` sets how far that extrusion reaches (the grid is
`2 * oop_halfwidth + 1` voxels wide out of plane); it defaults to the full
`n_depth`. Only the slices within `dists` are ever read, so extruding further
than `maximum(dists)` (plus margin) allocates and writes accumulator voxels that
are thrown away -- and `dE` pays that per direction, which is what makes the
out-of-plane extent worth trimming.

Truncating is safe because the medium is *extruded*: it does not vary out of
plane. Rays leaving the grid are clamped into MoCA's edge-replicated padding, so
they keep scattering against the correct optical properties and can re-enter,
while their deposits land in padding that `unpad` discards. The simulation
bounds are deliberately kept at the full untruncated extent -- shrinking them
with the grid would terminate exactly those excursions instead of letting them
return, which *would* change the answer.
"""
function dataset_geometry(n_depth, n_lateral, pixel_size; oop_halfwidth=n_depth,
                          skin_depth=MEAN_SKIN_DEPTH)
    shape = (n_depth, 2 * oop_halfwidth + 1, n_lateral)
    grid_size = shape .* pixel_size    # pixels -> cm
    # bounds follow the full extrusion, not the (possibly trimmed) grid
    full_oop = (2 * n_depth + 1) * pixel_size
    # aim the entry point so the beam crosses the slice centre at the skin line
    y_shift = grid_size[2]/2 - beam_oop_at_depth(skin_depth)
    beam_shift = (0, y_shift, grid_size[3]/2)
    # symmetric about the beam, so trimming the grid cannot bias which
    # out-of-plane excursions survive
    simulation_bounds = ((-0.01 * grid_size[1], beam_shift[2] - full_oop, -grid_size[3]),
                         ( 2 * grid_size[1],    beam_shift[2] + full_oop,  2 * grid_size[3]))
    beam = initialize_beam_sampler(Float32, BEAM_ANGLE, BEAM_DIVERGENCE, beam_shift)
    return (shape=shape, grid_size=grid_size,
            simulation_bounds=simulation_bounds, beam=beam)
end


"""
    prepare_transport_datasets!(f, dists, n_depth, n_lateral, K, N)

Create the `E`, `dE` groups and the resume marker if they are not there yet.

`E/<dist>` is `(depth, lateral, 1, N)` and `dE/<dist>` is `(depth, lateral, K, N)`;
read from python these are `(N, 1, X, Z)` and `(N, K, X, Z)`, which is what the
dataloader expects.

`store_type` is the on-disk element type. The transport still accumulates in
Float64 -- only the projected slices are stored, and `Float32` keeps ~7 digits
where the Monte Carlo noise is already at the second, so it halves the file for
no loss. `dE` dominates the file size (one array per direction per distance),
which is what makes this worth a flag.
"""
function prepare_transport_datasets!(f, dists, n_depth, n_lateral, K, N,
                                     store_type=Float32)
    if !("E" in keys(f))
        gE = create_group(f, "E")
        gdE = create_group(f, "dE")
        for dist in dists
            gE[string(dist)] = zeros(store_type, n_depth, n_lateral, 1, N)
            gdE[string(dist)] = zeros(store_type, n_depth, n_lateral, K, N)
        end
    else
        for dist in dists
            haskey(f["E"], string(dist)) ||
                error("file has an E group but no distance $(dist); it was " *
                      "started with a different `dists`")
        end
    end
    "light_transport_done" in keys(f) || (f["light_transport_done"] = zeros(UInt8, N))
    return nothing
end


"""
    generate_dataset(samples_file, N_rays; dists, resume, pixel_size)

Second stage of dataset generation: Monte Carlo light transport for every tissue
sample in `samples_file` (written by `run/generate_tissue.py`).

Each sample is simulated once; the `K` derivative directions stored in `v` are
propagated through that single ray ensemble, so they cost a fraction of a
separate run each. The absorbed energy and its derivatives are projected onto
the slice for every half-width in `dists` and appended to the same file.

Samples are read one at a time and flagged in `light_transport_done` as they are
finished, so a run of any length can be interrupted and resumed (`resume=false`
recomputes everything). Progress is echoed to stdout and appended to
`<samples_file>.log`.
"""
function generate_dataset(samples_file, N_rays; dists=DEFAULT_DISTANCES,
                          resume=true, pixel_size=1e-2, iterations_bunch=1000,
                          store_type=Float32, oop_halfwidth=nothing,
                          max_samples=nothing)
    n_depth, n_lateral, N_samples, K, skin_depth = h5open(samples_file, "r") do f
        (size(f["mu"], 1), size(f["mu"], 2), size(f["mu"], 4), size(f["v"], 4),
         # the tissue stage records where its skin line sits on average; the beam
         # has to be aimed at it, not at the surface of the coupling medium
         Float64(attrs(f)["mean_skin_depth_cm"]))
    end
    # only the slices up to maximum(dists) are ever read back out
    oop_halfwidth = something(oop_halfwidth,
                              min(n_depth, maximum(dists) + OOP_MARGIN))
    oop_halfwidth < maximum(dists) &&
        error("oop_halfwidth=$(oop_halfwidth) is smaller than the widest " *
              "requested slice ($(maximum(dists)))")

    todo = h5open(samples_file, "r+") do f
        prepare_transport_datasets!(f, dists, n_depth, n_lateral, K, N_samples,
                                    store_type)
        resume ? findall(iszero, Array(f["light_transport_done"])) : collect(1:N_samples)
    end
    # Simulate only a prefix of the file. The rows of a split are interchangeable
    # (a single RowSpec) or interleaved (several), so any prefix is a balanced
    # dataset -- which is what makes it possible to simulate 512 rows now, train
    # on them, and extend the same file to 1024 later without redoing anything.
    if !isnothing(max_samples)
        todo = filter(<=(max_samples), todo)
    end

    geom = dataset_geometry(n_depth, n_lateral, pixel_size;
                            oop_halfwidth=oop_halfwidth, skin_depth=skin_depth)
    log_line(msg) = begin
        println(msg)
        flush(stdout)
        open(io -> println(io, msg), "$(samples_file).log", "a+")
    end
    log_line("$(samples_file): $(length(todo))/$(N_samples) samples to simulate" *
             (isnothing(max_samples) ? "" : " (capped at row $(max_samples))") *
             ", $(K) directions, $(N_rays) rays, grid $(geom.shape)" *
             ", skin depth $(skin_depth) cm")

    # allocated once and refilled per sample; the derivative axis leads, matching
    # the (K, size(μ_a)...) layout MoCA expects
    μ_a = zeros(geom.shape)
    μ_s = zeros(geom.shape)
    _g = zeros(geom.shape)
    v_a = zeros(K, geom.shape...)
    v_s = zeros(K, geom.shape...)

    t_start = time()
    for (n, i) in enumerate(todo)
        t0 = time()
        # read this sample only: the full mu/v of a large dataset does not fit in
        # memory, and only one sample at a time is ever needed
        μ_i, v_i, g_i = h5open(samples_file, "r") do f
            (f["mu"][:, :, :, i], f["v"][:, :, :, :, i], f["g"][:, :, 1, i])
        end
        μ_a .= μ_i[:, newaxis, :, 1]
        μ_s .= μ_i[:, newaxis, :, 2]
        _g .= g_i[:, newaxis, :]
        for k = 1:K
            v_a[k, :, :, :] .= v_i[:, newaxis, :, 1, k]
            v_s[k, :, :, :] .= v_i[:, newaxis, :, 2, k]
        end

        E, dE = light_transport_gpu(geom.beam, μ_a, μ_s, v_a, v_s, _g,
                                    geom.grid_size, geom.simulation_bounds,
                                    N_rays, iterations_bunch; block=4)

        h5open(samples_file, "r+") do f
            centre = size(E, 2)÷2 + 1
            for dist in dists
                cols = (-dist:dist) .+ centre
                f["E"][string(dist)][:, :, 1, i] =
                    dropdims(sum(E[:, cols, :], dims=2), dims=2)
                # dE is (K, depth, out-of-plane, lateral); collapse the slice and
                # move K last so the stored layout matches E's trailing axes.
                ds = dropdims(sum(dE[:, :, cols, :], dims=3), dims=3)
                f["dE"][string(dist)][:, :, :, i] = permutedims(ds, (2, 3, 1))
            end
            f["light_transport_done"][i] = UInt8(1)
        end

        elapsed = time() - t_start
        eta = elapsed / n * (length(todo) - n)
        log_line("Sample $(i)/$(N_samples)\ttime $(round(time() - t0, digits=1)) s" *
                 "\tETA $(round(eta/3600, digits=2)) h")
    end
    log_line("$(samples_file): done in $(round((time() - t_start)/3600, digits=2)) h")
end


####################################################################################################
# Beam Sampler
####################################################################################################
function embed_and_shift(_p, s)
    p = Array{Float32, 2}(undef, 3, size(_p, 2))
    p[1,:] .= s[1]
    p[2,:] .= _p[2,:] .+ s[2]
    p[3,:] .= _p[1,:] .+ s[3]
    return p
end


function angle_sampler(p, mean_angle, Δθ)
    Δd = zeros(3, size(p,2))
    θ = Δθ .* randn(size(p, 2))
    ϕ = 2π .* rand(size(p, 2))

    Δd[1, :] .= cos.(θ)
    Δd[2, :] .= sin.(θ) .* cos.(ϕ)
    Δd[3, :] .= sin.(θ) .* sin.(ϕ)

    d = similar(Δd)
    @. d[1, :] = sin(mean_angle) * Δd[2,:] + cos(mean_angle) * Δd[1, :]
    @. d[2, :] = cos(mean_angle) * Δd[2,:] - sin(mean_angle) * Δd[1, :]
    @. d[3, :] = Δd[3, :]
    return d
end

function initialize_beam_sampler(T, angle, beam_divergence, shift)
    profile, res = h5open("$(@__DIR__)/illuminationProfile_new.h5") do file
        profile = T.(Array(file["profile"]))
        res = T(Array(file["px_size"])) * 100
        return profile, res
    end

    beam_sampler = BeamSampler(profile, (res, res), 
                               p -> angle_sampler(p, angle, beam_divergence),
                               _p -> embed_and_shift(_p, shift))
    return beam_sampler
end
