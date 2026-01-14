const newaxis = [CartesianIndex()]


function generate_dataset(samples_file, N_rays)
    μ, v, g = h5open(samples_file, "r+") do f
        μ = Array(f["mu"])
        v = Array(f["v"])
        g = Array(f["g"])
        if !("E" in keys(f))
            f["E"] = zeros(size(μ, 1), size(μ, 2), 1, size(μ, 4))
            f["dE"] = zeros(size(μ, 1), size(μ, 2), 1, size(μ, 4))
        end

        return μ, v, g
    end

    N_samples = size(μ, 4)
    grid_size = (size(μ, 1), 10 * size(μ, 1), size(μ, 2)) .* 1e-2 # convert pixels to m

    y_shift = tan(deg2rad(-32)) * 0.5 # mean skin_line depth = 5mm
    beam_shift = (0, grid_size[2]/2 + y_shift, grid_size[3]/2)

    simulation_bounds = ((-0.01 * grid_size[1], 0, -grid_size[3]),
                         ( 2 * grid_size[1],  grid_size[2],  2 * grid_size[3]))
    beam = initialize_beam_sampler(Float32, deg2rad(-32), deg2rad(1), beam_shift)

    for i = 1:N_samples
        t0 = time()
        μ_a = μ[:, newaxis, :, 1, i]
        μ_s = μ[:, newaxis, :, 2, i]
        v_a = v[:, newaxis, :, 1, i]
        v_s = v[:, newaxis, :, 2, i]
        _g = g[:, newaxis, :, 1, i]
        E, dE = light_transport_gpu(beam, μ_a, μ_s, v_a, v_s, _g,
                                    grid_size, simulation_bounds, N_rays, 1000)
        h5open(samples_file, "r+") do f
            for (arr, name) in zip((E, dE), ("E", "dE"))
                f[name][:, :, :, i] = arr
            end
        end
        open(io -> println(io, "Sample $(i)\ttime $(time() - t0)"), "$(samples_file).log", "a+")
    end
end


function generate_trivial_dataset(grid_size, n_pixels, N_rays)
    file_name = "new_trivial_dataset_size=$(grid_size)_Npix=$(n_pixels)_Nrays=$(N_rays)"
    isfile("$(file_name).h5") && error("File with same name already exitst, aborting")

    beam_shift = (0, grid_size[2]/2 + tan(deg2rad(-32)) * 0.5, grid_size[3]/2)
    beam = initialize_beam_sampler(Float32, deg2rad(-32), deg2rad(1), beam_shift)

    simulation_bounds = ((-0.1 * grid_size[1], -2.5 * grid_size[1], -grid_size[3]),
                         ( 2 * grid_size[1],  2.5 * grid_size[1],  2 * grid_size[3]))

    μ_a = 0.05 * ones(n_pixels...); μ_a[50:end, :, :] .= 0.5
    μ_s = 0.15 * ones(n_pixels...); μ_s[50:end, :, :] .= 250
    g = 0.5 * ones(n_pixels...); g[50:end, :, :] .= 0.9
    E = light_transport_gpu(beam, μ_a, μ_s, g,
                            grid_size, simulation_bounds, N_rays, 1000)
    μ_e = zeros(Float32, n_pixels..., 2)
    μ_e[:,:,:,1], μ_e[:,:,:,2] = μ_a, μ_s
    return E
    h5open("$file_name.h5", "w") do file
        write(file, "optical_coefficients", μ_e)
        write(file, "anisotropy", g)
        write(file, "absorbed_energy", E)
    end
    return E
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
                               DirectSampler{T}(p -> angle_sampler(p, angle, beam_divergence)),
                               _p -> embed_and_shift(_p, shift))
    return beam_sampler
end
