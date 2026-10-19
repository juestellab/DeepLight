module DeepLight

export generate_dataset, dataset_geometry,
       DEFAULT_DISTANCES, OOP_MARGIN

using HDF5
using Moca
using Random

include("datagen/generate_data.jl")

end
