using TOML
using DeepLight
using CUDA

function main(base_path)
    generate_dataset("$(base_path)/train_data.h5", 125_000_000)
  
    generate_dataset("$(base_path)/val_data.h5", 250_000_000)
    generate_dataset("$(base_path)/test_data.h5", 250_000_000)
end

main(TOML.parsefile("./paths.toml")["data_dir"])
