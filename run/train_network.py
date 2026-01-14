import jax
import flax.linen as nn
import deeplight as dl


if __name__ == "__main__":
    seed_key = jax.random.key(7864532)
    key_list = jax.random.split(seed_key, 5)

    n_pixels = (400, 272)
    paths = dl.load_paths('./paths.toml')
    data_dir = paths['data_dir']

    # dl.datagen.tissue_prior.compute_all_datasets(data_dir)

    data_train = dl.load_data(f"{data_dir}/train_data.h5")
    data_val = dl.load_data(f"{data_dir}/val_data.h5")
    data_test = dl.load_data(f"{data_dir}/test_data.h5")

    grid_size = [n/1e2 for n in n_pixels]
    train_ds = dl.DeepLightDataset(*data_train, grid_size=grid_size, n_pixels=n_pixels)
    val_ds = dl.DeepLightDataset(*data_val, grid_size=grid_size, n_pixels=n_pixels)
    test_ds = dl.DeepLightDataset(*data_test, grid_size=grid_size, n_pixels=n_pixels)

    for (i, key) in enumerate(key_list):
        print(i, ": ", key)
        init_key, train_shuffle = jax.random.split(key, 2)

        train = dl.DataLoader(train_ds, 4, train_shuffle)
        val = dl.DataLoader(val_ds, 4)
        test = dl.DataLoader(test_ds, 4)

        optim_config = (1e-3, 100, 2, 0, len(train), 1e-5)

        ################################################################################
        for alpha in [0.1, 0]:
            print(alpha)
            log_name = f"{i}_jac={alpha}"
            m, ms, w, cp, total_steps = dl.initialize(init_key,
                                                      model_type="full",
                                                      unet_params=((16, 32, 64, 128, 256, 256),
                                                                   (3, 3),
                                                                   nn.celu, nn.softplus),
                                                      input_shape=(1, *n_pixels, 3),
                                                      optimizer_config=optim_config,
                                                      name=log_name,
                                                      logdir=paths["network_weights_dir"])
            ms = dl.train_epochs(ms, total_steps, train, val, test, alpha, w, cp)
