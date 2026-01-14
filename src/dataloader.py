import numpy as np
import jax.numpy as jnp
import h5py as h5
import jax


def load_data(path):
    with h5.File(path) as file:
        mu = np.array(file["mu"])
        dmu = np.array(file["v"])
        E = np.array(file["E"])
        dE = np.array(file["dE"])
        g = np.array(file["g"])
    mu[:, 1:2, :, :] *= 1 - g
    dmu[:, 1:2, :, :] *= 1 - g
    E = E * 1e4
    dE = dE * 1e4
    mu, dmu, E, dE = [np.moveaxis(arr, 1, -1) for arr in (mu, dmu, E, dE)]
    return mu, dmu, E, dE


class DeepLightDataset:
    def __init__(self, mu, dmu, E, dE,
                 grid_size=(4., 4.), n_pixels=(400, 400), n_samples=None):
        if n_samples is not None:
            n_samples = mu.shape[0]
        mu, dmu, E, dE = [arr[0:n_samples, :n_pixels[0], :n_pixels[1], :]
                          for arr in (mu, dmu, E, dE)]
        self._len = mu.shape[0]
        depth_array = np.meshgrid(np.linspace(0, grid_size[1], n_pixels[1]),
                                  np.linspace(0, grid_size[0], n_pixels[0])
                                  )[0][None, :, :, None]
        # bring in correct shape and convert units to cm
        depth_array = np.repeat(depth_array, mu.shape[0], axis=0)
        _mu = np.concatenate((mu, depth_array), axis=3)
        _dmu = np.concatenate((dmu, jnp.zeros((*dmu.shape[:-1], 1))), axis=3)
        self.data = [jax.device_put(arr) for arr in (_mu, _dmu, E, dE)]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda x: x[idx, :, :, :], self.data)


class DataLoader:
    def __init__(self, ds: DeepLightDataset, batch_size: int, rng_key=None):
        # NOTE by default drops last batch, if smaller than batch_size
        self._len = len(ds)//batch_size
        self._batchsize = batch_size
        self._key = rng_key
        self.data = ds

    def __len__(self):
        return self._len

    def __iter__(self):
        self._index = 0
        self._permutation = jnp.arange(len(self.data))
        if self._key is not None:
            self._key, key = jax.random.split(self._key)
            self._permutation = jax.random.permutation(key, self._permutation)
        return self

    def __next__(self):
        if self._index < self._len:
            idx = self._index * self._batchsize
            val = self.data[self._permutation[idx:(idx + self._batchsize)]]
            self._index += 1
            return val
        else:
            if self._key is not None:
                self._shuffle()
            raise StopIteration

    def _shuffle(self):
        self._key, key = jax.random.split(self._key)
        self._permutation = jax.random.permutation(key, self._permutation)
