import numpy as np
import jax.numpy as jnp
import h5py as h5
import jax


def load_data(path, oop_slices, n_directions=None, per_sampler=True,
              samplers=None):
    """Load one split, returns
        mu  (N, X, Z, C)      dmu (N, K, X, Z, C)
        E   (N, X, Z, 1)      dE  (N, K, X, Z, 1)

    `n_directions` keeps a subset of the K stored derivative directions,
    balanced across the tangent samplers (see `select_directions`); None keeps
    all of them.

    `samplers` restricts which tangent samplers may contribute at all, as
    indices into `direction_sampler_names`, None uses every one of them.
    """
    with h5.File(path) as file:
        mu = np.array(file["mu"])
        dmu = np.array(file["v"])
        E = np.array(file["E"][str(oop_slices)])
        dE = np.array(file["dE"][str(oop_slices)])
        g = np.array(file["g"])
        sampler_index = np.asarray(file["direction_sampler_index"][()])
    if n_directions is not None or samplers is not None:
        dmu, dE = select_directions(dmu, dE, n_directions, sampler_index,
                                    per_sampler, samplers)
    mu[:, 1:2, :, :] *= 1 - g
    dmu[:, :, 1:2, :, :] *= (1 - g)[:, None]
    E = E * 1e4
    dE = dE * 1e4

    mu, E = [np.moveaxis(arr, 1, -1) for arr in (mu, E)]
    dmu = np.moveaxis(dmu, 2, -1)
    dE = dE[..., None]
    return mu, dmu, E, dE


def get_skin_index(path, threshold=0.5):
    with h5.File(path) as file:
        coupling = np.asarray(file["masks"][:, 0])       # (N, X, Z)
    return np.argmax(coupling < threshold, axis=2).astype(np.int32)


def load_direction_groups(path):
    """Tangent sampler of each stored direction: (index per direction, names)"""
    with h5.File(path) as file:
        index = np.asarray(file["direction_sampler_index"][()])
        names = tuple(file.attrs["direction_sampler_names"])
    return index, names


def load_sample_groups(path):
    """Returns index array of which sampler was used to compute the corresponding sample, and their names."""
    with h5.File(path) as file:
        index = np.asarray(file["sampler_index"][()])
        names = tuple(file.attrs["sampler_names"])
    return index, names


def select_directions(dmu, dE, n_directions, sampler_index, per_sampler=True,
                      samplers=None):
    """Keep a subset of the stored derivative directions of every sample while
    keeping the samplers balanced.

    `sampler_index` is the (K,) tangent sampler of each stored direction, as
    returned by `load_direction_groups`; `samplers` is the subset of sampler
    indices to draw from (None = all of them), which is how a sampler is held
    out of a split entirely. Everything below counts only the kept samplers, so
    `n_directions` is unaffected by how many samplers are excluded.

    `per_sampler=True` selects `n_directions` from each kept sampler, so every
    sample ends up with `n_directions x len(samplers)` directions.

    `per_sampler=False` applies it per *sample* instead: each sample keeps
    exactly `n_directions` directions, taken from the grid in direction-major
    order -- the first direction of every kept sampler, then the second, and so
    on -- with sample `r` starting at `r * n_directions` in that ring.

    `n_directions=None` keeps every direction of the kept samplers.
    """
    n_rows, n_stored = dmu.shape[:2]
    sampler_index = np.asarray(sampler_index)
    if sampler_index.shape != (n_stored,):
        raise ValueError(f"sampler_index has {sampler_index.shape[0]} entries "
                         f"for {n_stored} stored directions")

    stored = np.unique(sampler_index)
    if samplers is None:
        samplers = stored
    else:
        # sorted and deduplicated, so the kept directions stay in storage order
        samplers = np.unique(np.atleast_1d(np.asarray(samplers, dtype=int)))
        unknown = set(samplers.tolist()) - set(stored.tolist())
        if unknown or samplers.size == 0:
            raise ValueError(
                f"samplers must be a non-empty subset of {stored.tolist()}, "
                f"got {samplers.tolist()}")

    # (n_kept, per_sampler) grid of stored columns; rectangular by construction,
    # every sample carries the same number of directions from every sampler
    grouped = [np.flatnonzero(sampler_index == s) for s in samplers]
    if len({len(g) for g in grouped}) != 1:
        raise ValueError("tangent samplers hold unequal numbers of directions: "
                         f"{[len(g) for g in grouped]}")
    columns = np.stack(grouped)
    n_kept = columns.size

    limit = columns.shape[1] if per_sampler else n_kept
    if n_directions is None:
        n_directions = limit
    if not 0 < n_directions <= limit:
        raise ValueError(
            f"n_directions must be in 1..{limit} (directions per "
            f"{'tangent sampler' if per_sampler else 'sample'}), "
            f"got {n_directions}")
    if n_directions == limit and n_kept == n_stored:  # every column, in order
        return dmu, dE

    if per_sampler:
        keep = np.broadcast_to(columns[:, :n_directions].ravel(),
                               (n_rows, len(columns) * n_directions))
    else:
        ring = columns.T.ravel()
        starts = np.arange(n_rows)[:, None] * n_directions
        keep = np.sort(ring[(starts + np.arange(n_directions)) % n_kept], axis=1)

    take = lambda arr: np.take_along_axis(
        arr, keep.reshape(*keep.shape, *([1] * (arr.ndim - 2))), axis=1)
    return take(dmu), take(dE)


class DeepLightDataset:
    """Host-resident (mu, dmu, E, dE) with a shared leading sample axis.

    Samples are only moved to the device when indexed, thus saving device memory.
    """

    def __init__(self, mu, dmu, E, dE, n_samples=None):
        if n_samples is None:
            n_samples = mu.shape[0]
        self.data = [np.asarray(arr)[:n_samples] for arr in (mu, dmu, E, dE)]
        self._len = self.data[0].shape[0]
        self.n_directions = self.data[1].shape[1]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # jnp index arrays are in use at several call sites; numpy would pull
        # them back one element at a time
        if not isinstance(idx, (int, np.integer, slice)):
            idx = np.asarray(idx)
        return [jax.device_put(arr[idx, ...]) for arr in self.data]


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
        permutation = jnp.arange(len(self.data))
        if self._key is not None:
            self._key, key = jax.random.split(self._key)
            permutation = jax.random.permutation(key, permutation)
        self._permutation = np.asarray(permutation)
        return self

    def __next__(self):
        if self._index >= self._len:
            raise StopIteration
        idx = self._index * self._batchsize
        self._index += 1
        return self.data[self._permutation[idx:(idx + self._batchsize)]]
