from typing import Sequence, Tuple, Callable
from jax import numpy as jnp
from flax import linen as nn

from .utils import depth_grid

# Optical coefficient fields: (mu_a, mu_s').
N_MU_CHANNELS = 2


class UNet(nn.Module):
    """Absorbed-energy operator, mu = (mu_a, mu_s') -> E.

    `n_pixels` (X, Z) and `grid_size` (cm, cm) fix the geometry the network was built for. 
    `depth_channel` appends the pixel depth coordinate as a third channel *inside* the model. 
    """
    feature_counts: Sequence[int]
    kernel: Tuple[int, int]
    activation: Callable
    final_activation: Callable
    n_pixels: Tuple[int, int]
    grid_size: Tuple[float, float]
    norm: str = "batch"  # "batch" | "group"
    n_groups: int = 8
    depth_channel: bool = True

    @nn.compact
    def __call__(self, x, train):
        self._validate_input(x)
        # Fixed, sample-independent conditioning of the optical coefficients
        # mu_a ~ O(1), mu_s' ~ O(10)
        z = x.at[:, :, :, 1].divide(10.)
        if self.depth_channel:
            grid = depth_grid(self.n_pixels, self.grid_size)
            # NOTE check for performance hit here
            depth = jnp.broadcast_to(grid, (x.shape[0], *grid.shape[1:]))
            z = jnp.concatenate((z, depth.astype(z.dtype)), axis=3)
        z = self._inner_call(z, train)
        z = self.final_activation(z)
        return z * x[:, :, :, 0:1] / 20

    def _validate_input(self, x):
        if x.ndim != 4 or x.shape[-1] != N_MU_CHANNELS:
            raise ValueError(f"expected a (batch, X, Z, {N_MU_CHANNELS}) field, got shape {x.shape}.")
        if tuple(x.shape[1:3]) != tuple(self.n_pixels):
            raise ValueError(f"input is {tuple(x.shape[1:3])} pixels, but the model "
                             f"was built for n_pixels={tuple(self.n_pixels)}")

    def _inner_call(self, x, train):
        n_layers = len(self.feature_counts)
        # Input-channels are passed through to each block, such that each InnerBlock sees a (downscaled) version of those inputs
        n_passthrough = x.shape[3]

        def block(f):
            return InnerBlock(f, self.kernel, self.activation, n_passthrough, self.norm, self.n_groups)

        y = [block(self.feature_counts[0])(x, train),]
        padding = []
        for i in range(n_layers - 1):
            padding += [((y[-1].shape[1] % 4)//2, (y[-1].shape[2] % 4)//2),]
            r = nn.avg_pool(y[-1], (2, 2), strides=(2, 2),
                            padding=((padding[-1][0], padding[-1][0]), (padding[-1][1], padding[-1][1])))
            r = block(self.feature_counts[i + 1])(r, train)
            y += [r,]

        p = padding.pop()
        z = nn.ConvTranspose(self.feature_counts[-1], (2, 2),
                             strides=(2, 2), padding=(1 - p[0], 1 - p[1]))(y.pop())
        for i in range(n_layers - 2):
            z = jnp.concatenate((y.pop(), z), axis=3)
            z = block(self.feature_counts[-2-i])(z, train)
            p = padding.pop()
            z = nn.ConvTranspose(self.feature_counts[-2-i], (2, 2),
                                 strides=(2, 2), padding=(1 - p[0], 1 - p[1]))(z)
        # NOTE Change to previous version, validate in training
        z = jnp.concatenate((y.pop(), z), axis=3)
        z = block(self.feature_counts[0])(z, train)
        z = nn.Conv(1, (1, 1))(z)
        return z


class InnerBlock(nn.Module):
    features: int
    kernel: Tuple[int, int]
    activation: Callable
    n_passthrough: int
    norm: str = "batch"  # "batch" | "group"
    n_groups: int = 8

    def _norm(self, x, train):
        if self.norm == "batch":
            return nn.BatchNorm(use_running_average=not train)(x)
        elif self.norm == "group":
            channels = x.shape[-1]
            if channels % self.n_groups:
                raise ValueError(f"{channels} channels is not divisible by n_groups={self.n_groups}")
            return nn.GroupNorm(num_groups=self.n_groups)(x)
        else:
            raise ValueError("norm must be either 'group' or 'batch'")

    @nn.compact
    def __call__(self, x, train):
        # `n_passthrough` are the input channels, they shoudn't be rescaled and
        # are not considered in the initial normalisation layer
        inputs = x[:, :, :, :self.n_passthrough]
        body = x[:, :, :, self.n_passthrough:]
        if body.shape[-1]:
            body = self._norm(body, train)
        h = jnp.concatenate((inputs, body), axis=3)
        h = self.activation(nn.Conv(self.features, self.kernel)(h))
        h = self._norm(h, train)
        h = self.activation(nn.Conv(self.features, self.kernel)(h))
        return jnp.concatenate((inputs, h), axis=3)
