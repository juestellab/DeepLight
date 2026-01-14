from typing import Sequence, Tuple, Callable
from jax import numpy as jnp
from flax import linen as nn


class UNet(nn.Module):
    feature_counts: Sequence[int]
    kernel: Tuple[int, int]
    activation: Callable
    final_activation: Callable

    @nn.compact
    def __call__(self, x, train):
        # z = x.at[:, :, :, 0:2].divide(jnp.array([15, 1500]).reshape(1, 1, 1, 2))
        z = x
        z = self._inner_call(z, train)
        z = self.final_activation(z)
        return z * x[:, :, :, 0:1] / 20

    def _inner_call(self, x, train):
        n_layers = len(self.feature_counts)
        y = [InnerBlock(self.feature_counts[0], self.kernel, self.activation)(x, train),]
        padding = []
        for i in range(n_layers - 1):
            padding += [((y[-1].shape[1] % 4)//2, (y[-1].shape[2] % 4)//2),]
            r = nn.avg_pool(y[-1], (2, 2), strides=(2, 2),
                            padding=((padding[-1][0], padding[-1][0]), (padding[-1][1], padding[-1][1])))
            r = InnerBlock(self.feature_counts[i + 1], self.kernel, self.activation)(r, train)
            y += [r,]

        p = padding.pop()
        z = nn.ConvTranspose(self.feature_counts[-1], (2, 2),
                             strides=(2, 2), padding=(1 - p[0], 1 - p[1]))(y.pop())
        for i in range(n_layers - 2):
            z = jnp.concatenate((z, y.pop()), axis=3)
            z = InnerBlock(self.feature_counts[-2-i], self.kernel, self.activation)(z, train)
            p = padding.pop()
            z = nn.ConvTranspose(self.feature_counts[-2-i], (2, 2),
                                 strides=(2, 2), padding=(1 - p[0], 1 - p[1]))(z)
        z = InnerBlock(self.feature_counts[0], self.kernel, self.activation)(z, train)
        z = nn.Conv(1, (1, 1))(z)
        return z


class InnerBlock(nn.Module):
    features: int
    kernel: Tuple[int, int]
    activation: Callable

    @nn.compact
    def __call__(self, x, train: bool):
        inputs = x[:, :, :, :3]
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(nn.Conv(self.features, self.kernel)(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(nn.Conv(self.features, self.kernel)(x))
        x = jnp.concatenate((inputs, x), axis=3)
        return x


class InnerBlockSeparableConv(nn.Module):
    features: int
    kernel: Tuple[int, int]
    activation: Callable

    @nn.compact
    def __call__(self, x, train: bool):
        inputs = x[:, :, :, :3]
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(self.features, (1, 1))(x)
        x = nn.Conv(self.features, self.kernel, feature_group_count=self.features)(x)
        x = nn.Conv(self.features, (1, 1))(x)
        x = self.activation(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Conv(self.features, (1, 1))(x)
        x = nn.Conv(self.features, self.kernel, feature_group_count=self.features)(x)
        x = nn.Conv(self.features, (1, 1))(x)
        x = self.activation(x)
        x = jnp.concatenate((inputs, x), axis=3)
        return x


class UNet_basic(nn.Module):
    feature_counts: Sequence[int]
    kernel: Tuple[int, int]
    activation: Callable
    final_activation: Callable

    @nn.compact
    def __call__(self, x, train):
        z = self._inner_call(x, train)
        z = self.final_activation(z)
        return z

    def _inner_call(self, x, train):
        assert x.shape[1] == x.shape[2]
        assert x.shape[1] % 2 == 0
        n_layers = len(self.feature_counts)
        y = [InnerBlock_nocat(self.feature_counts[0], self.kernel, self.activation)(x, train),]
        padding = []
        for i in range(n_layers - 1):
            padding += [(y[-1].shape[1] % 4)//2,]
            r = nn.avg_pool(y[-1], (2, 2), strides=(2, 2),
                            padding=((padding[-1], padding[-1]), (padding[-1], padding[-1])))
            r = InnerBlock_nocat(self.feature_counts[i + 1], self.kernel, self.activation)(r, train)
            y += [r,]

        z = nn.ConvTranspose(self.feature_counts[-1], (2, 2), strides=(2, 2),
                             padding=1-padding.pop())(y.pop())
        for i in range(n_layers - 2):
            z = jnp.concatenate((z, y.pop()), axis=3)
            z = InnerBlock_nocat(self.feature_counts[-2-i], self.kernel, self.activation)(z, train)
            z = nn.ConvTranspose(self.feature_counts[-2-i], (2, 2), strides=(2, 2),
                                 padding=1-padding.pop())(z)
        z = InnerBlock_nocat(self.feature_counts[0], self.kernel, self.activation)(z, train)
        z = nn.Conv(1, (1, 1))(z)
        return z


class InnerBlock_nocat(nn.Module):
    features: int
    kernel: Tuple[int, int]
    activation: Callable

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(nn.Conv(self.features, self.kernel)(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation(nn.Conv(self.features, self.kernel)(x))
        return x
