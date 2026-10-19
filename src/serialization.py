"""HDF5 storage for trained DeepLight networks.

The layout mirrors the flax variable dicts: dict levels map to HDF5 groups, array leafs to datasets:
    /params/InnerBlock_0/Conv_0/kernel   (3, 3, 4, 16) float32
    /params/...
    /batch_stats/InnerBlock_0/BatchNorm_0/mean
    attrs: model_config (JSON), plus metadata
"""
import dataclasses
import json
from typing import Any

import h5py as h5
import jax
import numpy as np
from flax import linen as nn

from .network import UNet

# Fields that come back from JSON as lists but need a different type to
# construct the model. celu is the only activation actually used in this
# project, so build_model checks for it by name instead of keeping a registry.


def save_network(path, params, batch_stats=None, model_config=None, **metadata):
    """Write a trained network to `path` as HDF5.

    params/batch_stats: nested dicts of arrays, as produced by `model.init` or
    carried on `UNetTrainState`. batch_stats may be None or {} for models that
    keep no running statistics (GroupNorm variants).

    model_config: optional dict describing how to rebuild the model (model
    type, feature counts, ...). Stored as a JSON attribute; entries
    that JSON cannot represent, such as activation functions, are
    stored by name, but cannot be reconstructed automatically.

    metadata: extra scalars kept alongside
    """
    with h5.File(path, "w") as f:
        _write_group(f.create_group("params"), params)
        _write_group(f.create_group("batch_stats"), batch_stats or {})
        f.attrs["model_config"] = json.dumps(jsonify(model_config or {}))
        for key, value in metadata.items():
            f.attrs[key] = _attr_value(value)


def load_network(path, device=None):
    """Inverse of `save_network`.

    Returns dict(params=..., batch_stats=..., model_config=..., metadata=...).
    The two pytrees come back as plain nested dicts of jax arrays, and can be passed
    to `model.apply({"params": ..., "batch_stats": ...}, x, train=False)`.

    device: optional jax device to place the arrays on.
    """
    with h5.File(path, "r") as f:
        params = _read_group(f["params"])
        batch_stats = _read_group(f["batch_stats"])
        model_config = json.loads(f.attrs.get("model_config", "{}"))
        metadata = {k: _unattr_value(v) for k, v in f.attrs.items()
                    if k != "model_config"}

    if device is not None:
        params, batch_stats = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, device), (params, batch_stats))
    return dict(params=params, batch_stats=batch_stats,
                model_config=model_config, metadata=metadata)


def load_metadata(path):
    """`dict(model_config=..., metadata=...)` of a checkpoint, without its weights.
    The attributes are sufficient to identify a run.
    """
    with h5.File(path, "r") as f:
        model_config = json.loads(f.attrs.get("model_config", "{}"))
        metadata = {k: _unattr_value(v) for k, v in f.attrs.items()
                    if k != "model_config"}
    return dict(model_config=model_config, metadata=metadata)


def model_config_from(model):
    """The construction arguments of a UNet instance, as a plain dict ready for
    `save_network`'s `model_config`. `build_model` is the inverse."""
    fields = [f.name for f in dataclasses.fields(model) if f.name not in ("parent", "name")]
    return {name: getattr(model, name) for name in fields}


def build_model(model_config):
    """Reconstruct the UNet from a `model_config` dict, returned by `load_network`.
    Raises if an activation field isn't "celu" or "softplus"."""
    return UNet(**{key: _rebuild_field(key, value)
                   for key, value in model_config.items()})

def load_model_fn(path, device=None):
    """Returs apply-function for network stored at path"""
    stored = load_network(path, device)
    model = build_model(stored["model_config"])
    variables = {"params": stored["params"], "batch_stats": stored["batch_stats"]}

    @jax.jit
    def apply_model(x):
        return model.apply(variables, x, train=False)
    return apply_model




def _rebuild_field(key, value) -> Any:
    """Undo the JSON-ification of one constructor argument."""
    if key in ("activation", "final_activation"):
        if value == "celu":
            return nn.celu
        elif value == "softplus":
            return nn.softplus
        else:
            raise ValueError(f"{key}={value!r}: only 'celu' and 'softplus' can be automatically rebuilt")
    if key in ("feature_counts", "kernel", "n_pixels", "grid_size") and isinstance(value, list):
        return tuple(value)
    return value


def _write_group(group, tree):
    for key, value in tree.items():
        # str(key): a pytree that has been through tree_map may carry integer keys for sequences.
        if hasattr(value, "items"):
            _write_group(group.create_group(str(key)), value)
        else:
            group.create_dataset(str(key), data=np.asarray(value))


def _read_group(group):
    return {key: (_read_group(value) if isinstance(value, h5.Group)
                  else jax.numpy.asarray(value[()]))
            for key, value in group.items()}


def jsonify(config):
    """Best-effort JSON view of a model config: callables (activations) and
    other opaque objects degrade to their name rather than failing the save."""
    if isinstance(config, dict):
        return {str(k): jsonify(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [jsonify(v) for v in config]
    if isinstance(config, (str, int, float, bool)) or config is None:
        return config
    if isinstance(config, np.dtype):
        # Before the `shape` branch: np.dtype instances have a `.shape` ((), for
        # scalar dtypes) and would otherwise come back out of `tolist` unchanged,
        # which json.dumps then chokes on.
        return config.name
    if isinstance(config, np.ndarray) or hasattr(config, "shape"):
        return np.asarray(config).tolist()
    return getattr(config, "__name__", repr(config))


def _attr_value(value):
    """HDF5 attributes take scalars, strings and arrays; anything else (a jax
    scalar, a tuple, a dict) goes through numpy or JSON."""
    if isinstance(value, (str, bytes)):
        return value
    array = np.asarray(value)
    if array.dtype == object:
        return json.dumps(jsonify(value))
    return array


def _unattr_value(value):
    """Undo the numpy round-trip so metadata comes back as plain Python."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
