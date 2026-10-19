import tomli
import pathlib
import subprocess
import numpy as np
import jax.numpy as jnp

# `deeplight/__init__.py` star-imports this module; without __all__ the stdlib
# imports above would land in the package namespace as well.
__all__ = ["load_paths", "git_commit", "git_dirty", "nl", "depth_grid",
           "DEFAULT_A_E", "DEFAULT_A_DE", "LINEAR_A", "DEFAULT_C_E", "DEFAULT_C_DE", "LINEAR_C",
           "LOG_A", "LOG_C"]


def load_paths(config_file):
    with open(config_file, 'rb') as f:
        return tomli.load(f)['paths']


def git_commit():
    """Short hash of the working tree this package was imported from, for
    stamping provenance into generated data and checkpoints. "unknown" when
    git is unavailable or the tree is not a repository."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=pathlib.Path(__file__).parent,
                                       text=True).strip()
    except Exception:
        return "unknown"


def git_dirty():
    """Whether that working tree has uncommitted changes"""
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=pathlib.Path(__file__).parent,
                                            text=True).strip())
    except Exception:
        return False


# The scales the published runs were trained with
DEFAULT_A_E = 0.5
DEFAULT_C_E = 5e-4
DEFAULT_A_DE = 3
DEFAULT_C_DE = 1e-2
LINEAR_A = 1
LINEAR_C = jnp.inf

LOG_A = jnp.inf
LOG_C = 1e-6


def depth_grid(n_pixels, grid_size):
    """Depth coordinate of every pixel, as a (1, X, Z, 1) field in cm."""
    z = np.linspace(0., grid_size[1], n_pixels[1], endpoint=False)
    return jnp.asarray(np.broadcast_to(z[None, None, :, None],
                                       (1, n_pixels[0], n_pixels[1], 1)))


def nl(x, a, c):
    """Pseudo-log reweighting for absorbed energies and its derivative.
    `nl` introduces a log-distributed stretch over the range c < |x| < a,
    with the ranges outside of those bounds exhibiting linear scaling.

        nl(x, a, c) = x/a + sign(x) * log((e^c/a - e^-|x/a|) / (e^c/a - 1))

    # Numerically stable: the direct form evaluates `exp(c) - exp(-|x|)`, where
    Numerically stable implementation utilizing expm1 instead of `exp(c) - exp(-|x|)`,
    which can lead to substantial loss of precision due to near cancellation
    of the terms.

    The two limiting regimes are special-cased:

        c = inf           -> the identity (up to the 1/a scale): no stretch at all
        a = inf           -> pure log, with a floor at c: sign(x) * (log(|x| + c) - log(c))

    `a` and `c` must be known at trace time -- Python scalars, or constants
    closed over by the jitted function -- because the regime is chosen by a
    Python branch. That is what makes the limits free: the branches not taken
    emit no code, so `nl(x, 1., LINEAR_C)` compiles down to a copy. Passing them
    as jit arguments instead raises TracerBoolConversionError; bind them with
    `train.nl_transform` and close over the result.
    """
    if c == jnp.inf:                            # linear regime
        return x/a
    if a == jnp.inf:                            # log regime
        return jnp.sign(x) * jnp.log1p(jnp.abs(x)/c)
    u = x/a
    k = c/a
    stretch = k + jnp.log(-jnp.expm1(-(k + jnp.abs(u)))) - jnp.log(jnp.expm1(k))
    return u + jnp.sign(x) * stretch
