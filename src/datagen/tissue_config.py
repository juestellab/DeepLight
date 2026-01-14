import jax.numpy as jnp


def softplus(x):
    return x + jnp.log(1 + jnp.exp(-x))


skinline_params = dict(
    epidermis_params={"mean": 0.01, "std": 0.002},
    dermis_params={"mean": 0.05, "std": 0.05}
)

absorption_params = dict(
    coupling=dict(
        offset=dict(mean=-1.5, std=(0.5, 0.05)),
        fluctuations=(0.01, 0.001),
        loglogavgslope=(-4, 0.1),
        nl=lambda x: jnp.maximum(jnp.pow(10, x), 1e-3)
    ),
    epidermis=dict(
        offset=dict(mean=10, std=(5, 0.5)),
        fluctuations=(0.3, 0.1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: jnp.log(1 + jnp.exp(x))
    ),
    dermis=dict(
        offset=dict(mean=0.15, std=(0.06, 0.02)),
        fluctuations=(0.075, 0.02),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: jnp.maximum(jnp.log(1 + jnp.exp(x*100))/100, 0.05)
    ),
    tissue=dict(
        offset=dict(mean=-1.5, std=(0.4, 0.1)),
        fluctuations=(1.2, 0.2),
        loglogavgslope=(-2.3, 0.1),
        nl=lambda x: jnp.maximum(jnp.minimum(jnp.log(1 + jnp.exp(x)), 1.), 0.02)
    ),
    vessels=dict(
        offset=dict(mean=-250, std=(10, 5)),
        fluctuations=(200, 30),
        loglogavgslope=(-1.9, 0.15),
        nl=lambda x: jnp.log(1 + x + jnp.sqrt(1 + x**2)),
        alpha=(0.4, 0.15)
    )
)

scattering_params = dict(
    coupling=dict(
        offset=dict(mean=10, std=(1, 0.5)),
        fluctuations=(1, 0.1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x + jnp.log(1 + jnp.exp(-x))
    ),
    epidermis=dict(
        offset=dict(mean=50, std=(10, 3)),
        fluctuations=(5, 1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x + jnp.log(1 + jnp.exp(-x))
    ),
    dermis=dict(
        offset=dict(mean=50, std=(10, 3)),
        fluctuations=(5, 1),
        loglogavgslope=(-3, 0.1),
        nl=softplus
    ),
    tissue=dict(
        offset=dict(mean=260, std=(40, 10)),
        fluctuations=(40, 10),
        loglogavgslope=(-2.5, 0.1),
        nl=lambda x: jnp.maximum(softplus(x), 120)
    ),
    vessels=None
)


absorption_tangent_params = dict(
    coupling=dict(
        offset=dict(mean=0, std=(0.01, 0.005)),
        fluctuations=(0.01, 0.001),
        loglogavgslope=(-4, 0.1),
        nl=lambda x: x
    ),
    epidermis=dict(
        offset=dict(mean=0, std=(5, 0.5)),
        fluctuations=(0.3, 0.1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    dermis=dict(
        offset=dict(mean=0, std=(0.06, 0.02)),
        fluctuations=(0.075, 0.02),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    tissue=dict(
        offset=dict(mean=0, std=(0.4, 0.1)),
        fluctuations=(1.2, 0.2),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    vessels=dict(
        offset=dict(mean=-250, std=(10, 5)),
        fluctuations=(200, 30),
        loglogavgslope=(-1.9, 0.15),
        nl=lambda x: jnp.log(1 + x + jnp.sqrt(1 + x**2)),
        alpha=(2, 0.2)
    )
)

scattering_tangent_params = dict(
    coupling=dict(
        offset=dict(mean=0, std=(1, 0.5)),
        fluctuations=(1, 0.1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    epidermis=dict(
        offset=dict(mean=0, std=(10, 3)),
        fluctuations=(5, 1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    dermis=dict(
        offset=dict(mean=0, std=(10, 3)),
        fluctuations=(5, 1),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    tissue=dict(
        offset=dict(mean=0, std=(40, 10)),
        fluctuations=(40, 10),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
    ),
    vessels=None
)


correlated_absorption = dict(
        offset=dict(mean=0., std=(0.02, 0.01)),
        fluctuations=(0.2, 0.05),
        loglogavgslope=(-2.5, 0.1),
        nl=lambda x: jnp.maximum(jnp.log(1 + jnp.exp(10*x))/10, 0.001)
)

correlated_scattering = dict(
        offset=dict(mean=260, std=(40, 10)),
        fluctuations=(40, 10),
        loglogavgslope=(-3, 0.1),
        nl=softplus
)


correlated_absorption_tangent = dict(
        offset=dict(mean=0, std=(0.1, 0.02)),
        fluctuations=(0.2, 0.1),
        loglogavgslope=(-2.5, 0.1),
        nl=lambda x: x
)

correlated_scattering_tangent = dict(
        offset=dict(mean=0, std=(40, 10)),
        fluctuations=(40, 10),
        loglogavgslope=(-3, 0.1),
        nl=lambda x: x
)


params = dict(
    tissue_sample={"absorption": absorption_params,
                   "scattering": scattering_params,
                   "v_absorption": absorption_tangent_params,
                   "v_scattering": scattering_tangent_params},
    random_tissue_sample={"absorption": dict(coupling=absorption_params["coupling"],
                                             correlated=correlated_absorption),
                          "scattering": dict(coupling=scattering_params["coupling"],
                                             correlated=correlated_scattering),
                          "v_absorption": dict(coupling=absorption_tangent_params["coupling"],
                                               correlated=correlated_absorption_tangent),
                          "v_scattering": dict(coupling=scattering_tangent_params["coupling"],
                                               correlated=correlated_scattering_tangent)},
    random_sample={"absorption": correlated_absorption,
                   "scattering": correlated_scattering,
                   "v_absorption": correlated_absorption_tangent,
                   "v_scattering": correlated_scattering_tangent}
)
