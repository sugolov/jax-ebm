import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from dataclasses import dataclass


def Langevin_mc(score, x0, steps, eps):
    samples = [x0]
    x_curr = x0

    for i in range(steps):
        key = jax.random.PRNGKey(2024 * i + 16 * i**2 + 13)

        # sample noise
        z = jax.random.normal(key, shape=x0.shape)

        # Langevin step
        x_new = x_curr + eps * score(x_curr) / 2 + eps**0.5 * z
        samples.append(x_new)
    return x_new, samples