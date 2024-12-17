import jax
from jax import numpy as jnp

def mixture_of_gaussians(n, dim, proportions, mus, sigmas, key):
    assert jnp.sum(jnp.array(proportions)) == 1

    n_mixtures = len(proportions)

    keys = jax.random.split(key, n_mixtures + 1)
    key, subkeys = keys[0], keys[1:]

    gaussians = [] 
    for p, mu, sigma, subkey in zip(proportions, mus, sigmas, subkeys): 
        samples = mu + sigma * jax.random.normal(subkey, (int(n * p), dim))
        gaussians.append(samples)
    return jnp.concatenate(gaussians, axis=0)

if __name__ == "__main__":
    key = jax.random.PRNGKey(2002)
    data = mixture_of_gaussians(20, 2, (1/5, 4/5), (0.5, -0.5), (0.2, 0.2), key)
    print(data.shape)