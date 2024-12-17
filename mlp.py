import jax
import equinox as eqx
from typing import List


class MLP(eqx.Module):
    layers: List

    def __init__(self, layer_sizes, key):

        self.layers = []

        for (dim_in, dim_out) in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(dim_in, dim_out, use_bias=True, key=subkey)
            )

    def __call__(self, x):

        x = self.layers[0](x)

        for L in self.layers[1:-1]:
            if L.in_features == L.out_features:
                x = x + jax.nn.relu(L(x))
            else:
                x = jax.nn.relu(L(x))
        x = self.layers[-1](x)

        return x[0]