import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from dataclasses import dataclass

def SSM_estimator(score, x, v):
    # hutchinson
    s_x, grad_s_v = jax.jvp(score, (x,), (v,))

    # score dot
    J = 0.5 * jnp.sum(s_x * s_x)

    # score Jacobian
    J += jnp.sum(v * grad_s_v)

    return J

def SSM_sample(score, x, n_slice, key):

    key, subkey = jax.random.split(key, 2)

    # generate v batch for slicing
    v = jax.random.normal(
        shape=(n_slice, x.shape[0]),
        key=subkey
    )

    # apply batched functional
    ssm_out = jax.vmap(SSM_estimator, in_axes=(None, None, 0))(score, x, v)

    # average
    estimator = jnp.mean(ssm_out)

    return estimator

def SSM_objective(score, x, n_slice, key):
    samples = jax.vmap(SSM_sample, in_axes=(None, 0, None, None))(score, x, n_slice, key)
    return jnp.mean(samples)

def train_ssm(data, score, epochs, lr, key, n_x_batch=128, n_v_batch=128):
    n_data = len(data)
    n_batch = int(n_data / n_x_batch)
    # MC gradients
    divergence_and_grad = eqx.filter_value_and_grad(SSM_objective)

    # optimizer
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(score, eqx.is_array))

    # define step
    def opt_step(score, x_batch, n_v_batch, opt_state, key):

        # compute grad of energy and fisher divergence
        divergence, grad = divergence_and_grad(score, x_batch, n_slice=n_v_batch, key=key)

        # update parameters
        new_params, opt_state = opt.update(grad, opt_state, score)

        # new energy net
        energy = eqx.apply_updates(score, new_params)

        return energy, opt_state, divergence

    # train loop
    divs = []
    for iter in range(epochs):
        for b in range(n_batch):
            key = jax.random.PRNGKey(2000 * iter + 16*b + 67)

            x_batch = data[b*n_x_batch:(b+1)*n_x_batch]

            energy, opt_state, divergence = opt_step(score, x_batch, n_v_batch, opt_state, key)
            divs.append(divergence)

        if iter % 10 == 0:
            print(iter, divergence)

if __name__=="__main__":

    key = jax.random.PRNGKey(2024)
    key, subkey = jax.random.split(key, 2)

    from data import mixture_of_gaussians
    data = mixture_of_gaussians(256, 2, (1/5, 4/5), (0.5, -0.5), (0.2, 0.2), key)

    from mlp import MLP
    layers = [2, 32, 32, 2]
    score = MLP(layer_sizes=layers, key=key)

    train_ssm(data, score, epochs=50, lr=3e-4, key=key)
