import jax
import jax.numpy as jnp
import optax as opt
import equinox as eqx

from dataclasses import dataclass

"""
Sliced score matching
"""

class SSMPipeline(eqx.Module):

    def __init__(self, score_net, sigma):
        self.score_net = score_net
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        """
        Generate a sample with Langevin dynamics and scorenet
        :param args:
        :param kwargs:
        :return:
        """
        pass

@dataclass
class SSMTrainingConfig:
    n_slice_estimator = 64
    n_batch = 32
    epochs = 500
    x_shape = 5

def SSM_step(score, x, v):
    # pushforward mapping
    s_x, grad_s_v = jax.jvp(score, (x,), (v,))

    # score dot
    J = 0.5 * jnp.sum(s_x * s_x)

    # score Jacobian
    J += jnp.sum(v * grad_s_v)

    return J

def SSM_sample(score, x, config, key):

    key, subkey = jax.random.split(key, 2)

    # generate v batch for slicing
    v = jax.random.normal(
        shape=(config.n_slice_estimator, config.x_shape),
        key=subkey
    )

    # apply batched functional
    ssm_out = jax.vmap(SSM_step, in_axes=(None, None, 0))(score, x, v)

    # average
    estimator = jnp.mean(ssm_out)

    return estimator

def SSM_Fisher(score, x, config, key):
    samples = jax.vmap(SSM_sample, in_axes=(None, 0, None, None))(score, x, config.n_slice_estimator, key)
    return jnp.mean(samples)

def train_ssm(score_net, config, optimizer, key):
    pass

if __name__=="__main__":
    config = SSMTrainingConfig()
    #pipeline = SSMPipeline()

