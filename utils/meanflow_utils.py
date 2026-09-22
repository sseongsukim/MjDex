import flax
import jax
import jax.numpy as jnp


def adaptive_l2_loss(error, valid=None, p=0.5, c=1e-3):
    """Adaptive L2 loss used by mean-flow policies.

    If `valid` is provided, it masks invalid action-chunk steps from DualDex's
    sequence datasets.
    """
    squared_error = jnp.mean(jnp.square(error), axis=-1)
    weight = jax.lax.stop_gradient(1.0 / (squared_error + c) ** p)

    if valid is None:
        return jnp.mean(weight * squared_error)

    valid = valid.astype(squared_error.dtype)
    return jnp.sum(weight * squared_error * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def sample_t_r(batch_size, rng, flow_ratio=0.25):
    rng, t_rng, r_rng = jax.random.split(rng, 3)
    t = jax.nn.sigmoid(jax.random.normal(t_rng, (batch_size, 1)) - 0.4)
    r = jax.nn.sigmoid(jax.random.normal(r_rng, (batch_size, 1)) - 0.4)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)

    data_size = int(batch_size * (1.0 - flow_ratio))
    zero_mask = (jnp.arange(batch_size) < data_size).reshape(batch_size, 1)
    return t, jnp.where(zero_mask, t, r)


def sample_latent_dist(rng, sample_shape, latent_dist="sphere"):
    if latent_dist == "normal":
        return jax.random.normal(rng, sample_shape)
    if latent_dist == "truncated_normal":
        return jax.random.truncated_normal(rng, -2.0, 2.0, shape=sample_shape)
    if latent_dist == "uniform":
        return jax.random.uniform(rng, sample_shape, minval=-1.0, maxval=1.0)
    if latent_dist == "sphere":
        e = jax.random.normal(rng, sample_shape)
        norm = jnp.sqrt(jnp.sum(jnp.square(e), axis=-1, keepdims=True) + 1e-6)
        return e / norm * jnp.sqrt(sample_shape[-1])
    if latent_dist == "sphere_plus":
        action_dim = sample_shape[-1]
        e = jax.random.normal(rng, sample_shape[:-1] + (action_dim + 1,))
        norm = jnp.linalg.norm(e, axis=-1, keepdims=True)
        return e[..., :-1] / (norm + 1e-6)
    raise ValueError(f"Unsupported latent distribution: {latent_dist}")


def get_batch_shape(observations, ob_dims):
    if isinstance(observations, (dict, flax.core.FrozenDict)):
        leaf = jax.tree_util.tree_leaves(observations)[0]
        leaf_ob_dims = leaf.shape[-len(ob_dims) :] if len(ob_dims) > 0 else ()
        return leaf.shape[: leaf.ndim - len(leaf_ob_dims)]
    return observations.shape[: observations.ndim - len(ob_dims)]
