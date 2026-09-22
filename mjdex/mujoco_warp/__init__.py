"""Optional JAX-facing MuJoCo Warp environments.

Importing the ordinary MjDex environments does not require the Warp extra.
"""


def make_warp_env(env_id: str, num_envs: int, **kwargs):
    """Build a fixed-size batch of an existing MjDex environment."""
    from .vector_env import WarpVectorEnv

    return WarpVectorEnv(env_id, num_envs, **kwargs)


def __getattr__(name):
    if name == "WarpVectorEnv":
        from .vector_env import WarpVectorEnv

        return WarpVectorEnv
    raise AttributeError(name)


__all__ = ["WarpVectorEnv", "make_warp_env"]
