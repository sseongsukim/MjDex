import time

import gymnasium as gym
import numpy as np


class FlattenObsWrapper(gym.ObservationWrapper):
    """Concatenate a Dict observation space into a single flat Box."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        spaces = env.observation_space.spaces
        self._obs_keys = list(spaces.keys())
        low = np.concatenate([spaces[k].low.flatten() for k in self._obs_keys])
        high = np.concatenate([spaces[k].high.flatten() for k in self._obs_keys])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).flatten() for k in self._obs_keys]
        )


class ActionNormalizeEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.total_timesteps = 0

        self._action_min = env.action_space.low.copy()
        self._action_max = env.action_space.high.copy()

        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is None:
            return obs
        return np.clip((obs - self._obs_mean) / self._obs_std, -5.0, 5.0).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        ob, info = super().reset(seed=seed, options=options)
        self._total_reward = 0.0
        self._episode_length = 0
        self.start_time = time.time()
        return self.normalize_obs(ob), info

    def normalize_action(self, action: np.ndarray):
        return (
            2.0 * (action - self._action_min) / (self._action_max - self._action_min)
            - 1.0
        )

    def unnormalize_action(self, action: np.ndarray):
        return (action + 1.0) / 2.0 * (
            self._action_max - self._action_min
        ) + self._action_min

    def step(self, action: np.ndarray):
        action = self.unnormalize_action(action)
        info = dict()
        ob, reward, terminated, truncated, _ = self.env.step(action)

        done = np.logical_or(terminated, truncated)

        self._total_reward += reward
        self._episode_length += 1
        self.total_timesteps += 1

        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["total_rewards"] = self._total_reward
            info["final_reward"] = reward
            info["episode_length"] = self._episode_length
            info["duration"] = time.time() - self.start_time

        return self.normalize_obs(ob), reward, terminated, truncated, info


class WarpNormalizeEnv:
    """Batched counterpart of ``FlattenObsWrapper`` plus ``ActionNormalizeEnv``.

    ``WarpVectorEnv`` hands back a dict of ``(num_envs, ...)`` device arrays and
    expects raw actions, while a policy trained through :func:`create_env` sees a
    flat normalized vector and emits actions in ``[-1, 1]``. This applies the
    same flattening order and the same dataset statistics to the whole batch, so
    a checkpoint can be evaluated on the GPU without retraining.

    Every world steps together and none of them auto-resets, so ``reset`` starts
    a fresh batch of episodes and ``step`` advances all of them in lockstep.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.num_envs = env.num_envs
        # FlattenObsWrapper orders by observation_space, which is not the order
        # the environment happens to build its observation dict in.
        spaces = env.single_observation_space.spaces
        self._obs_keys = list(spaces.keys())
        low = np.concatenate([spaces[k].low.flatten() for k in self._obs_keys])
        high = np.concatenate([spaces[k].high.flatten() for k in self._obs_keys])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = env.single_action_space

        self._action_min = self.action_space.low.copy()
        self._action_max = self.action_space.high.copy()
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None
        self.total_timesteps = 0

    def flatten_obs(self, obs: dict) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs[k], dtype=np.float32).reshape(self.num_envs, -1)
                for k in self._obs_keys
            ],
            axis=-1,
        )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is None:
            return obs
        return np.clip((obs - self._obs_mean) / self._obs_std, -5.0, 5.0).astype(
            np.float32
        )

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (
            2.0 * (action - self._action_min) / (self._action_max - self._action_min)
            - 1.0
        )

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action + 1.0) / 2.0 * (
            self._action_max - self._action_min
        ) + self._action_min

    def reset(self, *, seed=None, options=None, mask=None):
        obs, info = self.env.reset(seed=seed, options=options, mask=mask)
        return self.normalize_obs(self.flatten_obs(obs)), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(
            self.unnormalize_action(np.asarray(action, dtype=np.float32))
        )
        self.total_timesteps += self.num_envs
        return (
            self.normalize_obs(self.flatten_obs(obs)),
            np.asarray(reward, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            info,
        )

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()


def create_warp_env(
    env_name: str,
    num_envs: int,
    dataset_dir=None,
    ob_normalize: bool = True,
    seed: int = 0,
    **warp_kwargs,
):
    """Build a GPU batch of ``env_name`` normalized like :func:`create_env`.

    The statistics come from the same training dataset the policy was trained
    on, so ``dataset_dir`` must point at the dataset used for that checkpoint.
    """
    import pickle

    from mjdex.mujoco_warp import make_warp_env

    env = WarpNormalizeEnv(make_warp_env(env_name, num_envs, seed=seed, **warp_kwargs))

    if dataset_dir is None:
        return env

    with open(dataset_dir + ".pkl", "rb") as f:
        train_dataset = pickle.load(f)

    observations = train_dataset["observations"].astype(np.float32)
    expected_obs_dim = env.observation_space.shape[0]
    if observations.shape[1] != expected_obs_dim:
        raise ValueError(
            f"Obs dim mismatch: dataset has {observations.shape[1]} but env expects "
            f"{expected_obs_dim}. Rebuild the dataset with the correct OBS_KEYS order."
        )

    # create_env leaves joint-position tasks on the model's own action bounds.
    if "jp" not in env_name:
        env._action_min = train_dataset["actions"].min(axis=0).astype(np.float32)
        env._action_max = train_dataset["actions"].max(axis=0).astype(np.float32)

    if ob_normalize:
        env._obs_mean = observations.mean(axis=0).astype(np.float32)
        env._obs_std = observations.std(axis=0).clip(0.1).astype(np.float32)

    return env


def create_env(
    env_name: str,
    dataset_dir=None,
    ob_normalize: bool = True,
):
    from mjdex import register_mjdex_envs
    import pickle

    register_mjdex_envs()
    env = ActionNormalizeEnv(FlattenObsWrapper(gym.make(env_name)))
    eval_env = ActionNormalizeEnv(FlattenObsWrapper(gym.make(env_name)))

    if dataset_dir is not None:
        with open(dataset_dir + ".pkl", "rb") as f:
            train_dataset = pickle.load(f)

        with open(dataset_dir + "-val.pkl", "rb") as f:
            val_dataset = pickle.load(f)

        train_dataset["observations"] = train_dataset["observations"].astype(np.float32)
        val_dataset["observations"] = val_dataset["observations"].astype(np.float32)

        expected_obs_dim = env.observation_space.shape[0]
        actual_obs_dim = train_dataset["observations"].shape[1]
        assert actual_obs_dim == expected_obs_dim, (
            f"Obs dim mismatch: dataset has {actual_obs_dim} but env expects {expected_obs_dim}. "
            "Rebuild the dataset with the correct OBS_KEYS order."
        )

        if "jp" not in env_name:
            env._action_min = train_dataset["actions"].min(axis=0).astype(np.float32)
            env._action_max = train_dataset["actions"].max(axis=0).astype(np.float32)
            eval_env._action_min = env._action_min.copy()
            eval_env._action_max = env._action_max.copy()

        train_dataset["actions"] = env.normalize_action(train_dataset["actions"])
        val_dataset["actions"] = env.normalize_action(val_dataset["actions"])

        if ob_normalize:
            obs_mean = train_dataset["observations"].mean(axis=0).astype(np.float32)
            obs_std = (
                train_dataset["observations"].std(axis=0).clip(0.1).astype(np.float32)
            )
            env._obs_mean = obs_mean
            env._obs_std = obs_std
            eval_env._obs_mean = obs_mean.copy()
            eval_env._obs_std = obs_std.copy()

            train_dataset["observations"] = np.clip(
                (train_dataset["observations"] - obs_mean) / obs_std, -5.0, 5.0
            ).astype(np.float32)
            val_dataset["observations"] = np.clip(
                (val_dataset["observations"] - obs_mean) / obs_std, -5.0, 5.0
            ).astype(np.float32)

        return env, eval_env, train_dataset, val_dataset

    return env, eval_env
