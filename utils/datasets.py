from functools import partial
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = (
            False  # Whether to additionally return next actions; set outside the class.
        )

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = (
            idxs[:, None] + np.arange(sequence_length)[None, :]
        )  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()

        # Batch fetch data to avoid loops
        batch_observations = self["observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["observations"].shape[1:]
        )
        batch_next_observations = self["next_observations"][all_idxs].reshape(
            batch_size, sequence_length, *self["next_observations"].shape[1:]
        )
        batch_actions = self["actions"][all_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )
        batch_rewards = self["rewards"][all_idxs].reshape(
            batch_size, sequence_length, *self["rewards"].shape[1:]
        )
        batch_masks = self["masks"][all_idxs].reshape(
            batch_size, sequence_length, *self["masks"].shape[1:]
        )
        batch_terminals = self["terminals"][all_idxs].reshape(
            batch_size, sequence_length, *self["terminals"].shape[1:]
        )

        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self["actions"][next_action_idxs].reshape(
            batch_size, sequence_length, *self["actions"].shape[1:]
        )

        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = (
                rewards[:, i - 1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            )
            masks[:, i] = np.minimum(masks[:, i - 1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(
                terminals[:, i - 1], batch_terminals[:, i].squeeze()
            )
            valid[:, i] = 1.0 - terminals[:, i - 1]

        # Reorganize observations data format - maintain the exact same shape as the original function
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            # Transpose to (batch, h, w, seq, c) format, consistent with the original function
            observations = batch_observations.transpose(
                0, 2, 3, 1, 4
            )  # (batch_size, h, w, sequence_length, c)
            next_observations = batch_next_observations.transpose(
                0, 2, 3, 1, 4
            )  # (batch_size, h, w, sequence_length, c)
        else:  # State data: maintain (batch, seq, state_dim) shape
            observations = (
                batch_observations  # (batch_size, sequence_length, state_dim)
            )
            next_observations = (
                batch_next_observations  # (batch_size, sequence_length, state_dim)
            )

        # Maintain the 3D shape of actions and next_actions, consistent with the original function
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)

        return dict(
            observations=data["observations"].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )


class QCDataset(Dataset):
    """Dataset class that exposes sequence sampling through `sample`.

    This is useful for Q-learning with action chunks, where the agent receives a horizon of actions and uses the last
    transition in the sampled sequence for the TD target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_horizon = None
        self.discount = None

    def sample(self, batch_size: int):
        """Sample a horizon-length sequence batch."""
        return self.sample_sequence(
            batch_size=batch_size,
            sequence_length=self.pred_horizon,
            discount=self.discount,
        )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


def add_history(dataset, history_length):

    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset["terminals"] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[
        np.searchsorted(initial_locs, idxs, side="right") - 1
    ]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(
            jax.tree_util.tree_map(
                lambda arr: arr[cur_idxs] * (~outside)
                + jnp.zeros_like(arr[cur_idxs]) * outside,
                dataset["observations"],
            )
        )
        acts_rets.append(
            jax.tree_util.tree_map(
                lambda arr: arr[cur_idxs] * (~outside)
                + jnp.zeros_like(arr[cur_idxs]) * outside,
                dataset["actions"],
            )
        )
    observation_history, action_history = jax.tree_util.tree_map(
        lambda *args: np.stack(args, axis=-2), *obs_rets
    ), jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *acts_rets)

    dataset = Dataset(
        dataset.copy(
            dict(observation_history=observation_history, action_history=action_history)
        )
    )

    return dataset


class MultistepDataset(FrozenDict):
    """Dataset class with optional multi-step action sampling."""

    @classmethod
    def create(cls, freeze=True, **fields):
        assert "observations" in fields and "actions" in fields
        assert "episode_ends" in fields or "terminals" in fields

        N = len(fields["observations"])
        if "episode_ends" in fields:
            terminals = np.zeros(N, dtype=bool)
            terminals[fields["episode_ends"]] = True
        else:
            terminals = np.asarray(fields["terminals"], dtype=bool)

        data = {
            "observations": fields["observations"],
            "actions": fields["actions"],
            "terminals": terminals,
        }
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        self.pred_horizon = None

        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def _compute_episode_ends(self, idxs):
        """Return episode end index for each idx."""
        ep_indices = np.searchsorted(self.terminal_locs, idxs, side="left")
        ep_ends = np.where(
            ep_indices < len(self.terminal_locs),
            self.terminal_locs[ep_indices],
            self.size - 1,
        )
        return ep_ends

    def _sample_multistep_actions(self, idxs):
        """
        Return (B, H, act_dim) action chunks with terminal padding.
        """
        B = len(idxs)
        H = self.pred_horizon

        ep_ends = self._compute_episode_ends(idxs)
        offsets = np.arange(H)  # (H,)
        candidate_idxs = idxs[:, None] + offsets[None, :]  # (B, H)
        action_idxs = np.minimum(candidate_idxs, ep_ends[:, None])

        return self._dict["actions"][action_idxs]

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)

        batch = self.get_subset(idxs)

        if self.pred_horizon is not None:
            batch["actions"] = self._sample_multistep_actions(idxs)

        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations"])

        return batch

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        padding = 4
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )

        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )


@dataclasses.dataclass
class ACDataset:
    """Dataset class for action chunking.

    This class provides methods to sample transition batches for action chunking. The returned batch contains
    observations, action chunks, next observations, rewards, and masks.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.chunk_next_obs = (
            self.config["chunk_next_obs"] if "chunk_next_obs" in self.config else False
        )

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        valid_idxs = []
        for i in range(len(self.initial_locs)):
            valid_idxs.append(
                np.arange(
                    self.initial_locs[i],
                    self.terminal_locs[i] - self.config["action_chunking"],
                )
            )
        self.valid_idxs = np.concatenate(valid_idxs)
        assert self.terminal_locs[-1] == self.size - 1

    def sample_consecutive(self, batch_size: int, batch_length: int):
        """Sample a consecutive batch of transitions.

        Args:
            batch_size: Batch size.
            batch_length: Number of consecutive chunks to sample.
        """
        valid_idxs = self.valid_idxs[
            self.valid_idxs + self.config["action_chunking"] * batch_length
            < self.terminal_locs[-1]
        ]
        x = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)]
        y = self.terminal_locs[
            np.searchsorted(
                self.terminal_locs,
                valid_idxs + self.config["action_chunking"] * batch_length,
            )
        ]
        new_valid_idxs = valid_idxs[x == y]

        idxs = np.random.choice(new_valid_idxs, size=batch_size)

        batch_idxs = np.stack(
            [idxs + i for i in range(batch_length * self.config["action_chunking"])],
            axis=0,
        )
        _batch = self.dataset.sample(0, batch_idxs)
        _batch = {
            k: np.swapaxes(
                v.reshape(
                    batch_size,
                    batch_length,
                    self.config["action_chunking"],
                    *v.shape[2:],
                ),
                0,
                1,
            )
            for k, v in _batch.items()
        }

        batch = {
            "observations": _batch["observations"][:, :, 0],
            "actions": np.swapaxes(
                np.concatenate(np.swapaxes(_batch["actions"], 0, 2), axis=-1), 0, 1
            ),
        }
        if batch["observations"].dtype == np.uint8:
            batch["observations"] = batch["observations"].astype(np.float32) / 255.0

        return batch

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = np.random.choice(self.valid_idxs, size=batch_size)

        ac_actor = (
            self.config["action_chunking_actor"]
            if "action_chunking_actor" in self.config
            and self.config["action_chunking_actor"] > 0
            else self.config["action_chunking"]
        )
        batch_idxs = np.stack(
            [idxs + i for i in range(self.config["action_chunking"] + 1)], axis=0
        )
        _batch = self.dataset.sample(0, batch_idxs)

        batch = {
            "observations": _batch["observations"][0],
            "actions": np.concatenate(_batch["actions"][:ac_actor], axis=-1),
            "next_observations": (
                np.concatenate(_batch["observations"][1:], axis=-1)
                if self.chunk_next_obs
                else _batch["observations"][-1]
            ),
        }

        rewards = _batch["rewards"][: self.config["action_chunking"]]
        discount_powers = self.config["discount"] ** np.arange(
            self.config["action_chunking"]
        )
        discount_shape = (self.config["action_chunking"],) + (1,) * (rewards.ndim - 1)
        chunk_rewards = np.sum(
            rewards * discount_powers.reshape(discount_shape), axis=0
        )
        chunk_masks = np.min(_batch["masks"][: self.config["action_chunking"]], axis=0)
        if chunk_rewards.ndim > 1 and chunk_rewards.shape[-1] == 1:
            chunk_rewards = chunk_rewards.squeeze(-1)
        if chunk_masks.ndim > 1 and chunk_masks.shape[-1] == 1:
            chunk_masks = chunk_masks.squeeze(-1)
        batch["rewards"] = chunk_rewards
        batch["masks"] = chunk_masks
        if batch["observations"].dtype == np.uint8:
            batch["observations"] = batch["observations"].astype(np.float32) / 255.0

        return batch
