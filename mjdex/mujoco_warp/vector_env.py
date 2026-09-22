"""Fixed-size GPU batches of MjDex's existing MuJoCo scenes."""

from __future__ import annotations

import importlib
import operator
import os
import sys
from dataclasses import dataclass
from typing import ClassVar

# Warp's CUDA graphs need memory outside XLA's allocation pool. Set before the
# first JAX operation; callers that initialize JAX earlier should set this too.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gymnasium as gym
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import mujoco_warp  # noqa: F401

    # The 3.12 MJX wheel references a vendored Warp namespace absent from the
    # PyPI wheel. Same narrow compatibility shim used by reference/DualDex.
    for suffix in ("", "._src", "._src.jax", "._src.jax.ffi"):
        sys.modules.setdefault(
            "mujoco.mjx.third_party.warp" + suffix,
            importlib.import_module("warp" + suffix),
        )
    from mujoco import mjx

    # If an application imported MJX first, its optional Warp type imports
    # already fell back to placeholders. Repair those two aliases as well.
    if mjx.warp.types.GraphMode is int:
        mjx.warp.types.GraphMode = sys.modules["warp._src.jax.ffi"].JaxCallableGraphMode
        mjx.warp.types.Callback = mjx.warp.mjwp_types.Callback
except ImportError as exc:
    raise ImportError(
        "Install the optional GPU backend with: pip install -e '.[warp]'"
    ) from exc

import mujoco

from mjdex.dual_arm import DualArmHandEnv
from mjdex.envs import ENV_MAX_STEPS, ENV_SPECS
from mjdex.single_arm import SingleArmHandEnv, SingleArmJointEnv

from .controller import BatchedDiffIK
from .math import mat_to_quat, normalize_quat
from .tasks import SUPPORTED_TASKS, task_observation, task_result

_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "time",
    "qacc_warmstart",
    "eq_active",
    "userdata",
)
_MODEL_FIELDS = ("body_pos", "body_quat")


@dataclass
class _Arm:
    prefix: str
    qpos: np.ndarray
    dof: np.ndarray
    actuators: np.ndarray
    tool_actuators: np.ndarray
    pads: tuple[int, int]
    site: int
    ik: BatchedDiffIK | None
    hand_qpos: np.ndarray | None = None
    hand_dof: np.ndarray | None = None


class WarpVectorEnv(gym.vector.VectorEnv):
    """JAX arrays with leading ``num_envs`` dimension and explicit reset masks.

    The original CPU environment compiles the scene and supplies fresh resets.
    ``reset_pool_size > 0`` opts into a finite GPU-resident pool sampled on reset.
    All control, physics, observations and rewards in ``step`` run on the GPU.
    Completed worlds are not automatically reset: store their final transition,
    then call ``reset(mask=terminated | truncated)``.
    """

    metadata: ClassVar[dict] = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
        "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
    }

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        *,
        seed: int = 0,
        device: str | None = None,
        reset_pool_size: int = 0,
        naconmax: int | None = None,
        njmax: int = 1024,
        return_numpy: bool = False,
        **env_kwargs,
    ):
        self.num_envs = operator.index(num_envs)
        reset_pool_size = operator.index(reset_pool_size)
        if self.num_envs < 1 or reset_pool_size < 0:
            raise ValueError(
                "num_envs must be positive and reset_pool_size nonnegative."
            )
        if env_id not in ENV_SPECS:
            raise ValueError(
                f"Unknown MjDex environment {env_id!r}; choose from {sorted(ENV_SPECS)}"
            )
        if naconmax is not None and naconmax < 1 or njmax < 1:
            raise ValueError("Contact and constraint capacities must be positive.")
        devices = [d for d in jax.devices() if d.platform == "gpu"]
        if device is not None:
            devices = [
                d for d in devices if device in (str(d), "gpu", "cuda", f"cuda:{d.id}")
            ]
        if not devices:
            raise RuntimeError(
                f"MuJoCo Warp requires a JAX CUDA device; requested {device!r}, available {jax.devices()}."
            )
        self.device = devices[0]
        self.env_id = env_id
        self.return_numpy = return_numpy
        self.closed = False
        self._env = ENV_SPECS[env_id](**env_kwargs)
        self._max_episode_steps = env_kwargs.get(
            "max_episode_steps",
            min(self._env._max_episode_steps, ENV_MAX_STEPS[env_id]),
        )
        self._env.reset(seed=seed)
        if type(self._env.task) not in SUPPORTED_TASKS:
            self._env.close()
            raise NotImplementedError(
                "This task has no Warp observation/reward adapter."
            )
        self.model = self._env.model
        self.render_mode = self._env.render_mode
        self.single_action_space = self._env.action_space
        self.single_observation_space = self._env.observation_space
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self._joint_control = isinstance(self._env, SingleArmJointEnv)
        with jax.default_device(self.device):
            self._cache_arms()
            self.mx = mjx.put_model(self.model, impl="warp", device=self.device)
            template = mjx.make_data(
                self.model,
                impl="warp",
                device=self.device,
                naconmax=naconmax or max(4096, 128 * self.num_envs),
                njmax=njmax,
            )
            self.data = jax.vmap(lambda _: template)(jnp.arange(self.num_envs))
            self._model_fields = {
                k: jnp.broadcast_to(
                    jnp.asarray(getattr(self.model, k)),
                    (self.num_envs,) + getattr(self.model, k).shape,
                )
                for k in _MODEL_FIELDS
            }
            self._episode_steps = jnp.zeros(self.num_envs, dtype=jnp.int32)
            self._pool = None
            # The Warp render context hardcodes nworld and the camera
            # resolution, so it is built on the first render and reused.
            self._render_context = None
            self._render_jit = None
            self._render_size = None
            if reset_pool_size:
                self._pool = self._stack_resets(
                    [seed + i for i in range(reset_pool_size)]
                )
            self._step_jit = jax.jit(self._step_impl)
            self._reset_jit = jax.jit(
                lambda d, fields, steps, selected, mask: jax.lax.cond(
                    jnp.any(mask),
                    lambda: self._reset_impl(d, fields, steps, selected, mask),
                    lambda: (d, fields, steps),
                )
            )
            self._observe_jit = jax.jit(self._observation)
            self._result_jit = jax.jit(self._task_result)
            self.reset(seed=seed)

    def _cache_arms(self):
        env = self._env
        self._arms = []
        if hasattr(env, "_arms"):
            for side, arm in env._arms.items():
                hand_ids = (
                    env._hand_state_ids(arm)
                    if isinstance(env, DualArmHandEnv)
                    else (None, None)
                )
                self._arms.append(
                    _Arm(
                        side + "_",
                        arm.qpos_ids,
                        arm.dof_ids,
                        arm.actuator_ids,
                        arm.gripper_actuator_ids,
                        arm.gripper_pad_body_ids,
                        arm.ee_site_id,
                        BatchedDiffIK(arm.ik, self.device),
                        *hand_ids,
                    )
                )
        else:
            hand_ids = (
                (env._hand_qpos_ids, env._hand_dof_ids)
                if isinstance(env, SingleArmHandEnv)
                else (None, None)
            )
            self._arms.append(
                _Arm(
                    "",
                    env._arm_qpos_ids,
                    env._arm_dof_ids,
                    env._arm_actuator_ids,
                    env._gripper_actuator_ids,
                    env._gripper_pad_body_ids,
                    env._ee_site_id,
                    None
                    if self._joint_control
                    else BatchedDiffIK(env._ik, self.device),
                    *hand_ids,
                )
            )
        groups = [a.dof for a in self._arms] + [
            a.hand_dof for a in self._arms if a.hand_dof is not None
        ]
        self._compensated_dofs = np.unique(np.concatenate(groups))
        self._ctrl_low = jnp.asarray(self.model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jnp.asarray(self.model.actuator_ctrlrange[:, 1])

    def _stack_resets(self, seeds):
        rows = []
        for seed in seeds:
            self._env.reset(seed=int(seed))
            row = {
                k: np.array(getattr(self._env.data, k), copy=True)
                for k in _STATE_FIELDS
            }
            row.update(
                {k: np.array(getattr(self.model, k), copy=True) for k in _MODEL_FIELDS}
            )
            rows.append(row)
        return {
            k: jax.device_put(np.stack([r[k] for r in rows]), self.device)
            for k in rows[0]
        }

    def reset(self, *, seed=None, options=None, mask=None):
        """Reset selected worlds. ``seed + world_index`` seeds fresh CPU resets.

        With a reset pool, seeds select entries from the pool built at creation;
        the pool is deliberately not rebuilt on every reset. Options supports
        Gymnasium's ``reset_mask`` spelling as well as ``mask``.
        """
        self._check_open()
        options = dict(options or {})
        if mask is None:
            mask = options.pop("reset_mask", options.pop("mask", None))
        if options:
            raise ValueError(f"Unknown reset options: {sorted(options)}")
        with jax.default_device(self.device):
            mask = (
                jnp.ones(self.num_envs, dtype=bool)
                if mask is None
                else jnp.asarray(mask, dtype=bool)
            )
            if mask.shape != (self.num_envs,):
                raise ValueError(
                    f"Expected reset mask {(self.num_envs,)}, got {mask.shape}."
                )
            if seed is not None:
                seed = operator.index(seed)
                self._rngs = [
                    np.random.default_rng(seed + i) for i in range(self.num_envs)
                ]
                self._key = jax.random.PRNGKey(seed)
            if self._pool is not None:
                self._key, key = jax.random.split(self._key)
                indices = jax.random.randint(
                    key, (self.num_envs,), 0, self._pool["qpos"].shape[0]
                )
                selected = {k: v[indices] for k, v in self._pool.items()}
            else:
                # Exact reset distributions, including rejection sampling and
                # fixed-body randomization, reuse the CPU task implementation.
                indices = np.flatnonzero(np.asarray(mask))
                if not len(indices):
                    return self._output(
                        (self._observe_jit(self.data), self._reset_info(mask))
                    )
                seeds = [
                    seed + i if seed is not None else self._rngs[i].integers(0, 2**32)
                    for i in indices
                ]
                rows = self._stack_resets(seeds)
                selected = {
                    k: getattr(self.data, k).at[indices].set(rows[k])
                    for k in _STATE_FIELDS
                }
                selected.update(
                    {
                        k: self._model_fields[k].at[indices].set(rows[k])
                        for k in _MODEL_FIELDS
                    }
                )
            self.data, self._model_fields, self._episode_steps = self._reset_jit(
                self.data, self._model_fields, self._episode_steps, selected, mask
            )
            return self._output((self._observe_jit(self.data), self._reset_info(mask)))

    def _reset_impl(self, data, fields, steps, selected, mask):
        def select(old, new):
            return jnp.where(
                mask.reshape((self.num_envs,) + (1,) * (old.ndim - 1)), new, old
            )

        data = data.replace(
            **{k: select(getattr(data, k), selected[k]) for k in _STATE_FIELDS}
        )
        fields = {k: select(fields[k], selected[k]) for k in _MODEL_FIELDS}
        forwarded = self._physics(mjx.forward, data, fields)
        # mj_step leaves poses/bias forces at its last integration stage. A
        # forward on an untouched world would change its next gravity command.
        # Restore its public derived arrays too, not just qpos/qvel/time.
        data = forwarded.replace(
            **{
                f.name: select(getattr(data, f.name), getattr(forwarded, f.name))
                for f in data.fields()
                if f.name != "_impl"
            }
        )
        return data, fields, jnp.where(mask, 0, steps)

    def _reset_info(self, mask):
        _, _, info = self._result_jit(self.data)
        return {**info, "reset_mask": mask, "elapsed_steps": self._episode_steps}

    def _physics(self, function, data, fields):
        def one(d, pos, quat):
            return function(self.mx.replace(body_pos=pos, body_quat=quat), d)

        return jax.vmap(one)(data, fields["body_pos"], fields["body_quat"])

    def step(self, actions):
        self._check_open()
        with jax.default_device(self.device):
            actions = jax.device_put(
                jnp.asarray(actions, dtype=jnp.float32), self.device
            )
            expected = (self.num_envs,) + self.single_action_space.shape
            if actions.shape != expected:
                raise ValueError(
                    f"Expected action shape {expected}, got {actions.shape}."
                )
            self.data, self._episode_steps, result = self._step_jit(
                self.data, self._model_fields, self._episode_steps, actions
            )
        return self._output(result)

    def _step_impl(self, data, fields, steps, actions):
        data = self._apply_action(data, actions)
        data = jax.lax.fori_loop(
            0, self._env._n_steps, lambda _, d: self._physics(mjx.step, d, fields), data
        )
        # Like CPU mj_step, derived poses/contacts correspond to the last
        # integration stage. Do not add a forward that changes that convention.
        steps = steps + 1
        observation = self._observation(data)
        reward, terminated, task_info = self._task_result(data)
        info = self._robot_info(data)
        info.update(task_info)
        info["elapsed_steps"] = steps
        truncated = steps >= self._max_episode_steps
        return (
            data,
            steps,
            (observation, reward.astype(jnp.float32), terminated, truncated, info),
        )

    def _apply_action(self, data, actions):
        offset = 0
        ctrl = data.ctrl
        for arm in self._arms:
            arm_width = len(arm.qpos) if self._joint_control else 7
            tool_width = len(arm.tool_actuators) if arm.hand_qpos is not None else 1
            action = actions[:, offset : offset + arm_width + tool_width]
            offset += arm_width + tool_width
            if self._joint_control:
                target = action[:, :arm_width]
            else:
                position = action[:, :3]
                bounds = getattr(self._env, "workspace_bounds", None)
                if bounds is not None:
                    position = jnp.clip(
                        position, jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
                    )
                target = arm.ik.solve(
                    data.qpos[:, arm.qpos], position, normalize_quat(action[:, 3:7])
                )
            ctrl = ctrl.at[:, arm.actuators].set(
                jnp.clip(
                    target,
                    self._ctrl_low[arm.actuators],
                    self._ctrl_high[arm.actuators],
                )
            )
            low, high = (
                self._ctrl_low[arm.tool_actuators],
                self._ctrl_high[arm.tool_actuators],
            )
            if arm.hand_qpos is not None:
                tool = jnp.clip(action[:, arm_width:], low, high)
            else:
                opening = 0.5 * (
                    1 - jnp.clip(action[:, arm_width : arm_width + 1], -1.0, 1.0)
                )
                tool = low + opening * jnp.maximum(high - low, 1e-6)
            ctrl = ctrl.at[:, arm.tool_actuators].set(tool)
        ids = self._compensated_dofs
        return data.replace(
            ctrl=ctrl,
            qfrc_applied=data.qfrc_applied.at[:, ids].set(data.qfrc_bias[:, ids]),
        )

    def _observation(self, data):
        obs = {}
        for arm in self._arms:
            p = arm.prefix
            obs[p + "arm_joint_position"] = data.qpos[:, arm.qpos]
            obs[p + "arm_joint_velocity"] = data.qvel[:, arm.dof]
            obs[p + "ee_pose"] = self._ee_pose(data, arm)
            if arm.hand_qpos is not None:
                obs[p + "hand_joint_position"] = data.qpos[:, arm.hand_qpos]
                obs[p + "hand_joint_velocity"] = data.qvel[:, arm.hand_dof]
            elif min(arm.pads) >= 0:
                obs[p + "gripper_width"] = self._gripper_width(data, arm)[:, None]
        obs.update(task_observation(self._env.task, data))
        return jax.tree.map(lambda x: x.astype(jnp.float32), obs)

    def _robot_info(self, data):
        info = {"qpos": data.qpos, "qvel": data.qvel}
        for arm in self._arms:
            p = arm.prefix
            info[p + "ee_pose"] = self._ee_pose(data, arm)
            info[p + "arm_qpos"] = data.qpos[:, arm.qpos]
            info[p + "arm_qvel"] = data.qvel[:, arm.dof]
            if min(arm.pads) >= 0:
                info[p + "gripper_distance"] = self._gripper_width(data, arm)
        if len(self._arms) == 1:
            env = self._env
            info["robot_table_contact"] = self._has_contact(
                data, env._robot_geom_ids, env._table_geom_ids
            )
            if self._arms[0].hand_qpos is not None:
                info["hand_qpos"] = data.qpos[:, self._arms[0].hand_qpos]
        return info

    def _task_result(self, data):
        positions = jnp.stack([data.site_xpos[:, a.site] for a in self._arms], axis=1)
        return task_result(self._env.task, data, positions, self._has_contact)

    @staticmethod
    def _ee_pose(data, arm):
        return jnp.concatenate(
            (data.site_xpos[:, arm.site], mat_to_quat(data.site_xmat[:, arm.site])),
            axis=-1,
        )

    @staticmethod
    def _gripper_width(data, arm):
        return jnp.linalg.norm(
            data.xpos[:, arm.pads[0]] - data.xpos[:, arm.pads[1]], axis=-1
        )

    def _has_contact(self, data, geoms_a, geoms_b):
        if not geoms_a or not geoms_b:
            return jnp.zeros(self.num_envs, dtype=bool)
        # MJX-Warp 3.12 uses aggregate contacts tagged by world, not a per-world
        # contact array. Ignore unused rows even if they contain stale IDs.
        geom = data._impl.contact__geom
        world = data._impl.contact__worldid
        valid = jnp.arange(geom.shape[0]) < data._impl.nacon[0]
        a = (
            jnp.zeros(self.model.ngeom, dtype=bool)
            .at[jnp.asarray(sorted(geoms_a))]
            .set(True)
        )
        b = (
            jnp.zeros(self.model.ngeom, dtype=bool)
            .at[jnp.asarray(sorted(geoms_b))]
            .set(True)
        )
        matched = valid & (
            (a[geom[:, 0]] & b[geom[:, 1]]) | (b[geom[:, 0]] & a[geom[:, 1]])
        )
        counts = (
            jnp.zeros(self.num_envs, dtype=jnp.int32)
            .at[jnp.where(valid, world, 0)]
            .add(matched.astype(jnp.int32), mode="drop")
        )
        return counts > 0

    def get_state(self):
        """Return independent batched simulation state (device arrays).

        Body transforms are included because rack resets modify the model.
        Derived arrays and aggregate contacts are recomputed by ``set_state``.
        """
        self._check_open()
        return self._output(
            {
                **{k: getattr(self.data, k) for k in _STATE_FIELDS},
                **self._model_fields,
                "elapsed_steps": self._episode_steps,
            }
        )

    def set_state(self, state, *, mask=None):
        """Restore a batched state from this environment, optionally by world."""
        self._check_open()
        expected = set(_STATE_FIELDS + _MODEL_FIELDS) | {"elapsed_steps"}
        if set(state) != expected:
            raise ValueError(f"State must have keys {sorted(expected)}")
        with jax.default_device(self.device):
            state = {k: jnp.asarray(v) for k, v in state.items()}
            current = {
                **{k: getattr(self.data, k) for k in _STATE_FIELDS},
                **self._model_fields,
                "elapsed_steps": self._episode_steps,
            }
            for k, value in state.items():
                if value.shape != current[k].shape:
                    raise ValueError(
                        f"State {k} must have shape {current[k].shape}, got {value.shape}."
                    )
                state[k] = value.astype(current[k].dtype)
            mask = (
                jnp.ones(self.num_envs, dtype=bool)
                if mask is None
                else jnp.asarray(mask, dtype=bool)
            )
            if mask.shape != (self.num_envs,):
                raise ValueError(f"Expected reset mask {(self.num_envs,)}")
            self.data, self._model_fields, _ = self._reset_jit(
                self.data, self._model_fields, self._episode_steps, state, mask
            )
            self._episode_steps = jnp.where(
                mask, state["elapsed_steps"], self._episode_steps
            )
            return self._output(self._observe_jit(self.data))

    def render(self, world_index=None, *, width=None, height=None):
        """Return an RGB frame for every world, or for one selected world.

        The Warp renderer draws the whole batch in a single GPU pass whose cost
        is dominated by fixed per-call overhead rather than by ``num_envs`` or
        resolution, so rendering everything is the default:
        ``(num_envs, height, width, 3)`` of ``uint8``. ``world_index`` selects
        one frame out of that same pass and returns ``(height, width, 3)``.

        Frames come from the Warp renderer, not the CPU renderer behind
        ``render_cpu``; shading is close but not pixel-identical.
        """
        self._check_open()
        if world_index is not None:
            world_index = operator.index(world_index)
            if not 0 <= world_index < self.num_envs:
                raise IndexError(world_index)
        frames = self._render_batch(width, height)
        return frames if world_index is None else frames[world_index]

    def _render_batch(self, width, height):
        """Render every world and unpack the buffer into ``uint8`` RGB."""
        size = (
            int(width if width is not None else self._env._render_width),
            int(height if height is not None else self._env._render_height),
        )
        if min(size) < 1:
            raise ValueError(f"Render size must be positive, got {size}.")
        with jax.default_device(self.device):
            if self._render_size != size:
                # Skybox and shadows keep the batch close to the CPU renderer
                # and cost nothing measurable in this ray-traced backend.
                # Keep the context itself alive: its device buffers are held in
                # a registry keyed by the handle, and dropping it invalidates
                # every pytree handle taken from it.
                self._render_context = mjx.create_render_context(
                    self.model,
                    nworld=self.num_envs,
                    cam_res=size,
                    render_skybox=True,
                    use_shadows=True,
                )
                handle = self._render_context.pytree()
                self._render_jit = jax.jit(lambda m, d: mjx.render(m, d, handle)[0])
                self._render_size = size
            packed = np.asarray(self._render_jit(self.mx, self.data))
        width, height = size
        packed = packed.reshape(self.num_envs, self.model.ncam, height, width)
        camera = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._env._render_camera_name
        )
        packed = packed[:, max(camera, 0)]
        return np.stack(
            ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF), axis=-1
        ).astype(np.uint8)

    def render_cpu(self, world_index=0):
        """Render one world through the original CPU renderer (host transfer)."""
        self._check_open()
        world_index = operator.index(world_index)
        if not 0 <= world_index < self.num_envs:
            raise IndexError(world_index)
        for k in _MODEL_FIELDS:
            getattr(self.model, k)[:] = np.asarray(self._model_fields[k][world_index])
        for k in _STATE_FIELDS:
            value = np.asarray(getattr(self.data, k)[world_index])
            if value.ndim:
                getattr(self._env.data, k)[:] = value
            else:
                setattr(self._env.data, k, value.item())
        mujoco.mj_forward(self.model, self._env.data)
        return self._env.render()

    def _output(self, value):
        return jax.tree.map(np.asarray, value) if self.return_numpy else value

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Environment is closed.")

    def close_extras(self, **kwargs):
        self._render_context = self._render_jit = self._render_size = None
        self._env.close()


def make_warp_env(env_id, num_envs, **kwargs):
    return WarpVectorEnv(env_id, num_envs, **kwargs)
