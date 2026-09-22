"""Shared MuJoCo environment scaffolding for MjDex tasks.

This module keeps task environments small: subclasses build or point to an MJCF
model, define observations/rewards, and optionally customize reset/control
logic. The base class handles compilation, reset/step bookkeeping, rendering,
and passive viewer integration.
"""

from __future__ import annotations

import abc
import contextlib
from pathlib import Path
from typing import Any, Callable, Optional, SupportsFloat

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf

from mjdex import contact as contact_utils
from mjdex.robots import RENDER_CAMERA_NAME


class MjDexEnv(gym.Env, abc.ABC):
    """Base class for MjDex MuJoCo environments.

    Subclasses should implement either :meth:`build_mjcf` or
    :meth:`build_xml_path`, plus observation and reward hooks. The default
    action space mirrors the compiled actuator control ranges.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.02,
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

        self._mjcf_model: Optional[mjcf.RootElement] = None
        self._xml_path: Optional[Path] = None
        self._model: Optional[mujoco.MjModel] = None
        self._data: Optional[mujoco.MjData] = None

        self._dirty = True
        self._never_compiled = True
        self._reset_next_step = True

        self._renderer: Optional[mujoco.Renderer] = None
        self._render_width = int(width)
        self._render_height = int(height)
        self._scene_option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._render_camera_name = RENDER_CAMERA_NAME
        self._passive_viewer_handle = None

        self.set_timesteps(
            physics_timestep=float(physics_timestep),
            control_timestep=float(control_timestep),
        )

    def build_mjcf(self) -> Optional[mjcf.RootElement]:
        """Build and return a dm_control MJCF model.

        Override this for programmatic scene construction. Returning ``None``
        tells the base class to use :meth:`build_xml_path` instead.
        """

        return None

    def build_xml_path(self) -> Optional[str | Path]:
        """Return an XML path for static MJCF assets.

        Override this when the task uses a checked-in XML file directly. This is
        convenient for Menagerie-style assets under ``mjdex/assets``.
        """

        return None

    def modify_mjcf(self, mjcf_model: mjcf.RootElement) -> mjcf.RootElement:
        """Modify the MJCF tree before compilation.

        Subclasses can use this hook for model randomization. Call
        :meth:`mark_dirty` if changes require recompilation.
        """

        return mjcf_model

    def initialize_episode(self) -> None:
        """Set initial qpos/qvel/ctrl after each reset."""

    def set_control(self, action: np.ndarray) -> None:
        """Apply a control command to the model actuators."""

        if self._data is None:
            raise ValueError("Call `reset` before applying actions.")
        if self._data.ctrl.size:
            self._data.ctrl[:] = np.asarray(action, dtype=np.float64)

    def pre_step(self) -> None:
        """Hook called after control is set and before physics is stepped."""

    def post_step(self) -> None:
        """Hook called after physics is stepped."""

    @abc.abstractmethod
    def compute_observation(self) -> Any:
        """Return the current observation."""

    @abc.abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """Return the current reward."""

    def terminate_episode(self) -> bool:
        """Return whether the task reached a terminal condition."""

        return False

    def truncate_episode(self) -> bool:
        """Return whether this episode should be truncated."""

        return self._elapsed_steps >= self._max_episode_steps

    def get_reset_info(self) -> dict[str, Any]:
        """Return info for ``reset``."""

        return {}

    def get_step_info(self) -> dict[str, Any]:
        """Return info for ``step``."""

        return {}

    def post_compile(self) -> None:
        """Cache model IDs after compilation."""

    def compile_model_and_data(self) -> None:
        """Compile MJCF/XML into ``mujoco.MjModel`` and ``mujoco.MjData``."""

        if self._mjcf_model is not None:
            self._model = mujoco.MjModel.from_xml_string(
                self._mjcf_model.to_xml_string(),
                assets=self._mjcf_model.get_assets(),
            )
        elif self._xml_path is not None:
            self._model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        else:
            raise ValueError("Environment did not provide MJCF or XML path.")

        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._physics_timestep

        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        if self._passive_viewer_handle is not None:
            self._passive_viewer_handle._sim().load(self._model, self._data, "")

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        self.post_compile()
        self._dirty = False

    def mark_dirty(self) -> None:
        """Force recompilation on the next reset."""

        self._dirty = True

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs
    ):
        """Reset the environment and return ``(observation, info)``."""

        super().reset(seed=seed, options=options, **kwargs)

        if self._mjcf_model is None and self._xml_path is None:
            self._mjcf_model = self.build_mjcf()
            xml_path = self.build_xml_path()
            self._xml_path = Path(xml_path).resolve() if xml_path is not None else None

        if self._mjcf_model is not None:
            self._mjcf_model = self.modify_mjcf(self._mjcf_model)

        if self._dirty or self._never_compiled:
            self.compile_model_and_data()
            self._never_compiled = False
        else:
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)

        self._elapsed_steps = 0
        self._reset_next_step = False

        self.initialize_episode()
        mujoco.mj_forward(self._model, self._data)

        return self.compute_observation(), self.get_reset_info()

    def step(self, action: np.ndarray):
        """Run one control step and return Gymnasium's 5-tuple."""

        if self._reset_next_step:
            return self.reset()

        self.set_control(action)
        self.pre_step()
        mujoco.mj_step(self._model, self._data, nstep=self._n_steps)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        self.post_step()

        self._elapsed_steps += 1
        observation = self.compute_observation()
        reward = self.compute_reward()
        terminated = bool(self.terminate_episode())
        truncated = bool(self.truncate_episode())
        info = self.get_step_info()
        info.setdefault("elapsed_steps", self._elapsed_steps)

        return observation, reward, terminated, truncated, info

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Set the MuJoCo state and run forward kinematics."""

        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
            raise ValueError(
                f"Expected qpos {(self.model.nq,)} and qvel {(self.model.nv,)}, "
                f"got {qpos.shape} and {qvel.shape}."
            )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        if self.model.na:
            self.data.act[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_timesteps(self, physics_timestep: float, control_timestep: float) -> None:
        """Set physics and control timesteps."""

        n_steps = control_timestep / physics_timestep
        rounded_n_steps = int(round(n_steps))
        if abs(n_steps - rounded_n_steps) > 1e-6:
            raise ValueError(
                f"Control timestep {control_timestep} must be an integer multiple "
                f"of physics timestep {physics_timestep}."
            )

        self._physics_timestep = physics_timestep
        self._control_timestep = control_timestep
        self._n_steps = rounded_n_steps

    @property
    def action_space(self):
        """Return a Box derived from actuator control ranges."""

        if self._model is None:
            self.reset()
        limited = self._model.actuator_ctrllimited.ravel().astype(bool)
        ctrlrange = self._model.actuator_ctrlrange
        low = np.where(limited, ctrlrange[:, 0], -mujoco.mjMAXVAL).astype(np.float32)
        high = np.where(limited, ctrlrange[:, 1], mujoco.mjMAXVAL).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def observation_space(self):
        """Infer a Dict observation space from the current observation."""

        if self._model is None:
            self.reset()
        observation = self.compute_observation()
        if not isinstance(observation, dict):
            raise TypeError("compute_observation() must return a dictionary.")
        return gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.asarray(value).shape,
                    dtype=np.asarray(value).dtype,
                )
                for key, value in observation.items()
            }
        )

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            raise ValueError("MjModel is not initialized. Call `reset` first.")
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        if self._data is None:
            raise ValueError("MjData is not initialized. Call `reset` first.")
        return self._data

    def body_pose(self, body_id: int) -> np.ndarray:
        """Return a body's world-frame position and quaternion."""

        if body_id < 0:
            return np.full(7, np.nan, dtype=np.float64)
        return np.concatenate(
            [self.data.xpos[body_id].copy(), self.data.xquat[body_id].copy()]
        )

    def find_body_id(self, names: tuple[str, ...]) -> int:
        """Return the first body ID matching one of ``names``, or -1."""

        return self._find_named_id(mujoco.mjtObj.mjOBJ_BODY, names)

    def find_joint_id(self, names: tuple[str, ...]) -> int:
        """Return the first joint ID matching one of ``names``, or -1."""

        return self._find_named_id(mujoco.mjtObj.mjOBJ_JOINT, names)

    def _find_named_id(
        self,
        obj_type: mujoco.mjtObj,
        names: tuple[str, ...],
    ) -> int:
        for name in names:
            obj_id = mujoco.mj_name2id(self.model, obj_type, name)
            if obj_id >= 0:
                return int(obj_id)
        return -1

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        if self._mjcf_model is None:
            raise ValueError("MJCF model is not initialized.")
        return self._mjcf_model

    @property
    def physics_timestep(self) -> float:
        return self._physics_timestep

    @property
    def control_timestep(self) -> float:
        return self._control_timestep

    def _initialize_renderer(self) -> None:
        if self._model is None:
            raise ValueError("Call `reset` before rendering.")
        self._renderer = mujoco.Renderer(
            model=self._model,
            height=self._render_height,
            width=self._render_width,
        )
        mujoco.mjv_defaultFreeCamera(self._model, self._camera)

    def render(
        self,
        camera: Any = None,
        depth: bool = False,
        segmentation: bool = False,
        scene_option: Optional[mujoco.MjvOption] = None,
        scene_callback: Optional[Callable[[mujoco.MjvScene], None]] = None,
    ) -> np.ndarray:
        """Render the current state as an RGB/depth/segmentation array."""

        if self._model is None or self._data is None:
            raise ValueError("Call `reset` before render.")
        if self._renderer is None:
            self._initialize_renderer()

        if depth and segmentation:
            raise ValueError("Only one of depth or segmentation can be enabled.")
        if depth:
            self._renderer.enable_depth_rendering()
        elif segmentation:
            self._renderer.enable_segmentation_rendering()
        else:
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()

        render_camera = camera
        if render_camera is None:
            camera_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self._render_camera_name,
            )
            render_camera = self._render_camera_name if camera_id >= 0 else self._camera

        self._renderer.update_scene(
            data=self._data,
            camera=render_camera,
            scene_option=scene_option or self._scene_option,
        )
        if scene_callback is not None:
            scene_callback(self._renderer.scene)
        return self._renderer.render()

    def launch_passive_viewer(self, *args, **kwargs):
        """Launch MuJoCo's passive viewer."""

        if self._passive_viewer_handle is not None:
            raise ValueError("Passive viewer already launched.")
        if self._model is None or self._data is None:
            raise ValueError("Call `reset` before launching the passive viewer.")
        self._passive_viewer_handle = mujoco.viewer.launch_passive(
            self._model,
            self._data,
            show_left_ui=kwargs.pop("show_left_ui", False),
            show_right_ui=kwargs.pop("show_right_ui", False),
            *args,
            **kwargs,
        )
        return self._passive_viewer_handle

    def launch_viewer(self, *args, **kwargs) -> None:
        """Launch MuJoCo's managed blocking viewer."""

        if self._model is None or self._data is None:
            raise ValueError("Call `reset` before launching the viewer.")
        mujoco.viewer.launch(
            self._model,
            self._data,
            show_left_ui=kwargs.pop("show_left_ui", True),
            show_right_ui=kwargs.pop("show_right_ui", True),
            *args,
            **kwargs,
        )

    def sync_passive_viewer(self) -> None:
        """Sync the passive viewer if it is running."""

        if self._passive_viewer_handle is None:
            raise ValueError("Passive viewer is not launched.")
        self._passive_viewer_handle.sync()

    def close_passive_viewer(self) -> None:
        """Close the passive viewer."""

        if self._passive_viewer_handle is not None:
            self._passive_viewer_handle.close()
            self._passive_viewer_handle = None

    def check_contact_geom_ids(self, geoms_a, geoms_b=None) -> bool:
        """Return True if geom sets are currently in contact."""

        return contact_utils.check_contact_geom_ids(self.data, geoms_a, geoms_b)

    def collect_geoms_under_body_prefix(
        self,
        body_prefix: str,
        collision_only: bool = False,
    ) -> set[int]:
        """Collect geom IDs under bodies whose names start with prefix."""

        return contact_utils.collect_geoms_under_body_prefix(
            self.model,
            body_prefix,
            collision_only=collision_only,
        )

    def collect_geoms_by_name_prefix(
        self,
        geom_prefix: str,
        collision_only: bool = False,
    ) -> set[int]:
        """Collect geom IDs whose names start with prefix."""

        return contact_utils.collect_geoms_by_name_prefix(
            self.model,
            geom_prefix,
            collision_only=collision_only,
        )

    def contact_pairs(self, geoms_a=None, geoms_b=None):
        """Return current contact pairs with geom names and contact distances."""

        return contact_utils.contact_pairs(self.model, self.data, geoms_a, geoms_b)

    @contextlib.contextmanager
    def passive_viewer(self, *args, **kwargs):
        """Context manager for MuJoCo's passive viewer."""

        viewer = self.launch_passive_viewer(*args, **kwargs)
        try:
            yield viewer
        finally:
            self.close_passive_viewer()

    def close(self) -> None:
        """Release renderer and viewer resources."""

        self.close_passive_viewer()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
