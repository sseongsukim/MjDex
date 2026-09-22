"""One viewer model holding a spaced copy of every GPU world.

``WarpVectorEnv`` batches a single compiled scene: ``model`` describes one world
and only ``data`` (plus the per-world ``body_pos``/``body_quat`` fields) carries
the batch. A passive viewer therefore shows one world at a time. This module
compiles a visualization-only model that attaches ``num_envs`` translated copies
of that scene under a shared floor, and copies the whole GPU batch into it.

The composite model is never stepped; :meth:`CompositeViewerModel.update` writes
state and calls ``mj_forward`` so the viewer has fresh ``xpos``/``xquat``.
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np


def grid_shape(num_envs: int) -> tuple[int, int]:
    """Return the (rows, columns) of the smallest near-square grid."""

    columns = math.ceil(math.sqrt(num_envs))
    return math.ceil(num_envs / columns), columns


def scene_bounds(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Return the xy center and size of the scene's footprint at the current pose.

    Infinite planes are skipped; every other geom contributes its world-aligned
    bounding box, which covers the arm's home configuration. ``geom_rbound``
    would do the same job in one line, but the bounding sphere of a wide flat
    tabletop overstates the footprint by nearly a factor of two.
    """

    finite = np.flatnonzero(model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    if finite.size == 0:
        return np.zeros(2), np.ones(2)
    rotations = data.geom_xmat[finite].reshape(-1, 3, 3)
    boxes = model.geom_aabb[finite].reshape(-1, 6)
    centers = data.geom_xpos[finite] + np.einsum("nij,nj->ni", rotations, boxes[:, :3])
    halves = np.einsum("nij,nj->ni", np.abs(rotations), boxes[:, 3:])
    low = (centers - halves)[:, :2].min(axis=0)
    high = (centers + halves)[:, :2].max(axis=0)
    return (low + high) / 2.0, high - low


def world_offsets(num_envs: int, pitch: tuple[float, float]) -> np.ndarray:
    """Place worlds on a centered, row-major, Isaac-style grid.

    Task scenes face -x, so columns spread along y and rows recede along -x:
    a camera in front of the grid then sees every world from the same angle its
    own ``render_camera`` uses, instead of edge-on.
    """

    rows, columns = grid_shape(num_envs)
    pitch_x, pitch_y = pitch
    return np.asarray(
        [
            (
                ((rows - 1) / 2.0 - index // columns) * pitch_x,
                (index % columns - (columns - 1) / 2.0) * pitch_y,
                0.0,
            )
            for index in range(num_envs)
        ],
        dtype=np.float64,
    )


def _joint_widths(joint_type: int) -> tuple[int, int]:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _spec_factory(cpu_env):
    """Return a callable building a fresh spec of the task scene.

    ``attach`` consumes the child spec, so the root and every world copy need
    their own. A checked-in XML is reopened by path to keep its asset directories
    resolvable; a programmatic MJCF tree is serialized once and reused.
    """

    if cpu_env._mjcf_model is not None:
        xml = cpu_env._mjcf_model.to_xml_string()
        assets = cpu_env._mjcf_model.get_assets()
        return lambda: mujoco.MjSpec.from_string(xml, assets=assets)
    if cpu_env._xml_path is not None:
        path = str(cpu_env._xml_path)
        return lambda: mujoco.MjSpec.from_file(path)
    raise ValueError("Environment did not provide MJCF or XML path.")


def _is_plane(geom) -> bool:
    return geom.type == mujoco.mjtGeom.mjGEOM_PLANE


class CompositeViewerModel:
    """Visualization-only model containing one translated copy per GPU world."""

    def __init__(
        self, env, *, pitch: tuple[float, float] | None = None, margin: float = 0.3
    ):
        self.env = env
        self.num_envs = env.num_envs
        source = env.model
        self.center, size = scene_bounds(source, env._env.data)
        self.pitch = pitch or (float(size[0] + margin), float(size[1] + margin))
        self.offsets = world_offsets(self.num_envs, self.pitch)

        build_spec = _spec_factory(env._env)
        # The root keeps the task's own <visual>, <option> and asset payload, so
        # the grid view retains the skybox, haze and lighting of a single world.
        # Everything that exists once per world is deleted here and re-attached
        # below; the infinite floor and the directional lights are shared.
        spec = build_spec()
        for key in list(spec.keys):
            spec.delete(key)
        for body in list(spec.worldbody.bodies):
            spec.delete(body)
        for camera in list(spec.worldbody.cameras):
            spec.delete(camera)
        for site in list(spec.worldbody.sites):
            spec.delete(site)
        for geom in list(spec.worldbody.geoms):
            if not _is_plane(geom):
                spec.delete(geom)

        for world in range(self.num_envs):
            child = build_spec()
            for key in list(child.keys):
                child.delete(key)
            # Coplanar copies of the shared floor only cause z-fighting, and one
            # light per world washes the grid out.
            for geom in list(child.worldbody.geoms):
                if _is_plane(geom):
                    child.delete(geom)
            for light in list(child.worldbody.lights):
                child.delete(light)
            frame = spec.worldbody.add_frame(pos=self.offsets[world])
            spec.attach(child, prefix=self.prefix(world), frame=frame)

        self.model = spec.compile()
        self.model.opt.timestep = source.opt.timestep
        self.data = mujoco.MjData(self.model)
        self._build_maps(source)

    @staticmethod
    def prefix(world: int) -> str:
        return f"world_{world}/"

    def _destination(self, object_type, world: int, name: str | None) -> int:
        if name is None:
            raise ValueError(
                "The composite viewer maps worlds by name; the task scene has an "
                f"unnamed {mujoco.mju_type2Str(object_type)}."
            )
        target = self.prefix(world) + name
        destination = mujoco.mj_name2id(self.model, object_type, target)
        if destination < 0:
            raise ValueError(f"Composite model is missing {target!r}.")
        return destination

    def _build_maps(self, source) -> None:
        """Precompute source-to-composite index arrays for vectorized copies."""

        worlds = self.num_envs
        # Body 0 is the shared worldbody and has no per-world counterpart.
        self._source_bodies = np.arange(1, source.nbody)
        self._body_index = np.zeros((worlds, self._source_bodies.size), np.int32)
        self._body_shift = np.zeros((worlds, self._source_bodies.size, 3))
        self._qpos_index = np.zeros((worlds, source.nq), np.int32)
        self._qvel_index = np.zeros((worlds, source.nv), np.int32)
        self._qpos_shift = np.zeros((worlds, source.nq))
        # Mocap rows are addressed by slot, not by body id, in both models.
        mocap_bodies = np.flatnonzero(source.body_mocapid >= 0)
        self._source_mocap = source.body_mocapid[mocap_bodies].astype(np.int32)
        self._mocap_index = np.zeros((worlds, mocap_bodies.size), np.int32)
        mocap_slot = np.full(source.nbody, -1)
        mocap_slot[mocap_bodies] = np.arange(mocap_bodies.size)

        for world in range(worlds):
            offset = self.offsets[world]
            for joint in range(source.njnt):
                name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint)
                destination = self._destination(mujoco.mjtObj.mjOBJ_JOINT, world, name)
                qwidth, vwidth = _joint_widths(int(source.jnt_type[joint]))
                start = int(source.jnt_qposadr[joint])
                self._qpos_index[world, start : start + qwidth] = np.arange(
                    self.model.jnt_qposadr[destination],
                    self.model.jnt_qposadr[destination] + qwidth,
                )
                start_dof = int(source.jnt_dofadr[joint])
                self._qvel_index[world, start_dof : start_dof + vwidth] = np.arange(
                    self.model.jnt_dofadr[destination],
                    self.model.jnt_dofadr[destination] + vwidth,
                )
                if source.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    self._qpos_shift[world, start : start + 3] = offset

            for slot, body in enumerate(self._source_bodies):
                name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, int(body))
                destination = self._destination(mujoco.mjtObj.mjOBJ_BODY, world, name)
                self._body_index[world, slot] = destination
                # ``body_pos`` is parent-relative, so only worlds' root bodies
                # absorb the grid offset; the frame flattened it into them.
                if source.body_parentid[body] == 0:
                    self._body_shift[world, slot] = offset
                if mocap_slot[body] >= 0:
                    self._mocap_index[world, mocap_slot[body]] = (
                        self.model.body_mocapid[destination]
                    )

    def update(self, values: dict) -> None:
        """Write one host-side batch of GPU state into the composite model."""

        source_bodies = self._source_bodies
        self.model.body_pos[self._body_index] = (
            values["body_pos"][:, source_bodies] + self._body_shift
        )
        self.model.body_quat[self._body_index] = values["body_quat"][:, source_bodies]
        self.data.qpos[self._qpos_index] = values["qpos"] + self._qpos_shift
        self.data.qvel[self._qvel_index] = values["qvel"]
        if self._source_mocap.size:
            self.data.mocap_pos[self._mocap_index] = (
                values["mocap_pos"][:, self._source_mocap] + self.offsets[:, None]
            )
            self.data.mocap_quat[self._mocap_index] = values["mocap_quat"][
                :, self._source_mocap
            ]
        self.data.time = float(np.asarray(values["time"]).reshape(-1)[0])
        mujoco.mj_forward(self.model, self.data)

    def sync(self) -> None:
        """Pull every world off the GPU and refresh the composite model."""

        import jax

        env = self.env
        self.update(
            jax.device_get(
                {
                    "qpos": env.data.qpos,
                    "qvel": env.data.qvel,
                    "mocap_pos": env.data.mocap_pos,
                    "mocap_quat": env.data.mocap_quat,
                    "time": env.data.time,
                    "body_pos": env._model_fields["body_pos"],
                    "body_quat": env._model_fields["body_quat"],
                }
            )
        )

    def frame_camera(self, viewer) -> None:
        """Point the free camera at the whole grid."""

        rows, columns = grid_shape(self.num_envs)
        pitch_x, pitch_y = self.pitch
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # The grid is centered on the origin, but the scene itself sits off to
        # one side of it, so orbit around the footprint instead.
        viewer.cam.lookat[:] = self.offsets.mean(axis=0) + (*self.center, 0.0)
        viewer.cam.distance = 1.2 * max(rows * pitch_x, columns * pitch_y)
        # Look along -x, the direction each world's own render_camera faces.
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -40.0


class BatchViewer:
    """Passive MuJoCo viewer mirroring a whole Warp batch while it runs.

    Wraps :class:`CompositeViewerModel` in a ``launch_passive`` window so a
    batched rollout can be watched live. The window is a mirror: closing it
    stops the drawing, never the simulation.
    """

    def __init__(
        self,
        env,
        *,
        pitch: tuple[float, float] | None = None,
        margin: float = 0.3,
        speed: float = 1.0,
        show_ui: bool = False,
    ):
        import mujoco.viewer

        if speed <= 0:
            raise ValueError("Viewer speed must be positive.")
        self.composite = CompositeViewerModel(env, pitch=pitch, margin=margin)
        self.composite.sync()
        self._period = env._env.control_timestep / speed
        self._viewer = mujoco.viewer.launch_passive(
            self.composite.model,
            self.composite.data,
            show_left_ui=show_ui,
            show_right_ui=show_ui,
        )
        with self._viewer.lock():
            self.composite.frame_camera(self._viewer)
        self._viewer.sync()
        self._deadline = time.monotonic()

    @property
    def running(self) -> bool:
        return self._viewer.is_running()

    def sync(self) -> bool:
        """Redraw from the current GPU batch, paced to wall-clock time.

        Returns whether the window is still open, so a caller can stop paying
        for the host transfer once the viewer is gone.
        """
        if not self._viewer.is_running():
            return False
        with self._viewer.lock():
            self.composite.sync()
        self._viewer.sync()
        remaining = self._deadline + self._period - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._deadline = time.monotonic()
        return True

    def close(self) -> None:
        self._viewer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
