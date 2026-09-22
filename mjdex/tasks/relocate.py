"""YCB object relocation task for MjDex environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.contact import geom_can_collide
from mjdex.robots import ASSET_ROOT
from mjdex.tasks.base import Task
from mjdex.transform import euler_delta_quat, normalize_quat, yaw_quat

YCB_ASSET_ROOT = ASSET_ROOT / "objects" / "ycb"
YCB_OBJECTS = {
    "cracker": "003_cracker_box",
    "mustard": "006_mustard_bottle",
    "meat": "010_potted_meat_can",
    "banana": "011_banana",
    "bleach": "021_bleach_cleanser",
}
YCB_INIT_HEIGHT = {
    "cracker": 0.112,
    "mustard": 0.085,
    "meat": 0.048,
    "banana": 0.035,
    "bleach": 0.11,
}
YCB_HEIGHT = {
    "cracker": 0.0717,
    "mustard": 0.1913,
    "meat": 0.0835,
    "banana": 0.07,
    "bleach": 0.2506,
}
YCB_YAW_RANGE_DEG = {
    "cracker": (45.0, 90.0),
    "mustard": (-45.0, 45.0),
    "meat": (-45.0, 45.0),
    "banana": (-180.0, 180.0),
    "bleach": (-45.0, 45.0),
}


class RelocateTask(Task):
    """Move a free YCB object to a sampled mocap target pose."""

    def __init__(
        self,
        object_id: str = "mustard",
        object_xml: str | Path | None = None,
        # The x ranges sit 0.10 m further out than the table-flush layout, to
        # follow the hand scenes' tabletop (see _HAND_TABLE_SETBACK in
        # mjdex/envs.py). Keeping them in step holds the object and target in the
        # same spot *on the table* rather than letting the table slide out from
        # under them.
        object_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
            (0.45, 0.65),
            (-0.25, 0.25),
        ),
        target_xyz_range: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((0.40, 0.65), (-0.30, -0.05), (0.20, 0.40)),
        min_object_target_distance: float = 0.30,
        require_orientation: bool = False,
        sparse_reward: bool = False,
        position_threshold: float = 0.025,
        rotation_threshold: float = 0.05,
        success_reward: float = 1000.0,
    ) -> None:
        if object_id not in YCB_OBJECTS:
            raise ValueError(
                f"Unknown YCB object '{object_id}'. Options: {sorted(YCB_OBJECTS)}"
            )

        object_name = YCB_OBJECTS[object_id]
        self.object_id = object_id
        self.object_name = object_name
        self.object_xml = Path(
            object_xml or YCB_ASSET_ROOT / object_name / f"{object_name}.xml"
        )
        self.object_xy_range = object_xy_range
        self.target_xyz_range = target_xyz_range
        self.min_object_target_distance = float(min_object_target_distance)
        self.require_orientation = bool(require_orientation)
        self.sparse_reward = bool(sparse_reward)
        self.position_threshold = float(position_threshold)
        self.rotation_threshold = float(rotation_threshold)
        self.success_reward = float(success_reward)

        self.object_pos = np.array(
            [
                np.mean(object_xy_range[0]),
                np.mean(object_xy_range[1]),
                YCB_INIT_HEIGHT[object_id],
            ],
            dtype=np.float64,
        )
        self.object_quat = np.asarray(yaw_quat(0.0), dtype=np.float64)
        self.target_pos = np.array(
            [np.mean(axis_range) for axis_range in target_xyz_range],
            dtype=np.float64,
        )
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._object_body_id = -1
        self._target_body_id = -1
        self._target_mocap_id = -1
        self._object_joint_id = -1
        self._object_qpos_addr = -1
        self._object_dof_addr = -1
        self._object_geom_ids: set[int] = set()
        self._robot_geom_ids: set[int] = set()
        self._table_geom_ids: set[int] = set()

    def build_mjcf(self, scene: mjcf.RootElement) -> mjcf.RootElement:
        root = ET.parse(self.object_xml).getroot()
        target_xml = root.find("./worldbody/body[@name='target_body']")
        if target_xml is None:
            raise ValueError(
                f"Relocate object XML must contain target_body: {self.object_xml}"
            )

        asset_dir = self.object_xml.parent / "assets"
        assets: dict[tuple[str, str], mjcf.Element] = {}
        for asset_xml in root.findall("./asset/*"):
            tag = asset_xml.tag
            source_name = asset_xml.get("name") or Path(asset_xml.get("file", "")).stem
            kwargs = self._element_kwargs(asset_xml, skip={"name", "file", "texture"})
            kwargs["name"] = f"relocate_{source_name}"
            if asset_xml.get("file"):
                kwargs["file"] = (asset_dir / asset_xml.get("file", "")).as_posix()
            if tag == "material" and asset_xml.get("texture"):
                kwargs["texture"] = assets[("texture", asset_xml.get("texture", ""))]
            assets[(tag, source_name)] = scene.asset.add(tag, **kwargs)

        defaults = {
            default_xml.get("class", ""): self._element_kwargs(default_xml.find("geom"))
            for default_xml in root.findall("./default/default")
            if default_xml.find("geom") is not None
        }
        object_xml = root.find("./worldbody/body[@name='body']")
        if object_xml is None:
            raise ValueError(f"YCB XML must contain body: {self.object_xml}")

        object_body = scene.worldbody.add(
            "body",
            name="relocate_object",
            pos=tuple(self.object_pos),
            quat=tuple(self.object_quat),
        )
        inertial_xml = object_xml.find("inertial")
        if inertial_xml is not None:
            object_body.add("inertial", **self._element_kwargs(inertial_xml))
        joint_xml = object_xml.find("joint")
        if joint_xml is None:
            raise ValueError(f"YCB object must contain a free joint: {self.object_xml}")
        joint_kwargs = self._element_kwargs(joint_xml, skip={"name"})
        object_body.add("joint", name="relocate_object_joint", **joint_kwargs)
        for index, geom_xml in enumerate(object_xml.findall("geom")):
            self._add_geom(
                object_body,
                geom_xml,
                defaults,
                assets,
                name=f"relocate_object_geom_{index}",
            )

        target_body = scene.worldbody.add(
            "body",
            name="relocate_target",
            mocap=True,
            pos=tuple(self.target_pos),
            quat=tuple(self.target_quat),
        )
        target_geom_xml = target_xml.find("geom")
        if target_geom_xml is None:
            raise ValueError(
                f"target_body must contain a visual geom: {self.object_xml}"
            )
        self._add_geom(
            target_body,
            target_geom_xml,
            defaults,
            assets,
            name="relocate_target_visual_geom",
        )
        return scene

    def post_compile(self, env: Any) -> None:
        self._object_body_id = env.find_body_id(("relocate_object",))
        self._target_body_id = env.find_body_id(("relocate_target",))
        self._object_joint_id = env.find_joint_id(("relocate_object_joint",))
        if self._object_joint_id >= 0:
            self._object_qpos_addr = int(env.model.jnt_qposadr[self._object_joint_id])
            self._object_dof_addr = int(env.model.jnt_dofadr[self._object_joint_id])
        if self._target_body_id >= 0:
            self._target_mocap_id = int(env.model.body_mocapid[self._target_body_id])

        self._object_geom_ids = env.collect_geoms_under_body_prefix(
            "relocate_object", collision_only=True
        )
        self._table_geom_ids = env.collect_geoms_under_body_prefix(
            "table", collision_only=True
        )
        excluded = set(self._object_geom_ids) | set(self._table_geom_ids)
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            excluded.add(int(floor_id))

        if hasattr(env, "_arms"):
            self._robot_geom_ids = set()
            for arm in env._arms.values():
                self._robot_geom_ids |= env.collect_geoms_under_body_prefix(
                    arm.prefix, collision_only=True
                )
        else:
            self._robot_geom_ids = {
                geom_id
                for geom_id in range(env.model.ngeom)
                if geom_id not in excluded and geom_can_collide(env.model, geom_id)
            }

    def initialize_episode(self, env: Any) -> None:
        rng = getattr(env, "np_random", None) or np.random.default_rng()
        object_x = float(rng.uniform(*self.object_xy_range[0]))
        object_y = float(rng.uniform(*self.object_xy_range[1]))
        self.object_pos = np.array(
            [object_x, object_y, YCB_INIT_HEIGHT[self.object_id]], dtype=np.float64
        )
        yaw_low, yaw_high = YCB_YAW_RANGE_DEG[self.object_id]
        object_yaw = np.deg2rad(float(rng.uniform(yaw_low, yaw_high)))
        if self.object_id == "cracker" and rng.random() < 0.5:
            object_yaw *= -1.0
        self.object_quat = np.asarray(yaw_quat(object_yaw), dtype=np.float64)

        for _ in range(1000):
            self.target_pos = np.array(
                [
                    float(rng.uniform(*axis_range))
                    for axis_range in self.target_xyz_range
                ],
                dtype=np.float64,
            )
            if (
                np.linalg.norm(self.object_pos - self.target_pos)
                > self.min_object_target_distance
            ):
                break
        target_offset = euler_delta_quat(
            np.deg2rad(float(rng.uniform(-10.0, 10.0))),
            np.deg2rad(float(rng.uniform(-10.0, 10.0))),
            np.deg2rad(float(rng.uniform(-30.0, 30.0))),
        )
        self.target_quat = normalize_quat(target_offset)

        if self._object_qpos_addr >= 0:
            env.data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3] = (
                self.object_pos
            )
            env.data.qpos[self._object_qpos_addr + 3 : self._object_qpos_addr + 7] = (
                self.object_quat
            )
        if self._object_dof_addr >= 0:
            env.data.qvel[self._object_dof_addr : self._object_dof_addr + 6] = 0.0
        if self._target_mocap_id >= 0:
            env.data.mocap_pos[self._target_mocap_id] = self.target_pos
            env.data.mocap_quat[self._target_mocap_id] = self.target_quat
        mujoco.mj_forward(env.model, env.data)

    def compute_observation(self, env: Any) -> dict[str, np.ndarray]:
        return {
            "object_pose": self.object_pose(env).astype(np.float32),
            "target_pose": self.target_pose(env).astype(np.float32),
        }

    def compute_reward(self, env: Any) -> float:
        if self.sparse_reward:
            return self.success_reward if self.is_success(env) else -1.0
        if self._robot_table_contact(env):
            return -100.0

        object_pose = self.object_pose(env)
        target_pose = self.target_pose(env)
        ee_pose = self._nearest_ee_pose(env)
        robot_object_contact = env.check_contact_geom_ids(
            self._robot_geom_ids, self._object_geom_ids
        )
        ee_object_distance = np.linalg.norm(ee_pose[:3] - object_pose[:3])
        ee_target_distance = np.linalg.norm(ee_pose[:3] - target_pose[:3])
        object_target_distance = np.linalg.norm(object_pose[:3] - target_pose[:3])
        rotation_error = self._quat_distance(object_pose[3:], target_pose[3:])
        lift = max(
            min(object_pose[2], target_pose[2]) - YCB_HEIGHT[self.object_id] / 2.0,
            0.0,
        )

        reward = -0.1 * ee_object_distance
        if robot_object_contact:
            reward += 0.05 + lift
            if lift > 0.015:
                reward += 0.1
                reward += 0.3 * np.exp(-3.0 * ee_target_distance)
                reward += 0.7 * np.exp(-5.0 * object_target_distance)
                if object_target_distance < 0.05:
                    reward += np.exp(-10.0 * object_target_distance)
                    if (
                        self.require_orientation
                        and rotation_error < self.rotation_threshold
                    ):
                        reward += 2.0
                    if self.is_success(env):
                        reward += self.success_reward
        return float(reward)

    def terminate_episode(self, env: Any) -> bool:
        return self.is_success(env)

    def get_info(self, env: Any) -> dict[str, Any]:
        object_pose = self.object_pose(env)
        target_pose = self.target_pose(env)
        success = self.is_success(env)
        return {
            "object_pose": object_pose,
            "target_pose": target_pose,
            "object_target_distance": float(
                np.linalg.norm(object_pose[:3] - target_pose[:3])
            ),
            "object_target_rotation_error": self._quat_distance(
                object_pose[3:], target_pose[3:]
            ),
            "relocate_success": self.is_position_success(env),
            "success": success,
            "robot_object_contact": env.check_contact_geom_ids(
                self._robot_geom_ids, self._object_geom_ids
            ),
            "robot_table_contact": self._robot_table_contact(env),
        }

    def robot_excluded_geom_ids(self, env: Any) -> set[int]:
        return set(self._object_geom_ids)

    def object_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._object_body_id)

    def target_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._target_body_id)

    def is_success(self, env: Any) -> bool:
        position_ok = self.is_position_success(env)
        if not self.require_orientation:
            return bool(position_ok)
        object_pose = self.object_pose(env)
        target_pose = self.target_pose(env)
        return bool(
            position_ok
            and self._quat_distance(object_pose[3:], target_pose[3:])
            < self.rotation_threshold
        )

    def is_position_success(self, env: Any) -> bool:
        object_pose = self.object_pose(env)
        target_pose = self.target_pose(env)
        return bool(
            np.linalg.norm(object_pose[:3] - target_pose[:3]) < self.position_threshold
        )

    def _nearest_ee_pose(self, env: Any) -> np.ndarray:
        """Return the ee pose of the arm nearest to the object (dual) or the single arm."""
        if hasattr(env, "_arms"):
            object_pos = self.object_pose(env)[:3]
            return min(
                (env.ee_pose(side) for side in env._arms),
                key=lambda p: np.linalg.norm(p[:3] - object_pos),
            )
        return env.ee_pose()

    def _robot_table_contact(self, env: Any) -> bool:
        return bool(
            self._robot_geom_ids
            and self._table_geom_ids
            and env.check_contact_geom_ids(
                self._robot_geom_ids,
                self._table_geom_ids,
            )
        )

    def _add_geom(
        self,
        body: mjcf.Element,
        geom_xml: ET.Element,
        defaults: dict[str, dict[str, Any]],
        assets: dict[tuple[str, str], mjcf.Element],
        name: str,
    ) -> None:
        kwargs = dict(defaults.get(geom_xml.get("class", ""), {}))
        kwargs.update(
            self._element_kwargs(
                geom_xml,
                skip={"name", "class", "mesh", "material"},
            )
        )
        kwargs["name"] = name
        if geom_xml.get("mesh"):
            kwargs["mesh"] = assets[("mesh", geom_xml.get("mesh", ""))]
        if geom_xml.get("material"):
            kwargs["material"] = assets[("material", geom_xml.get("material", ""))]
        body.add("geom", **kwargs)

    @staticmethod
    def _element_kwargs(
        element: ET.Element | None,
        skip: set[str] | None = None,
    ) -> dict[str, Any]:
        if element is None:
            return {}
        skipped = skip or set()
        return {
            key: value for key, value in element.attrib.items() if key not in skipped
        }

    @staticmethod
    def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = normalize_quat(q1)
        q2 = normalize_quat(q2)
        return float(2.0 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)))


__all__ = ["RelocateTask", "YCB_OBJECTS"]
