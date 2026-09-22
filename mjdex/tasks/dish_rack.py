"""Dish-rack task for MjDex environments."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.robots import ASSET_ROOT
from mjdex.tasks.base import Task
from mjdex.transform import normalize_quat, pose_to_mat, quat_mul, yaw_quat

DISH_RACK_ASSET_ROOT = ASSET_ROOT / "objects" / "dish_rack_small"
DISH_ASSET_ROOT = ASSET_ROOT / "objects" / "dish"
DISH_TARGET_OFFSET = np.array([0.0, 0.0, 0.10], dtype=np.float64)
DISH_TARGET_LOCAL_QUAT = np.array(
    [0.7071, 0.7071, 0.0, 0.0],
    dtype=np.float64,
)


class DishRackTask(Task):
    """Attach a fixed dish rack and a free dish to a MjDex scene."""

    def __init__(
        self,
        rack_pos: tuple[float, float, float] = (0.45, -0.14, 0.005),
        rack_quat: tuple[float, float, float, float] | None = None,
        dish_pos: tuple[float, float, float] = (0.45, 0.14, 0.0025),
        dish_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        show_target: bool = False,
        reset_randomness: str = "low",
        reset_xy_bounds: tuple[tuple[float, float], tuple[float, float]] = (
            (0.25, 0.65),
            (-0.35, 0.35),
        ),
        min_part_xy_distance: float = 0.20,
        success_position_tolerance: tuple[float, float, float] = (0.02, 0.02, 0.018),
        success_angle_tolerance: float = 15.0,
        success_linear_speed: float = 0.05,
        success_angular_speed: float = 0.5,
        require_low_speed_for_success: bool = False,
    ) -> None:
        self.base_rack_pos = np.asarray(rack_pos, dtype=np.float64)
        # Default: -90° yaw so rack slots face the robot's side (y-axis direction).
        if rack_quat is None:
            self.base_rack_quat = normalize_quat(
                np.asarray(yaw_quat(-np.pi / 2), dtype=np.float64)
            )
        else:
            self.base_rack_quat = normalize_quat(rack_quat)
        self.base_dish_pos = np.asarray(dish_pos, dtype=np.float64)
        self.base_dish_quat = normalize_quat(dish_quat)
        self.rack_pos = self.base_rack_pos.copy()
        self.rack_quat = self.base_rack_quat.copy()
        self.dish_pos = self.base_dish_pos.copy()
        self.dish_quat = self.base_dish_quat.copy()
        self.show_target = bool(show_target)
        self.reset_randomness = reset_randomness
        self.reset_xy_bounds = (
            tuple(float(value) for value in reset_xy_bounds[0]),
            tuple(float(value) for value in reset_xy_bounds[1]),
        )
        self.min_part_xy_distance = float(min_part_xy_distance)
        self.success_position_tolerance = np.asarray(success_position_tolerance, dtype=float)
        self.success_angle_tolerance = float(success_angle_tolerance)
        self.success_linear_speed = float(success_linear_speed)
        self.success_angular_speed = float(success_angular_speed)
        self.require_low_speed_for_success = bool(require_low_speed_for_success)
        if (self.success_position_tolerance.shape != (3,)
                or not np.isfinite(self.success_position_tolerance).all()
                or np.any(self.success_position_tolerance <= 0)):
            raise ValueError("success_position_tolerance must contain three positive finite values")
        if not 0 < self.success_angle_tolerance < 90:
            raise ValueError("success_angle_tolerance must be in (0, 90) degrees")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.success_linear_speed, self.success_angular_speed)):
            raise ValueError("Success speed limits must be positive and finite")
        # Mesh AABB center: the OBJ origin is on the bottom face, not at its center.
        with (DISH_ASSET_ROOT / "assets" / "plate.obj").open() as stream:
            vertices = np.array([
                [float(v) for v in line.split()[1:4]]
                for line in stream if line.startswith("v ")
            ])
        self.dish_center_offset = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        # Far-end slot (+y), center height measured by collision-based seating in
        # collect/dishrack_success_pose.py for this rack and plate geometry.
        self.success_target_center = np.array([0.0, 0.075, 0.1341])

        self._rack_body_id = -1
        self._dish_body_id = -1
        self._target_body_id = -1
        self._dish_joint_id = -1
        self._dish_qpos_addr = -1
        self._dish_dof_addr = -1
        self._rack_geom_ids: set[int] = set()
        self._dish_geom_ids: set[int] = set()

    def build_mjcf(self, scene: mjcf.RootElement) -> mjcf.RootElement:
        rack_texture = scene.asset.add(
            "texture",
            type="2d",
            name="dish_rack_wood_texture",
            file=(DISH_RACK_ASSET_ROOT / "assets" / "light_wood.png").as_posix(),
        )
        rack_material = scene.asset.add(
            "material",
            name="dish_rack_wood_material",
            texture=rack_texture,
            specular=0.5,
            shininess=0.5,
            rgba=(0.7, 0.7, 0.7, 1),
        )
        dish_texture = scene.asset.add(
            "texture",
            type="2d",
            name="dish_rack_dish_texture",
            file=(DISH_ASSET_ROOT / "assets" / "plate.png").as_posix(),
        )
        dish_material = scene.asset.add(
            "material",
            name="dish_rack_dish_material",
            texture=dish_texture,
            specular=0.5,
            shininess=0.5,
        )
        dish_mesh = scene.asset.add(
            "mesh",
            name="dish_rack_dish",
            file=(DISH_ASSET_ROOT / "assets" / "plate.obj").as_posix(),
        )
        dish_collision_meshes = [
            scene.asset.add(
                "mesh",
                name=f"dish_rack_dish_collision_{idx}",
                file=(
                    DISH_ASSET_ROOT / "assets" / f"plate_collision_{idx}.obj"
                ).as_posix(),
            )
            for idx in range(32)
        ]

        rack_body = scene.worldbody.add(
            "body",
            name="dish_rack",
            pos=tuple(self.rack_pos),
            quat=tuple(self.rack_quat),
        )
        for name, pos, size in (
            ("front", (0.0, 0.1, 0.0), (0.05, 0.005, 0.005)),
            ("rear", (0.0, -0.1, 0.0), (0.05, 0.005, 0.005)),
            ("right", (0.05, 0.0, 0.0), (0.005, 0.16, 0.005)),
            ("left", (-0.05, 0.0, 0.0), (0.005, 0.16, 0.005)),
        ):
            rack_body.add(
                "geom",
                name=f"dish_rack_base_{name}",
                type="box",
                pos=pos,
                size=size,
                material=rack_material,
            )
        rack_columns = rack_body.add("body", name="dish_rack_columns")
        for idx, y_pos in enumerate((0.1, 0.05, 0.0, -0.05, -0.1)):
            for side, x_pos in (("left", -0.05), ("right", 0.05)):
                rack_columns.add(
                    "geom",
                    name=f"dish_rack_column_{idx}_{side}",
                    type="cylinder",
                    pos=(x_pos, y_pos, 0.045),
                    size=(0.005, 0.0425),
                    material=rack_material,
                )

        dish_body = scene.worldbody.add(
            "body",
            name="dish",
            pos=tuple(self.dish_pos),
            quat=tuple(self.dish_quat),
        )
        dish_body.add("joint", name="dish_joint", type="free", damping=0.005)
        dish_body.add(
            "geom",
            name="dish_visual_geom",
            type="mesh",
            mesh=dish_mesh,
            material=dish_material,
            group=2,
            contype=0,
            conaffinity=0,
            mass=0,
        )
        for idx, mesh in enumerate(dish_collision_meshes):
            dish_body.add(
                "geom",
                name=f"dish_collision_geom_{idx}",
                type="mesh",
                mesh=mesh,
                group=3,
                condim=6,
                priority=1,
                solref=(0.01, 1),
                solimp=(0.99, 0.999, 0.001, 0.5, 1),
            )

        if self.show_target:
            target_quat = quat_mul(self.rack_quat, DISH_TARGET_LOCAL_QUAT)
            target_body = scene.worldbody.add(
                "body",
                name="dish_target",
                pos=tuple(self.rack_pos + DISH_TARGET_OFFSET),
                quat=tuple(target_quat),
            )
            target_body.add(
                "geom",
                name="dish_target_visual_geom",
                type="mesh",
                mesh=dish_mesh,
                material=dish_material,
                group=2,
                contype=0,
                conaffinity=0,
                rgba=(1, 1, 1, 0.25),
            )
        return scene

    def post_compile(self, env: Any) -> None:
        self._rack_body_id = env.find_body_id(("dish_rack",))
        self._dish_body_id = env.find_body_id(("dish",))
        self._target_body_id = env.find_body_id(("dish_target",))
        self._dish_joint_id = env.find_joint_id(("dish_joint",))
        if self._dish_joint_id >= 0:
            self._dish_qpos_addr = int(env.model.jnt_qposadr[self._dish_joint_id])
            self._dish_dof_addr = int(env.model.jnt_dofadr[self._dish_joint_id])

        self._rack_geom_ids = env.collect_geoms_under_body_prefix(
            "dish_rack", collision_only=True
        )
        self._dish_geom_ids = env.collect_geoms_under_body_prefix(
            "dish", collision_only=True
        )

    def initialize_episode(self, env: Any) -> None:
        self.rack_pos, self.rack_quat, self.dish_pos, self.dish_quat = (
            self._sample_reset_pose(env)
        )
        if self._dish_qpos_addr >= 0:
            env.data.qpos[self._dish_qpos_addr : self._dish_qpos_addr + 3] = (
                self.dish_pos
            )
            env.data.qpos[self._dish_qpos_addr + 3 : self._dish_qpos_addr + 7] = (
                self.dish_quat
            )
        if self._dish_dof_addr >= 0:
            env.data.qvel[self._dish_dof_addr : self._dish_dof_addr + 6] = 0.0
        if self._rack_body_id >= 0:
            env.model.body_pos[self._rack_body_id] = self.rack_pos
            env.model.body_quat[self._rack_body_id] = self.rack_quat
        if self._target_body_id >= 0:
            env.model.body_pos[self._target_body_id] = (
                self.rack_pos + DISH_TARGET_OFFSET
            )
            env.model.body_quat[self._target_body_id] = quat_mul(
                self.rack_quat, DISH_TARGET_LOCAL_QUAT
            )
        mujoco.mj_forward(env.model, env.data)

    def compute_observation(self, env: Any) -> dict[str, np.ndarray]:
        return {
            "rack_pose": self.rack_pose(env).astype(np.float32),
            "dish_pose": self.dish_pose(env).astype(np.float32),
        }

    def compute_reward(self, env: Any) -> float:
        return float(self.is_success(env))

    def terminate_episode(self, env: Any) -> bool:
        return self.is_success(env)

    def success_metrics(self, env: Any) -> dict[str, Any]:
        """Geometric/instantaneous success; angles in degrees, speeds in SI units.

        Compare the mesh center, not the asymmetric body origin. The absolute
        normal dot product accepts both faces and ignores in-plane plate spin.
        By default success uses pose only, so it can be reconstructed from stored
        observations. Optional speed gating does not certify stability over time.
        """
        rack = pose_to_mat(self.rack_pose(env))
        dish = pose_to_mat(self.dish_pose(env))
        relative_rotation = rack[:3, :3].T @ dish[:3, :3]
        center = rack[:3, :3].T @ (
            dish[:3, 3] + dish[:3, :3] @ self.dish_center_offset - rack[:3, 3]
        )
        error = center - self.success_target_center
        angle = float(np.degrees(np.arccos(np.clip(abs(relative_rotation[1, 2]), 0, 1))))
        velocity = env.data.qvel[self._dish_dof_addr:self._dish_dof_addr + 6]
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        pose_success = bool(
            np.all(np.abs(error) <= self.success_position_tolerance)
            and angle <= self.success_angle_tolerance
        )
        low_speed = bool(linear_speed <= self.success_linear_speed
                         and angular_speed <= self.success_angular_speed)
        success = pose_success and (not self.require_low_speed_for_success or low_speed)
        return {
            "success": success,
            "dish_rack_pose_success": pose_success,
            "dish_low_speed": low_speed,
            "dish_rack_center_local": center,
            "dish_rack_position_error": error,
            "dish_rack_angle_error_deg": angle,
            "dish_linear_speed": linear_speed,
            "dish_angular_speed": angular_speed,
        }

    def is_success(self, env: Any) -> bool:
        return self.success_metrics(env)["success"]

    def get_info(self, env: Any) -> dict[str, Any]:
        return {
            "rack_pose": self.rack_pose(env),
            "dish_pose": self.dish_pose(env),
            "dish_rack_relative_pos": (
                self.dish_pose(env)[:3] - self.rack_pose(env)[:3]
            ),
            "dish_rack_contact": self.dish_rack_contact(env),
            **self.success_metrics(env),
        }

    def rack_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._rack_body_id)

    def dish_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._dish_body_id)

    def dish_rack_contact(self, env: Any) -> bool:
        if not self._rack_geom_ids or not self._dish_geom_ids:
            return False
        return env.check_contact_geom_ids(self._dish_geom_ids, self._rack_geom_ids)

    def _sample_reset_pose(
        self, env: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xy_range, yaw_range = self._reset_ranges()
        if xy_range == 0.0 and yaw_range == 0.0:
            return (
                self.base_rack_pos.copy(),
                self.base_rack_quat.copy(),
                self.base_dish_pos.copy(),
                self.base_dish_quat.copy(),
            )
        rng = getattr(env, "np_random", None) or np.random.default_rng()
        x_bounds, y_bounds = self.reset_xy_bounds
        for _ in range(1000):
            rack_pos, rack_quat = self._sample_part_pose(
                rng, self.base_rack_pos, self.base_rack_quat, xy_range, yaw_range
            )
            dish_pos, dish_quat = self._sample_part_pose(
                rng, self.base_dish_pos, self.base_dish_quat, xy_range, yaw_range
            )
            positions_in_bounds = all(
                x_bounds[0] <= pos[0] <= x_bounds[1]
                and y_bounds[0] <= pos[1] <= y_bounds[1]
                for pos in (rack_pos, dish_pos)
            )
            parts_separated = (
                np.linalg.norm(rack_pos[:2] - dish_pos[:2]) >= self.min_part_xy_distance
            )
            if positions_in_bounds and parts_separated:
                return rack_pos, rack_quat, dish_pos, dish_quat
        return (
            self.base_rack_pos.copy(),
            self.base_rack_quat.copy(),
            self.base_dish_pos.copy(),
            self.base_dish_quat.copy(),
        )

    def _reset_ranges(self) -> tuple[float, float]:
        randomness = self.reset_randomness.lower()
        if randomness in {"none", "fixed"}:
            return 0.0, 0.0
        if randomness == "low":
            return 0.05, np.radians(30.0)
        if randomness in {"med", "medium"}:
            return 0.10, np.radians(45.0)
        raise ValueError(
            "reset_randomness must be one of 'fixed', 'low', 'med', or 'medium'."
        )

    @staticmethod
    def _sample_part_pose(
        rng: np.random.Generator,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        xy_range: float,
        yaw_range: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = base_pos.copy()
        pos[:2] += rng.uniform(-xy_range, xy_range, size=2)
        quat = quat_mul(
            np.asarray(yaw_quat(float(rng.uniform(-yaw_range, yaw_range)))),
            base_quat,
        )
        return pos, quat / np.linalg.norm(quat)


__all__ = ["DishRackTask"]
