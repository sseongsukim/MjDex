"""Mug-rack task for single-arm MjDex environments."""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.robots import ASSET_ROOT
from mjdex.tasks.base import Task
from mjdex.transform import (
    is_similar_pose,
    normalize_quat,
    pose_to_mat,
    quat_mul,
    yaw_quat,
)

MUG_RACK_ASSET_ROOT = ASSET_ROOT / "objects" / "mug_rack"
MUG_CUP_ASSET_ROOT = ASSET_ROOT / "objects" / "mug_cup"
_DEFAULT_ASSEMBLY_JSON = MUG_RACK_ASSET_ROOT / "assembly_candidates.json"
_DEFAULT_ASSEMBLY_DIR = MUG_RACK_ASSET_ROOT / "assembly"


class MugRackTask(Task):
    """Attach a fixed rack and a free mug to a MjDex scene.

    Assembly success is determined by comparing the current mug-to-rack relative
    pose against candidate poses extracted from teleoperated demonstrations,
    following the furniture-bench MugRack approach.
    """

    def __init__(
        self,
        rack_pos: tuple[float, float, float] = (0.4, -0.25, 0.0025),
        rack_quat: tuple[float, float, float, float] = (0.7071, 0.7071, 0.0, 0.0),
        mug_pos: tuple[float, float, float] = (0.4, 0.25, 0.0025),
        mug_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        rack_xml: str | Path = MUG_RACK_ASSET_ROOT / "rack.xml",
        mug_xml: str | Path = MUG_CUP_ASSET_ROOT / "mug_cup_no_target.xml",
        show_target: bool = False,
        xy_range: float = 0.05,
        yaw_range: float = np.radians(45.0),
        assembly_json: str | Path | None = _DEFAULT_ASSEMBLY_JSON,
        assembly_json_dir: str | Path | None = _DEFAULT_ASSEMBLY_DIR,
        assembly_pos_threshold: float | list[float] = (0.010, 0.010, 0.010),
        assembly_ori_bound: float = 0.94,
    ) -> None:
        self.base_rack_pos = np.asarray(rack_pos, dtype=np.float64)
        self.base_rack_quat = normalize_quat(rack_quat)
        self.base_mug_pos = np.asarray(mug_pos, dtype=np.float64)
        self.base_mug_quat = normalize_quat(mug_quat)
        self.rack_pos = self.base_rack_pos.copy()
        self.rack_quat = self.base_rack_quat.copy()
        self.mug_pos = self.base_mug_pos.copy()
        self.mug_quat = self.base_mug_quat.copy()
        self.rack_xml = Path(rack_xml)
        self.mug_xml = Path(mug_xml)
        self.show_target = bool(show_target)
        self.xy_range = float(xy_range)
        self.yaw_range = float(yaw_range)  # default = 0° to +45°
        self.assembly_pos_threshold = assembly_pos_threshold
        self.assembly_ori_bound = float(assembly_ori_bound)

        # Load assembly candidates (relative pose matrices: inv(rack) @ mug).
        # Supports two formats:
        #   1. assembly_candidates.json: {"candidates": [[4x4], ...]}  (pre-computed relative poses)
        #   2. assembly/assembly_*.json: {"data": {"reference": {"pose": [16]}, "moved": {"pose": [16]}}}
        self._assembly_candidates: list[np.ndarray] = []
        if assembly_json is not None and Path(assembly_json).exists():
            with open(assembly_json) as f:
                for c in json.load(f)["candidates"]:
                    self._assembly_candidates.append(np.asarray(c, dtype=np.float64))

        if assembly_json_dir is not None and Path(assembly_json_dir).exists():
            for json_path in sorted(
                glob(str(Path(assembly_json_dir) / "assembly_*.json"))
            ):
                with open(json_path) as f:
                    data = json.load(f)["data"]
                ref_mat = np.asarray(
                    data["reference"]["pose"], dtype=np.float64
                ).reshape(4, 4)
                moved_mat = np.asarray(data["moved"]["pose"], dtype=np.float64).reshape(
                    4, 4
                )
                self._assembly_candidates.append(np.linalg.inv(ref_mat) @ moved_mat)

        self._rack_body_id = -1
        self._mug_body_id = -1
        self._target_body_id = -1
        self._mug_joint_id = -1
        self._mug_qpos_addr = -1
        self._mug_dof_addr = -1
        self._rack_geom_ids: set[int] = set()
        self._mug_geom_ids: set[int] = set()

    def build_mjcf(self, scene: mjcf.RootElement) -> mjcf.RootElement:
        rack_asset_dir = self.rack_xml.parent / "assets"
        mug_asset_dir = self.mug_xml.parent / "assets"
        rack_texture = scene.asset.add(
            "texture",
            type="2d",
            name="mug_rack_rack_texture",
            file=(rack_asset_dir / "rack2_material_0.png").as_posix(),
        )
        rack_material = scene.asset.add(
            "material",
            name="mug_rack_rack_material",
            texture=rack_texture,
        )
        mug_texture = scene.asset.add(
            "texture",
            type="2d",
            name="mug_rack_mug_texture",
            file=(mug_asset_dir / "mug_cup.png").as_posix(),
        )
        mug_material = scene.asset.add(
            "material",
            name="mug_rack_mug_material",
            texture=mug_texture,
        )

        rack_mesh = scene.asset.add(
            "mesh",
            name="mug_rack_rack",
            file=(rack_asset_dir / "mugrack2.obj").as_posix(),
        )
        rack_collision_meshes = [
            scene.asset.add(
                "mesh",
                name=f"mug_rack_rack_collision_{idx}",
                file=(rack_asset_dir / f"mugrack2_collision_{idx}.obj").as_posix(),
            )
            for idx in range(7)
        ]
        mug_mesh = scene.asset.add(
            "mesh",
            name="mug_rack_mug",
            file=(mug_asset_dir / "mug_cup.obj").as_posix(),
            scale=(1, 1, 1.2),
        )
        mug_collision_meshes = [
            scene.asset.add(
                "mesh",
                name=f"mug_rack_mug_collision_{idx}",
                file=(mug_asset_dir / f"mug_cup_collision_{idx}.obj").as_posix(),
                scale=(1, 1, 1.2),
            )
            for idx in range(32)
        ]

        rack_body = scene.worldbody.add(
            "body",
            name="rack",
            pos=tuple(self.rack_pos),
            quat=tuple(self.rack_quat),
        )
        rack_body.add(
            "geom",
            name="rack_visual_geom",
            type="mesh",
            mesh=rack_mesh,
            material=rack_material,
            group=2,
            contype=0,
            conaffinity=0,
            mass=0,
        )
        for idx, mesh in enumerate(rack_collision_meshes):
            rack_body.add(
                "geom",
                name=f"rack_collision_geom_{idx}",
                type="mesh",
                mesh=mesh,
                group=3,
                condim=6,
                priority=1,
                solref=(0.01, 1),
                solimp=(0.99, 0.999, 0.001, 0.5, 1),
            )

        mug_body = scene.worldbody.add(
            "body",
            name="mug",
            pos=tuple(self.mug_pos),
            quat=tuple(self.mug_quat),
        )
        mug_body.add("joint", name="mug_joint", type="free", damping=0.005)
        mug_body.add(
            "geom",
            name="mug_visual_geom",
            type="mesh",
            mesh=mug_mesh,
            material=mug_material,
            group=2,
            contype=0,
            conaffinity=0,
            mass=0,
        )
        for idx, mesh in enumerate(mug_collision_meshes):
            mug_body.add(
                "geom",
                name=f"mug_collision_geom_{idx}",
                type="mesh",
                mesh=mesh,
                group=3,
                condim=6,
                priority=1,
                solref=(0.01, 1),
                solimp=(0.99, 0.999, 0.001, 0.5, 1),
            )
        if self.show_target:
            target_body = scene.worldbody.add(
                "body",
                name="mug_target",
                pos=tuple(self.mug_pos),
                quat=tuple(self.mug_quat),
            )
            target_body.add(
                "geom",
                name="mug_target_visual_geom",
                type="mesh",
                mesh=mug_mesh,
                material=mug_material,
                group=2,
                contype=0,
                conaffinity=0,
                rgba=(1, 1, 1, 0.25),
            )
        return scene

    def post_compile(self, env: Any) -> None:
        self._rack_body_id = env.find_body_id(("rack/rack", "rack"))
        self._mug_body_id = env.find_body_id(("mug", "mug/body", "body"))
        self._target_body_id = env.find_body_id(("mug_target", "mug_target/body"))
        self._mug_joint_id = env.find_joint_id(("mug_joint", "mug/joint", "joint"))
        if self._mug_joint_id >= 0:
            self._mug_qpos_addr = int(env.model.jnt_qposadr[self._mug_joint_id])
            self._mug_dof_addr = int(env.model.jnt_dofadr[self._mug_joint_id])

        self._rack_geom_ids = env.collect_geoms_under_body_prefix(
            "rack",
            collision_only=True,
        )
        self._mug_geom_ids = env.collect_geoms_under_body_prefix(
            "mug",
            collision_only=True,
        )

    def initialize_episode(self, env: Any) -> None:
        self.rack_pos, self.rack_quat, self.mug_pos, self.mug_quat = (
            self._sample_reset_pose(env)
        )

        if self._mug_qpos_addr >= 0:
            env.data.qpos[self._mug_qpos_addr : self._mug_qpos_addr + 3] = self.mug_pos
            env.data.qpos[self._mug_qpos_addr + 3 : self._mug_qpos_addr + 7] = (
                self.mug_quat
            )
        if self._mug_dof_addr >= 0:
            env.data.qvel[self._mug_dof_addr : self._mug_dof_addr + 6] = 0.0

        if self._rack_body_id >= 0:
            env.model.body_pos[self._rack_body_id] = self.rack_pos
            env.model.body_quat[self._rack_body_id] = self.rack_quat
        if self._target_body_id >= 0:
            env.model.body_pos[self._target_body_id] = self.mug_pos
            env.model.body_quat[self._target_body_id] = self.mug_quat

        mujoco.mj_forward(env.model, env.data)

    def compute_observation(self, env: Any) -> dict[str, np.ndarray]:
        rack = self.rack_pose(env)
        return {
            "rack_pose": np.concatenate([rack[:2], rack[3:]]).astype(np.float32),
            "mug_pose": self.mug_pose(env).astype(np.float32),
        }

    def compute_reward(self, env: Any) -> float:
        return 1.0 if self.is_success(env) else 0.0

    def terminate_episode(self, env: Any) -> bool:
        return self.is_success(env)

    def is_success(self, env: Any) -> bool:
        return self._is_assembled(env)

    def _is_assembled(self, env: Any) -> bool:
        if not self._assembly_candidates:
            return False
        rack_mat = pose_to_mat(self.rack_pose(env))
        mug_mat = pose_to_mat(self.mug_pose(env))
        rel_mat = np.linalg.inv(rack_mat) @ mug_mat
        return bool(
            any(
                is_similar_pose(
                    candidate,
                    rel_mat,
                    pos_threshold=self.assembly_pos_threshold,
                    ori_bound=self.assembly_ori_bound,
                )
                for candidate in self._assembly_candidates
            )
        )

    def get_info(self, env: Any) -> dict[str, Any]:
        return {
            "rack_pose": self.rack_pose(env),
            "mug_pose": self.mug_pose(env),
            "mug_rack_relative_pos": self.mug_pose(env)[:3] - self.rack_pose(env)[:3],
            "mug_rack_contact": self.mug_rack_contact(env),
            "success": self.is_success(env),
        }

    def robot_excluded_geom_ids(self, env: Any) -> set[int]:
        return set(self._rack_geom_ids) | set(self._mug_geom_ids)

    def rack_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._rack_body_id)

    def mug_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._mug_body_id)

    def mug_rack_contact(self, env: Any) -> bool:
        if not self._rack_geom_ids or not self._mug_geom_ids:
            return False
        return env.check_contact_geom_ids(self._mug_geom_ids, self._rack_geom_ids)

    def _sample_reset_pose(
        self, env: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.xy_range == 0.0 and self.yaw_range == 0.0:
            return (
                self.base_rack_pos.copy(),
                self.base_rack_quat.copy(),
                self.base_mug_pos.copy(),
                self.base_mug_quat.copy(),
            )

        rng = getattr(env, "np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        rack_pos, rack_quat = self._sample_part_pose(
            rng,
            self.base_rack_pos,
            self.base_rack_quat,
            0,
            -self.yaw_range / 2,
            self.yaw_range / 2,
        )
        mug_pos, mug_quat = self._sample_part_pose(
            rng,
            self.base_mug_pos,
            self.base_mug_quat,
            self.xy_range,
            0,
            self.yaw_range,
        )
        return rack_pos, rack_quat, mug_pos, mug_quat

    def _sample_part_pose(
        self,
        rng: np.random.Generator,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        xy_range: float,
        yaw_range_min: float,
        yaw_range_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos = base_pos.copy()
        pos[:2] += rng.uniform(-xy_range, xy_range, size=2)
        yaw_delta = float(rng.uniform(yaw_range_min, yaw_range_max))
        quat = quat_mul(np.asarray(yaw_quat(yaw_delta)), base_quat)
        quat = quat / np.linalg.norm(quat)
        return pos, quat


__all__ = ["MugRackTask"]
