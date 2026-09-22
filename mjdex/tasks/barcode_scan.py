"""Barcode-scan task for MjDex environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.robots import ASSET_ROOT
from mjdex.tasks.base import Task
from mjdex.tasks.relocate import (
    YCB_ASSET_ROOT,
    YCB_INIT_HEIGHT,
    YCB_OBJECTS,
    YCB_YAW_RANGE_DEG,
)
from mjdex.transform import normalize_quat, quat_mul, yaw_quat

BARCODE_SCANNER_ASSET_ROOT = ASSET_ROOT / "barcode_scanner"
BARCODE_SCAN_OBJECTS = {k: v for k, v in YCB_OBJECTS.items() if k != "banana"}


class BarcodeScanTask(Task):
    """YCB object on the left, barcode scanner on the right; both randomised each reset."""

    def __init__(
        self,
        object_id: str = "meat",
        scanner_pos: tuple[float, float, float] = (0.40, -0.20, 0.003),
        scanner_quat: tuple[float, float, float, float] | None = None,
        scanner_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
            (0.30, 0.45),
            (-0.25, -0.15),
        ),
        object_xy_range: tuple[tuple[float, float], tuple[float, float]] = (
            (0.30, 0.50),
            (0.10, 0.30),
        ),
        reset_randomness: str = "low",
    ) -> None:
        if object_id not in BARCODE_SCAN_OBJECTS:
            raise ValueError(
                f"Unknown object '{object_id}'. Options: {sorted(BARCODE_SCAN_OBJECTS)}"
            )

        self.object_id = object_id
        self.object_name = YCB_OBJECTS[object_id]

        no_target_xml = (
            YCB_ASSET_ROOT / self.object_name / f"{self.object_name}_no_target.xml"
        )
        regular_xml = YCB_ASSET_ROOT / self.object_name / f"{self.object_name}.xml"
        self.object_xml = no_target_xml if no_target_xml.exists() else regular_xml

        self.base_scanner_pos = np.asarray(scanner_pos, dtype=np.float64)

        # Default: DualDex hard-variant table orientation + 90° yaw so the
        # scanner barrel faces the robot workspace (positive-x direction).
        if scanner_quat is None:
            _q_base = np.array([0.01242572, 0.00272717, 0.96985322, 0.24335761])
            _q_yaw90 = np.asarray(yaw_quat(-np.pi / 2), dtype=np.float64)
            self.base_scanner_quat = normalize_quat(quat_mul(_q_yaw90, _q_base))
        else:
            self.base_scanner_quat = normalize_quat(
                np.asarray(scanner_quat, dtype=np.float64)
            )

        self.scanner_pos = self.base_scanner_pos.copy()
        self.scanner_quat = self.base_scanner_quat.copy()

        self.scanner_xy_range = (
            tuple(float(v) for v in scanner_xy_range[0]),
            tuple(float(v) for v in scanner_xy_range[1]),
        )
        self.object_xy_range = (
            tuple(float(v) for v in object_xy_range[0]),
            tuple(float(v) for v in object_xy_range[1]),
        )
        self.base_object_pos = np.array(
            [
                float(np.mean(object_xy_range[0])),
                float(np.mean(object_xy_range[1])),
                YCB_INIT_HEIGHT[object_id],
            ],
            dtype=np.float64,
        )
        self.base_object_quat = np.asarray(yaw_quat(0.0), dtype=np.float64)
        self.object_pos = self.base_object_pos.copy()
        self.object_quat = self.base_object_quat.copy()
        self.reset_randomness = reset_randomness

        self._object_body_id = -1
        self._scanner_body_id = -1
        self._object_joint_id = -1
        self._object_qpos_addr = -1
        self._object_dof_addr = -1
        self._scanner_joint_id = -1
        self._scanner_qpos_addr = -1
        self._scanner_dof_addr = -1
        self._object_geom_ids: set[int] = set()
        self._scanner_geom_ids: set[int] = set()

    def build_mjcf(self, scene: mjcf.RootElement) -> mjcf.RootElement:
        self._build_ycb_object(scene)
        self._build_barcode_scanner(scene)
        return scene

    def _build_ycb_object(self, scene: mjcf.RootElement) -> None:
        root = ET.parse(self.object_xml).getroot()
        asset_dir = self.object_xml.parent / "assets"

        assets: dict[tuple[str, str], mjcf.Element] = {}
        for asset_xml in root.findall("./asset/*"):
            tag = asset_xml.tag
            source_name = asset_xml.get("name") or Path(asset_xml.get("file", "")).stem
            kwargs = self._element_kwargs(asset_xml, skip={"name", "file", "texture"})
            kwargs["name"] = f"bscan_obj_{source_name}"
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
            raise ValueError(
                f"YCB XML must contain a body named 'body': {self.object_xml}"
            )

        object_body = scene.worldbody.add(
            "body",
            name="bscan_object",
            pos=tuple(self.object_pos),
            quat=tuple(self.object_quat),
        )
        inertial_xml = object_xml.find("inertial")
        if inertial_xml is not None:
            object_body.add("inertial", **self._element_kwargs(inertial_xml))
        joint_xml = object_xml.find("joint")
        if joint_xml is None:
            raise ValueError(f"YCB object must contain a free joint: {self.object_xml}")
        object_body.add(
            "joint",
            name="bscan_object_joint",
            **self._element_kwargs(joint_xml, skip={"name"}),
        )
        for index, geom_xml in enumerate(object_xml.findall("geom")):
            self._add_geom(
                object_body,
                geom_xml,
                defaults,
                assets,
                name=f"bscan_object_geom_{index}",
            )

    def _build_barcode_scanner(self, scene: mjcf.RootElement) -> None:
        scanner_xml_path = BARCODE_SCANNER_ASSET_ROOT / "barcode_scanner.xml"
        root = ET.parse(scanner_xml_path).getroot()
        asset_dir = BARCODE_SCANNER_ASSET_ROOT / "assets"

        assets: dict[tuple[str, str], mjcf.Element] = {}
        for asset_xml in root.findall("./asset/*"):
            tag = asset_xml.tag
            source_name = asset_xml.get("name") or Path(asset_xml.get("file", "")).stem
            kwargs = self._element_kwargs(asset_xml, skip={"name", "file", "texture"})
            kwargs["name"] = f"bscan_scanner_{source_name}"
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
        scanner_xml = root.find("./worldbody/body[@name='body']")
        if scanner_xml is None:
            raise ValueError(
                f"Barcode scanner XML must contain a body named 'body': {scanner_xml_path}"
            )

        scanner_body = scene.worldbody.add(
            "body",
            name="bscan_scanner",
            pos=tuple(self.scanner_pos),
            quat=tuple(self.scanner_quat),
        )
        # Free joint so initialize_episode can place the scanner at a random position.
        scanner_body.add(
            "joint", name="bscan_scanner_joint", type="free", damping="0.001"
        )
        for index, geom_xml in enumerate(scanner_xml.findall("geom")):
            self._add_geom(
                scanner_body,
                geom_xml,
                defaults,
                assets,
                name=f"bscan_scanner_geom_{index}",
            )

    def post_compile(self, env: Any) -> None:
        self._object_body_id = env.find_body_id(("bscan_object",))
        self._scanner_body_id = env.find_body_id(("bscan_scanner",))

        self._object_joint_id = env.find_joint_id(("bscan_object_joint",))
        if self._object_joint_id >= 0:
            self._object_qpos_addr = int(env.model.jnt_qposadr[self._object_joint_id])
            self._object_dof_addr = int(env.model.jnt_dofadr[self._object_joint_id])

        self._scanner_joint_id = env.find_joint_id(("bscan_scanner_joint",))
        if self._scanner_joint_id >= 0:
            self._scanner_qpos_addr = int(env.model.jnt_qposadr[self._scanner_joint_id])
            self._scanner_dof_addr = int(env.model.jnt_dofadr[self._scanner_joint_id])

        self._object_geom_ids = env.collect_geoms_under_body_prefix(
            "bscan_object", collision_only=True
        )
        self._scanner_geom_ids = env.collect_geoms_under_body_prefix(
            "bscan_scanner", collision_only=True
        )

    def initialize_episode(self, env: Any) -> None:
        rng = getattr(env, "np_random", None) or np.random.default_rng()

        # --- YCB object ---
        if self.reset_randomness in {"none", "fixed"}:
            self.object_pos = self.base_object_pos.copy()
            self.object_quat = self.base_object_quat.copy()
        else:
            object_x = float(rng.uniform(*self.object_xy_range[0]))
            object_y = float(rng.uniform(*self.object_xy_range[1]))
            self.object_pos = np.array(
                [object_x, object_y, YCB_INIT_HEIGHT[self.object_id]], dtype=np.float64
            )
            yaw_low, yaw_high = YCB_YAW_RANGE_DEG[self.object_id]
            self.object_quat = np.asarray(
                yaw_quat(np.deg2rad(float(rng.uniform(yaw_low, yaw_high)))),
                dtype=np.float64,
            )

        if self._object_qpos_addr >= 0:
            env.data.qpos[self._object_qpos_addr : self._object_qpos_addr + 3] = (
                self.object_pos
            )
            env.data.qpos[self._object_qpos_addr + 3 : self._object_qpos_addr + 7] = (
                self.object_quat
            )
        if self._object_dof_addr >= 0:
            env.data.qvel[self._object_dof_addr : self._object_dof_addr + 6] = 0.0

        # --- Barcode scanner ---
        if self.reset_randomness in {"none", "fixed"}:
            self.scanner_pos = self.base_scanner_pos.copy()
            self.scanner_quat = self.base_scanner_quat.copy()
        else:
            scanner_x = float(rng.uniform(*self.scanner_xy_range[0]))
            scanner_y = float(rng.uniform(*self.scanner_xy_range[1]))
            self.scanner_pos = np.array(
                [scanner_x, scanner_y, self.base_scanner_pos[2]], dtype=np.float64
            )
            # Small yaw jitter (±30°) so orientation varies slightly each reset.
            yaw_offset = float(rng.uniform(-np.pi / 6, np.pi / 6))
            q_delta = np.asarray(yaw_quat(yaw_offset), dtype=np.float64)
            self.scanner_quat = normalize_quat(
                quat_mul(q_delta, self.base_scanner_quat)
            )

        if self._scanner_qpos_addr >= 0:
            env.data.qpos[self._scanner_qpos_addr : self._scanner_qpos_addr + 3] = (
                self.scanner_pos
            )
            env.data.qpos[self._scanner_qpos_addr + 3 : self._scanner_qpos_addr + 7] = (
                self.scanner_quat
            )
        if self._scanner_dof_addr >= 0:
            env.data.qvel[self._scanner_dof_addr : self._scanner_dof_addr + 6] = 0.0

        mujoco.mj_forward(env.model, env.data)

    def compute_observation(self, env: Any) -> dict[str, np.ndarray]:
        return {
            "object_pose": self.object_pose(env).astype(np.float32),
            "scanner_pose": self.scanner_pose(env).astype(np.float32),
        }

    def compute_reward(self, env: Any) -> float:
        # TODO: Define reward for successfully scanning the barcode.
        return 0.0

    def terminate_episode(self, env: Any) -> bool:
        # TODO: Terminate when barcode scan success predicate is available.
        return False

    def get_info(self, env: Any) -> dict[str, Any]:
        return {
            "object_pose": self.object_pose(env),
            "scanner_pose": self.scanner_pose(env),
            "object_scanner_relative_pos": (
                self.object_pose(env)[:3] - self.scanner_pose(env)[:3]
            ),
            "object_scanner_contact": self.object_scanner_contact(env),
        }

    def robot_excluded_geom_ids(self, env: Any) -> set[int]:
        return set(self._object_geom_ids) | set(self._scanner_geom_ids)

    def object_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._object_body_id)

    def scanner_pose(self, env: Any) -> np.ndarray:
        return env.body_pose(self._scanner_body_id)

    def object_scanner_contact(self, env: Any) -> bool:
        if not self._object_geom_ids or not self._scanner_geom_ids:
            return False
        return env.check_contact_geom_ids(self._object_geom_ids, self._scanner_geom_ids)

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
            self._element_kwargs(geom_xml, skip={"name", "class", "mesh", "material"})
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


__all__ = ["BarcodeScanTask", "BARCODE_SCAN_OBJECTS"]
