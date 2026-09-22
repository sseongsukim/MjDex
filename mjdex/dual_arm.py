"""Dual-arm MjDex environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.controller.diff_ik import DiffIKController
from mjdex.core import MjDexEnv
from mjdex.gains import apply_tuned_arm_gains
from mjdex.robots import (
    build_dual_robot_scene,
    build_robot_model,
    default_dual_arm_mounts,
    mounted_robot_base_quat,
)
from mjdex.tasks.base import Task

LEFT_FR3_HOME_QPOS = np.array(
    [-0.00751, 0.268, -0.00622, -2.4, 0.0036, 2.67, -0.151],
    dtype=np.float64,
)
RIGHT_FR3_HOME_QPOS = np.array(
    [0.00754, 0.268, 0.00619, -2.4, -0.00358, 2.67, -0.0905],
    dtype=np.float64,
)
LEFT_UR5E_HOME_QPOS = np.array(
    [0.588, -3.12, 1.87, -1.77, 0.5, -1.41],
    dtype=np.float64,
)
RIGHT_UR5E_HOME_QPOS = np.array(
    [-3.75, -0.0202, -1.85, -1.42, -0.336, -1.69],
    dtype=np.float64,
)
LEFT_UR5E_HAND_HOME_QPOS = np.array(
    [0.576, -3.34, 2.1, 2.81, -1.57, -2.15],
    dtype=np.float64,
)
RIGHT_UR5E_HAND_HOME_QPOS = np.array(
    [-3.72, 0.199, -2.1, 0.329, 1.57, -0.995],
    dtype=np.float64,
)
LEFT_UR5E_ALLEGRO_HOME_QPOS = np.array(
    [0.5770109, -3.34004954, 2.10008739, 2.8107588, -1.57079646, -5.28930279],
    dtype=np.float64,
)
RIGHT_UR5E_ALLEGRO_HOME_QPOS = np.array(
    [-3.72, 0.199, -2.1, 0.329, 1.57, -0.995],
    dtype=np.float64,
)
ROBOT_JOINT_NAMES = {
    "fr3": [f"fr3_joint{i}" for i in range(1, 8)],
    "ur5e": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
}
ROBOT_HOME_QPOS = {
    "fr3": {
        "left": LEFT_FR3_HOME_QPOS,
        "right": RIGHT_FR3_HOME_QPOS,
    },
    "ur5e": {
        "left": LEFT_UR5E_HOME_QPOS,
        "right": RIGHT_UR5E_HOME_QPOS,
    },
}
HAND_DOF_BY_MODEL = {
    "allegro_left": 16,
    "allegro_right": 16,
    "inspire_left": 6,
    "inspire_right": 6,
    "sharpa_left": 22,
    "sharpa_right": 22,
}


@dataclass
class ArmIds:
    prefix: str
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray
    gripper_actuator_ids: np.ndarray
    gripper_pad_body_ids: tuple[int, int]
    ee_site_id: int
    ik: DiffIKController


class DualArmEnv(MjDexEnv):
    """Two robot arms with Cartesian IK actions.

    The action is ``[left_xyz, left_quat_wxyz, left_gripper,
    right_xyz, right_quat_wxyz, right_gripper]``. MjDex uses absolute
    ``action_type="pos"`` actions for stability.
    """

    def __init__(
        self,
        robot: str = "fr3",
        gripper: str = "robotiq_2f85",
        action_type: str = "pos",
        position_action_scale: float = 0.05,
        include_table: bool = True,
        table_pos: tuple[float, float, float] | None = None,
        task: Task | None = None,
        mounts: (
            dict[
                str,
                tuple[tuple[float, float, float], tuple[float, float, float, float]],
            ]
            | None
        ) = None,
        **kwargs,
    ) -> None:
        if robot not in ROBOT_JOINT_NAMES:
            raise ValueError(
                f"DualArmEnv supports {sorted(ROBOT_JOINT_NAMES)}, got '{robot}'."
            )
        if action_type not in ("pos", "delta"):
            raise ValueError(
                f"Unsupported action_type '{action_type}'. Use 'pos' or 'delta'."
            )

        self.robot_name = robot
        self.gripper_name = gripper
        self.action_type = "pos"
        self.position_action_scale = float(position_action_scale)
        self.include_table = include_table
        self.table_pos = table_pos
        self.task = task
        self.mounts = mounts or default_dual_arm_mounts(robot)
        self._arms: dict[str, ArmIds] = {}
        super().__init__(**kwargs)

    @property
    def action_space(self):
        if self.action_type == "pos":
            low_one = [-np.inf, -np.inf, -np.inf, -1, -1, -1, -1, -1]
            high_one = [np.inf, np.inf, np.inf, 1, 1, 1, 1, 1]
            return gym.spaces.Box(
                low=np.asarray([*low_one, *low_one], dtype=np.float32),
                high=np.asarray([*high_one, *high_one], dtype=np.float32),
                dtype=np.float32,
            )
        return gym.spaces.Box(
            low=-np.ones(16, dtype=np.float32),
            high=np.ones(16, dtype=np.float32),
            dtype=np.float32,
        )

    def build_mjcf(self) -> mjcf.RootElement:
        scene = build_dual_robot_scene(
            robot=self.robot_name,
            gripper=self.gripper_name,
            include_table=self.include_table,
            table_pos=self.table_pos,
            mounts=self.mounts,
        )
        if self.task is not None:
            scene = self.task.build_mjcf(scene)
        return scene

    def post_compile(self) -> None:
        self._arms = {
            side: self._build_arm_ids(side, pos, quat)
            for side, (pos, quat) in self.mounts.items()
        }
        for arm in self._arms.values():
            apply_tuned_arm_gains(self.model, self.robot_name, arm.actuator_ids)
        if self.task is not None:
            self.task.post_compile(self)

    def initialize_episode(self) -> None:
        home_qpos_by_side = ROBOT_HOME_QPOS[self.robot_name]
        for side, arm in self._arms.items():
            home_qpos = home_qpos_by_side[side]
            self.data.qpos[arm.qpos_ids] = home_qpos
            self.data.qvel[arm.dof_ids] = 0.0
            self.data.ctrl[arm.actuator_ids] = home_qpos
            if arm.gripper_actuator_ids.size:
                ctrlrange = self.model.actuator_ctrlrange[arm.gripper_actuator_ids]
                self.data.ctrl[arm.gripper_actuator_ids] = ctrlrange[:, 0]
        mujoco.mj_forward(self.model, self.data)
        self._apply_gravity_compensation(
            *(arm.dof_ids for arm in self._arms.values())
        )
        if self.task is not None:
            self.task.initialize_episode(self)

    def compute_observation(self) -> dict[str, np.ndarray]:
        observation: dict[str, np.ndarray] = {}
        uses_hands = hasattr(self, "_hand_dof_by_side")
        for side, arm in self._arms.items():
            observation[f"{side}_arm_joint_position"] = self.data.qpos[
                arm.qpos_ids
            ].astype(np.float32)
            observation[f"{side}_arm_joint_velocity"] = self.data.qvel[
                arm.dof_ids
            ].astype(np.float32)
            observation[f"{side}_ee_pose"] = self.ee_pose(side).astype(np.float32)
            if uses_hands:
                hand_qpos_ids, hand_dof_ids = self._hand_state_ids(arm)
                observation[f"{side}_hand_joint_position"] = self.data.qpos[
                    hand_qpos_ids
                ].astype(np.float32)
                observation[f"{side}_hand_joint_velocity"] = self.data.qvel[
                    hand_dof_ids
                ].astype(np.float32)
            else:
                gripper_width = self.gripper_distance(side)
                if gripper_width is not None:
                    observation[f"{side}_gripper_width"] = np.asarray(
                        [gripper_width], dtype=np.float32
                    )
        if self.task is not None:
            observation.update(self.task.compute_observation(self))
        return observation

    def compute_reward(self) -> float:
        if self.task is None:
            return 0.0
        return float(self.task.compute_reward(self))

    def terminate_episode(self) -> bool:
        if self.task is None:
            return False
        return bool(self.task.terminate_episode(self))

    def get_reset_info(self) -> dict[str, Any]:
        if self.task is None:
            return {}
        return self.task.get_info(self)

    def get_step_info(self) -> dict[str, Any]:
        info = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "left_ee_pose": self.ee_pose("left"),
            "right_ee_pose": self.ee_pose("right"),
        }
        for side, arm in self._arms.items():
            info[f"{side}_arm_qpos"] = self.data.qpos[arm.qpos_ids].copy()
            info[f"{side}_arm_qvel"] = self.data.qvel[arm.dof_ids].copy()
            gripper_distance = self.gripper_distance(side)
            if gripper_distance is not None:
                info[f"{side}_gripper_distance"] = gripper_distance
        if self.task is not None:
            info.update(self.task.get_info(self))
        return info

    def ee_pose(self, side: str) -> np.ndarray:
        arm = self._arms[side]
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[arm.ee_site_id])
        return np.concatenate([self.data.site_xpos[arm.ee_site_id].copy(), quat])

    def gripper_distance(self, side: str) -> float | None:
        left_body_id, right_body_id = self._arms[side].gripper_pad_body_ids
        if left_body_id < 0 or right_body_id < 0:
            return None
        return float(
            np.linalg.norm(self.data.xpos[left_body_id] - self.data.xpos[right_body_id])
        )

    def set_control(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (16,):
            raise ValueError(
                f"Expected dual-arm action shape (16,), got {action.shape}."
            )
        for i, side in enumerate(("left", "right")):
            self._set_arm_control(side, action[i * 8 : (i + 1) * 8])

    def _set_arm_control(self, side: str, action: np.ndarray) -> None:
        arm = self._arms[side]
        ee_pose = self.ee_pose(side)
        action_pos = action[:3]
        action_quat = self._normalize_quat(action[3:7])
        gripper_action = np.clip(action[7], -1.0, 1.0)

        if self.action_type == "delta":
            target_pos = (
                ee_pose[:3]
                + np.clip(action_pos, -1.0, 1.0) * self.position_action_scale
            )
            target_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(target_quat, ee_pose[3:], action_quat)
        else:
            target_pos = action_pos.copy()
            target_quat = action_quat.copy()

        target_quat /= np.linalg.norm(target_quat)
        qpos_target = arm.ik.solve(
            pos=target_pos,
            quat=target_quat,
            curr_qpos=self.data.qpos[arm.qpos_ids],
        )
        ctrlrange = self.model.actuator_ctrlrange[arm.actuator_ids]
        self.data.ctrl[arm.actuator_ids] = np.clip(
            qpos_target,
            ctrlrange[:, 0],
            ctrlrange[:, 1],
        )

        if arm.gripper_actuator_ids.size:
            grip_range = self.model.actuator_ctrlrange[arm.gripper_actuator_ids]
            curr = self.data.ctrl[arm.gripper_actuator_ids]
            denom = np.maximum(grip_range[:, 1] - grip_range[:, 0], 1e-6)
            opening = (curr - grip_range[:, 0]) / denom
            if self.action_type == "delta":
                opening -= gripper_action
            else:
                opening = np.full_like(opening, 0.5 * (1.0 - gripper_action))
            opening = np.clip(opening, 0.0, 1.0)
            self.data.ctrl[arm.gripper_actuator_ids] = (
                grip_range[:, 0] + opening * denom
            )
        self._apply_gravity_compensation(arm.dof_ids)

    def _build_arm_ids(
        self,
        side: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> ArmIds:
        prefix = f"{side}_{self.robot_name}"
        joint_names = ROBOT_JOINT_NAMES[self.robot_name]
        joint_ids = np.asarray(
            [
                self._name2id(mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/{name}")
                for name in joint_names
            ],
            dtype=np.int32,
        )
        qpos_ids = self.model.jnt_qposadr[joint_ids].astype(np.int32)
        dof_ids = self.model.jnt_dofadr[joint_ids].astype(np.int32)
        joint_id_set = set(joint_ids.tolist())
        actuator_ids = np.asarray(
            [
                actuator_id
                for actuator_id in range(self.model.nu)
                if int(self.model.actuator_trnid[actuator_id, 0]) in joint_id_set
                and "/2f85/"
                not in (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        actuator_id,
                    )
                    or ""
                )
            ],
            dtype=np.int32,
        )
        gripper_actuator_ids = np.asarray(
            [
                actuator_id
                for actuator_id in range(self.model.nu)
                if actuator_id not in set(actuator_ids.tolist())
                and (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                    )
                    or ""
                ).startswith(prefix)
            ],
            dtype=np.int32,
        )
        site_id = self._name2id(mujoco.mjtObj.mjOBJ_SITE, f"{prefix}/attachment_site")
        ik_mjcf = build_robot_model(
            self.robot_name,
            base_pos=pos,
            base_quat=mounted_robot_base_quat(self.robot_name, quat),
        )
        ik_model = mujoco.MjModel.from_xml_string(
            ik_mjcf.to_xml_string(),
            assets=ik_mjcf.get_assets(),
        )
        return ArmIds(
            prefix=prefix,
            joint_ids=joint_ids,
            qpos_ids=qpos_ids,
            dof_ids=dof_ids,
            actuator_ids=actuator_ids,
            gripper_actuator_ids=gripper_actuator_ids,
            gripper_pad_body_ids=(
                self._find_gripper_pad_body_id(prefix, "left_pad"),
                self._find_gripper_pad_body_id(prefix, "right_pad"),
            ),
            ee_site_id=site_id,
            ik=DiffIKController(model=ik_model, sites=["attachment_site"]),
        )

    def _name2id(self, objtype: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, objtype, name)
        if obj_id < 0:
            raise ValueError(f"Could not find {objtype.name} '{name}'.")
        return int(obj_id)

    def _find_gripper_pad_body_id(self, prefix: str, suffix: str) -> int:
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if (
                name is not None
                and prefix in name
                and (name == suffix or name.endswith("/" + suffix))
            ):
                return int(body_id)
        return -1

    def _apply_gravity_compensation(self, *dof_groups: np.ndarray) -> None:
        dof_ids = [np.asarray(ids, dtype=np.int32) for ids in dof_groups if ids.size]
        if not dof_ids:
            return
        controlled_dof_ids = np.unique(np.concatenate(dof_ids))
        self.data.qfrc_applied[controlled_dof_ids] = self.data.qfrc_bias[
            controlled_dof_ids
        ]

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm


class DualArmRobotGripperEnv(DualArmEnv):
    """Default dual robot + Robotiq environment."""

    def __init__(self, robot: str = "fr3", **kwargs) -> None:
        super().__init__(robot=robot, gripper="robotiq_2f85", **kwargs)


class DualArmHandEnv(DualArmEnv):
    """Dual UR5e + Inspire RH56F1 hands with Cartesian arm and hand qpos control."""

    def __init__(
        self,
        robot: str = "ur5e",
        left_hand: str = "inspire_left",
        right_hand: str = "inspire_right",
        **kwargs,
    ) -> None:
        if robot != "ur5e":
            raise ValueError("Hand environments are only supported on ur5e.")
        for side, hand in (("left", left_hand), ("right", right_hand)):
            if hand not in HAND_DOF_BY_MODEL:
                raise ValueError(
                    f"Unknown {side} hand model '{hand}'. "
                    f"Options: {sorted(HAND_DOF_BY_MODEL)}"
                )
        self.left_hand_name = left_hand
        self.right_hand_name = right_hand
        self._hand_dof_by_side = {
            "left": HAND_DOF_BY_MODEL[left_hand],
            "right": HAND_DOF_BY_MODEL[right_hand],
        }
        super().__init__(
            robot=robot,
            gripper={"left": left_hand, "right": right_hand},
            **kwargs,
        )

    @property
    def action_space(self):
        if self._model is None:
            self.reset()

        lows = []
        highs = []
        for side in ("left", "right"):
            arm = self._arms[side]
            hand_range = self.model.actuator_ctrlrange[arm.gripper_actuator_ids]
            if self.action_type == "pos":
                lows.extend([-np.inf, -np.inf, -np.inf, -1, -1, -1, -1])
                highs.extend([np.inf, np.inf, np.inf, 1, 1, 1, 1])
            else:
                lows.extend([-1, -1, -1, -1, -1, -1, -1])
                highs.extend([1, 1, 1, 1, 1, 1, 1])
            lows.extend(hand_range[:, 0])
            highs.extend(hand_range[:, 1])
        return gym.spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def post_compile(self) -> None:
        super().post_compile()
        for side, arm in self._arms.items():
            hand_dof = self._hand_dof_by_side[side]
            if arm.gripper_actuator_ids.size != hand_dof:
                raise ValueError(
                    f"Expected {hand_dof} actuators for {side} "
                    f"{getattr(self, f'{side}_hand_name')}, "
                    f"got {arm.gripper_actuator_ids.size}."
                )

    def initialize_episode(self) -> None:
        if self._uses_allegro_hand():
            hand_home_qpos = {
                "left": LEFT_UR5E_ALLEGRO_HOME_QPOS,
                "right": RIGHT_UR5E_ALLEGRO_HOME_QPOS,
            }
        else:
            hand_home_qpos = {
                "left": LEFT_UR5E_HAND_HOME_QPOS,
                "right": RIGHT_UR5E_HAND_HOME_QPOS,
            }
        for side, arm in self._arms.items():
            home_qpos = hand_home_qpos[side]
            self.data.qpos[arm.qpos_ids] = home_qpos
            self.data.qvel[arm.dof_ids] = 0.0
            self.data.ctrl[arm.actuator_ids] = home_qpos

            hand_qpos_ids, hand_dof_ids = self._hand_state_ids(arm)
            hand_home = self._hand_neutral_qpos(arm)
            self.data.qpos[hand_qpos_ids] = hand_home
            self.data.qvel[hand_dof_ids] = 0.0
            self.data.ctrl[arm.gripper_actuator_ids] = hand_home
        mujoco.mj_forward(self.model, self.data)
        gravity_dof_groups = []
        for arm in self._arms.values():
            _, hand_dof_ids = self._hand_state_ids(arm)
            gravity_dof_groups.extend([arm.dof_ids, hand_dof_ids])
        self._apply_gravity_compensation(*gravity_dof_groups)
        if self.task is not None:
            self.task.initialize_episode(self)

    def _uses_allegro_hand(self) -> bool:
        return self.left_hand_name.startswith(
            "allegro_"
        ) or self.right_hand_name.startswith("allegro_")

    def set_control(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (
            sum(7 + self._hand_dof_by_side[side] for side in ("left", "right")),
        )
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected dual hand action shape {expected_shape}, got {action.shape}."
            )
        start = 0
        for side in ("left", "right"):
            width = 7 + self._hand_dof_by_side[side]
            self._set_hand_arm_control(side, action[start : start + width])
            start += width

    def _set_hand_arm_control(self, side: str, action: np.ndarray) -> None:
        arm = self._arms[side]
        ee_pose = self.ee_pose(side)
        action_pos = action[:3]
        action_quat = self._normalize_quat(action[3:7])
        hand_qpos = action[7:]

        if self.action_type == "delta":
            target_pos = (
                ee_pose[:3]
                + np.clip(action_pos, -1.0, 1.0) * self.position_action_scale
            )
            target_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(target_quat, ee_pose[3:], action_quat)
        else:
            target_pos = action_pos.copy()
            target_quat = action_quat.copy()

        target_quat /= np.linalg.norm(target_quat)
        qpos_target = arm.ik.solve(
            pos=target_pos,
            quat=target_quat,
            curr_qpos=self.data.qpos[arm.qpos_ids],
        )
        ctrlrange = self.model.actuator_ctrlrange[arm.actuator_ids]
        self.data.ctrl[arm.actuator_ids] = np.clip(
            qpos_target,
            ctrlrange[:, 0],
            ctrlrange[:, 1],
        )

        hand_range = self.model.actuator_ctrlrange[arm.gripper_actuator_ids]
        self.data.ctrl[arm.gripper_actuator_ids] = np.clip(
            hand_qpos,
            hand_range[:, 0],
            hand_range[:, 1],
        )
        _, hand_dof_ids = self._hand_state_ids(arm)
        self._apply_gravity_compensation(arm.dof_ids, hand_dof_ids)

    def hand_qpos(self, side: str) -> np.ndarray:
        hand_qpos_ids, _ = self._hand_state_ids(self._arms[side])
        return self.data.qpos[hand_qpos_ids].copy()

    def _hand_state_ids(self, arm: ArmIds) -> tuple[np.ndarray, np.ndarray]:
        joint_ids = self.model.actuator_trnid[arm.gripper_actuator_ids, 0]
        return (
            self.model.jnt_qposadr[joint_ids].astype(np.int32),
            self.model.jnt_dofadr[joint_ids].astype(np.int32),
        )

    def _hand_neutral_qpos(self, arm: ArmIds) -> np.ndarray:
        ctrlrange = self.model.actuator_ctrlrange[arm.gripper_actuator_ids]
        return np.clip(
            np.zeros(ctrlrange.shape[0], dtype=np.float64),
            ctrlrange[:, 0],
            ctrlrange[:, 1],
        )
