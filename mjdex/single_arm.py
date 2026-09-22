"""Single-arm MjDex environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf

from mjdex.controller.diff_ik import DiffIKController
from mjdex.core import MjDexEnv
from mjdex.contact import geom_can_collide
from mjdex.gains import apply_tuned_arm_gains
from mjdex.robots import (
    DEFAULT_ROBOT_BASE_POS,
    build_robot_model,
    build_single_robot_scene,
)
from mjdex.tasks.base import Task

ROBOT_DOF = {
    "ur5e": 6,
    "fr3": 7,
}
ROBOT_HOME_QPOS = {
    "ur5e": np.array(
        [-3.48, -1.79, 2.12, -1.9, -1.57, -0.342],
        dtype=np.float64,
    ),
    "fr3": np.array(
        [0.13, -0.335, -0.133, -2.65, -0.0595, 2.32, -1.53],
        dtype=np.float64,
    ),
}
UR5E_HAND_HOME_QPOS = np.array(
    [-3.60443189, -1.67645421, 2.32117981, -0.64474591, 1.10800338, -1.57083611],
    dtype=np.float64,
)

HAND_DOF_BY_MODEL = {
    "allegro_left": 16,
    "allegro_right": 16,
    "inspire_left": 6,
    "inspire_right": 6,
    "sharpa_left": 22,
    "sharpa_right": 22,
}


class SingleArmBaseEnv(MjDexEnv):
    """A minimal single robot + gripper MuJoCo environment.

    This environment is intentionally task-light. It provides a stable
    simulation shell with observations, action forwarding, home reset, and a
    small diagnostic reward. Object/task logic can be layered on top later.
    """

    def __init__(
        self,
        robot: str = "ur5e",
        gripper: str = "robotiq_2f85",
        home_key: str = "home",
        include_table: bool = True,
        table_pos: tuple[float, float, float] | None = None,
        robot_base_pos: tuple[float, float, float] = DEFAULT_ROBOT_BASE_POS,
        task: Task | None = None,
        **kwargs,
    ) -> None:
        self.robot_name = robot
        self.gripper_name = gripper
        self.home_key = home_key
        self.include_table = include_table
        self.table_pos = table_pos
        self.robot_base_pos = robot_base_pos
        self.task = task
        self._home_key_id = -1
        self._table_geom_ids: set[int] = set()
        self._robot_geom_ids: set[int] = set()
        super().__init__(**kwargs)

    def build_mjcf(self) -> mjcf.RootElement:
        scene = build_single_robot_scene(
            robot=self.robot_name,
            gripper=self.gripper_name,
            include_table=self.include_table,
            table_pos=self.table_pos,
            robot_base_pos=self.robot_base_pos,
        )
        if self.task is not None:
            scene = self.task.build_mjcf(scene)
        return scene

    def post_compile(self) -> None:
        self._home_key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            self.home_key,
        )
        self._table_geom_ids = self.collect_geoms_under_body_prefix(
            "table",
            collision_only=True,
        )
        if self.task is not None:
            self.task.post_compile(self)
        self._robot_geom_ids = self._collect_robot_collision_geom_ids()

    def initialize_episode(self) -> None:
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if self.model.nu:
            nkey_ctrl = self.model.key_ctrl.shape[1] if self.model.nkey else 0
            if self._home_key_id >= 0 and nkey_ctrl == self.model.nu:
                self.data.ctrl[:] = self.model.key_ctrl[self._home_key_id]
            else:
                self.data.ctrl[:] = self.data.qpos[: self.model.nu]

        if self.task is not None:
            mujoco.mj_forward(self.model, self.data)
            self.task.initialize_episode(self)

    def compute_observation(self) -> dict[str, np.ndarray]:
        observation = self.base_observation()
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
            "ee_pose": self.ee_pose(),
            "robot_table_contact": self.is_robot_table_contact(),
        }
        if hasattr(self, "_arm_qpos_ids"):
            info["arm_qpos"] = self.data.qpos[self._arm_qpos_ids].copy()
            info["arm_qvel"] = self.data.qvel[self._arm_dof_ids].copy()
        if hasattr(self, "gripper_distance"):
            gripper_distance = self.gripper_distance()
            if gripper_distance is not None:
                info["gripper_distance"] = gripper_distance
        if hasattr(self, "hand_qpos"):
            info["hand_qpos"] = self.hand_qpos()
        if self.task is not None:
            info.update(self.task.get_info(self))
        return info

    def is_robot_table_contact(self) -> bool:
        """Return True if the robot or attached gripper contacts the table."""

        if not self._table_geom_ids:
            return False
        return self.check_contact_geom_ids(self._robot_geom_ids, self._table_geom_ids)

    def robot_table_contact_pairs(self) -> list[tuple[str, str, float]]:
        """Return named contact pairs between robot/gripper and table."""

        return self.contact_pairs(self._robot_geom_ids, self._table_geom_ids)

    def _collect_robot_collision_geom_ids(self) -> set[int]:
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ignored = set(self._table_geom_ids)
        if floor_id >= 0:
            ignored.add(floor_id)
        if self.task is not None:
            ignored.update(self.task.robot_excluded_geom_ids(self))

        return {
            geom_id
            for geom_id in range(self.model.ngeom)
            if geom_id not in ignored and geom_can_collide(self.model, geom_id)
        }

    def base_observation(self) -> dict[str, np.ndarray]:
        """Return robot-only state observation without task terms."""

        observation = {
            "arm_joint_position": self.data.qpos[self._arm_qpos_ids].astype(
                np.float32
            ),
            "arm_joint_velocity": self.data.qvel[self._arm_dof_ids].astype(
                np.float32
            ),
            "ee_pose": self.ee_pose().astype(np.float32),
        }
        if hasattr(self, "_hand_qpos_ids") and self._hand_qpos_ids.size:
            observation["hand_joint_position"] = self.data.qpos[
                self._hand_qpos_ids
            ].astype(np.float32)
            observation["hand_joint_velocity"] = self.data.qvel[
                self._hand_dof_ids
            ].astype(np.float32)
        else:
            gripper_width = self.gripper_distance()
            if gripper_width is not None:
                observation["gripper_width"] = np.asarray(
                    [gripper_width], dtype=np.float32
                )
        return observation


class SingleArmEnv(SingleArmBaseEnv):
    """Single-arm environment with Cartesian pose IK control.

    The public action is ``[xyz, quat_wxyz, gripper]``. MjDex uses absolute
    ``action_type="pos"`` actions for stability.
    """

    def __init__(
        self,
        robot: str = "ur5e",
        gripper: str = "robotiq_2f85",
        ee_site: str = "attachment_site",
        action_type: str = "pos",
        position_action_scale: float = 0.05,
        workspace_bounds: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None,
        **kwargs,
    ) -> None:
        if action_type not in ("pos", "delta"):
            raise ValueError(
                f"Unsupported action_type '{action_type}'. Use 'pos' or 'delta'."
            )

        self.ee_site = ee_site
        self.action_type = "pos"
        self.position_action_scale = float(position_action_scale)
        self.workspace_bounds = (
            None
            if workspace_bounds is None
            else np.asarray(workspace_bounds, dtype=np.float64)
        )

        robot_base_pos = kwargs.get("robot_base_pos", DEFAULT_ROBOT_BASE_POS)
        ik_mjcf = build_robot_model(robot, base_pos=robot_base_pos)
        ik_model = mujoco.MjModel.from_xml_string(
            ik_mjcf.to_xml_string(),
            assets=ik_mjcf.get_assets(),
        )
        self._ik = DiffIKController(model=ik_model, sites=[ee_site])

        self._arm_dof = ROBOT_DOF[robot]
        self._arm_joint_ids = np.empty(0, dtype=np.int32)
        self._arm_qpos_ids = np.empty(0, dtype=np.int32)
        self._arm_dof_ids = np.empty(0, dtype=np.int32)
        self._arm_actuator_ids = np.empty(0, dtype=np.int32)
        self._gripper_actuator_ids = np.empty(0, dtype=np.int32)
        self._gripper_pad_body_ids = (-1, -1)
        self._ee_site_id = -1

        super().__init__(robot=robot, gripper=gripper, **kwargs)

    @property
    def action_space(self):
        if self.action_type == "pos":
            xyz_low = (
                self.workspace_bounds[0]
                if self.workspace_bounds is not None
                else [-np.inf, -np.inf, -np.inf]
            )
            xyz_high = (
                self.workspace_bounds[1]
                if self.workspace_bounds is not None
                else [np.inf, np.inf, np.inf]
            )
            low = np.asarray(
                [*xyz_low, -1.0, -1.0, -1.0, -1.0, -1.0],
                dtype=np.float32,
            )
            high = np.asarray(
                [*xyz_high, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            )
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)

        return gym.spaces.Box(
            low=-np.ones(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

    def post_compile(self) -> None:
        super().post_compile()

        self._arm_joint_ids = np.arange(self._arm_dof, dtype=np.int32)
        self._arm_qpos_ids = self.model.jnt_qposadr[self._arm_joint_ids].astype(
            np.int32
        )
        self._arm_dof_ids = self.model.jnt_dofadr[self._arm_joint_ids].astype(np.int32)
        self._arm_actuator_ids = np.arange(self._arm_dof, dtype=np.int32)
        self._gripper_actuator_ids = np.arange(
            self._arm_dof,
            self.model.nu,
            dtype=np.int32,
        )
        apply_tuned_arm_gains(self.model, self.robot_name, self._arm_actuator_ids)
        self._ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.ee_site,
        )
        if self._ee_site_id < 0:
            raise ValueError(f"Could not find end-effector site '{self.ee_site}'.")
        self._gripper_pad_body_ids = self._find_gripper_pad_body_ids()

    def initialize_episode(self) -> None:
        super().initialize_episode()
        home_qpos = ROBOT_HOME_QPOS[self.robot_name]
        self.data.qpos[self._arm_qpos_ids] = home_qpos
        self.data.qvel[self._arm_dof_ids] = 0.0
        self.data.ctrl[self._arm_actuator_ids] = home_qpos
        if self._gripper_actuator_ids.size:
            ctrlrange = self.model.actuator_ctrlrange[self._gripper_actuator_ids]
            self.data.ctrl[self._gripper_actuator_ids] = ctrlrange[:, 0]
        mujoco.mj_forward(self.model, self.data)
        self._apply_gravity_compensation(self._arm_dof_ids)

    def ee_pose(self) -> np.ndarray:
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._ee_site_id])
        return np.concatenate([self.data.site_xpos[self._ee_site_id].copy(), quat])

    def gripper_distance(self) -> float | None:
        left_body_id, right_body_id = self._gripper_pad_body_ids
        if left_body_id < 0 or right_body_id < 0:
            return None
        return float(
            np.linalg.norm(self.data.xpos[left_body_id] - self.data.xpos[right_body_id])
        )

    def solve_ik(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        curr_qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        if curr_qpos is None:
            curr_qpos = self.data.qpos[self._arm_qpos_ids]
        return self._ik.solve(pos=pos, quat=quat, curr_qpos=curr_qpos)

    def set_control(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,):
            raise ValueError(
                f"Expected Cartesian IK action shape (8,), got {action.shape}."
            )

        ee_pose = self.ee_pose()
        action_pos = action[:3]
        action_quat = self._normalize_quat(action[3:7])
        gripper_action = np.clip(action[7], -1.0, 1.0)

        if self.action_type == "delta":
            action_pos = np.clip(action_pos, -1.0, 1.0)
            target_pos = ee_pose[:3] + action_pos * self.position_action_scale
            target_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(target_quat, ee_pose[3:], action_quat)
        else:
            target_pos = action_pos.copy()
            target_quat = action_quat.copy()

        if self.workspace_bounds is not None:
            np.clip(
                target_pos,
                self.workspace_bounds[0],
                self.workspace_bounds[1],
                out=target_pos,
            )

        target_quat /= np.linalg.norm(target_quat)

        qpos_target = self.solve_ik(target_pos, target_quat)
        arm_ctrlrange = self.model.actuator_ctrlrange[self._arm_actuator_ids]
        self.data.ctrl[self._arm_actuator_ids] = np.clip(
            qpos_target,
            arm_ctrlrange[:, 0],
            arm_ctrlrange[:, 1],
        )

        if self._gripper_actuator_ids.size:
            ctrlrange = self.model.actuator_ctrlrange[self._gripper_actuator_ids]
            curr_ctrl = self.data.ctrl[self._gripper_actuator_ids]
            denom = np.maximum(ctrlrange[:, 1] - ctrlrange[:, 0], 1e-6)
            opening = (curr_ctrl - ctrlrange[:, 0]) / denom
            if self.action_type == "delta":
                opening = opening - gripper_action
            else:
                opening = np.full_like(opening, 0.5 * (1.0 - gripper_action))
            opening = np.clip(opening, 0.0, 1.0)
            self.data.ctrl[self._gripper_actuator_ids] = (
                ctrlrange[:, 0] + opening * denom
            )
        self._apply_gravity_compensation(self._arm_dof_ids)

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def _find_gripper_pad_body_ids(self) -> tuple[int, int]:
        return (
            self._find_body_id_by_suffix("left_pad"),
            self._find_body_id_by_suffix("right_pad"),
        )

    def _find_body_id_by_suffix(self, suffix: str) -> int:
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name == suffix or (name is not None and name.endswith("/" + suffix)):
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


class SingleArmJointEnv(SingleArmEnv):
    """Single-arm environment with direct joint position control.

    Action: ``[q0, ..., q_{n-1}, gripper]`` where q_i are absolute joint
    positions (radians) clipped to actuator limits, and gripper is in [-1, 1].
    """

    @property
    def action_space(self):
        if self._model is None:
            self.reset()
        arm_range = self.model.actuator_ctrlrange[self._arm_actuator_ids]
        low = np.concatenate([arm_range[:, 0], [-1.0]]).astype(np.float32)
        high = np.concatenate([arm_range[:, 1], [1.0]]).astype(np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def set_control(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (self._arm_dof + 1,)
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected joint position action shape {expected_shape}, got {action.shape}."
            )

        joint_pos = action[: self._arm_dof]
        gripper_action = float(np.clip(action[self._arm_dof], -1.0, 1.0))

        arm_ctrlrange = self.model.actuator_ctrlrange[self._arm_actuator_ids]
        self.data.ctrl[self._arm_actuator_ids] = np.clip(
            joint_pos, arm_ctrlrange[:, 0], arm_ctrlrange[:, 1]
        )

        if self._gripper_actuator_ids.size:
            ctrlrange = self.model.actuator_ctrlrange[self._gripper_actuator_ids]
            denom = np.maximum(ctrlrange[:, 1] - ctrlrange[:, 0], 1e-6)
            opening = np.full(
                self._gripper_actuator_ids.size, 0.5 * (1.0 - gripper_action)
            )
            opening = np.clip(opening, 0.0, 1.0)
            self.data.ctrl[self._gripper_actuator_ids] = ctrlrange[:, 0] + opening * denom

        self._apply_gravity_compensation(self._arm_dof_ids)


class RobotGripperEnv(SingleArmEnv):
    """Today's default robot + Robotiq environment with Cartesian IK actions."""

    def __init__(self, robot: str = "ur5e", **kwargs) -> None:
        super().__init__(robot=robot, gripper="robotiq_2f85", **kwargs)


class SingleArmHandEnv(SingleArmEnv):
    """UR5e + dexterous hand environment with Cartesian arm and hand qpos control."""

    def __init__(
        self,
        robot: str = "ur5e",
        hand: str = "inspire_right",
        **kwargs,
    ) -> None:
        if robot != "ur5e":
            raise ValueError("Dexterous hand environments are only supported on ur5e.")
        if hand not in HAND_DOF_BY_MODEL:
            raise ValueError(
                f"Unknown hand model '{hand}'. Options: {sorted(HAND_DOF_BY_MODEL)}"
            )
        self.hand_name = hand
        self._hand_dof = HAND_DOF_BY_MODEL[hand]
        self._hand_qpos_ids = np.empty(0, dtype=np.int32)
        self._hand_dof_ids = np.empty(0, dtype=np.int32)
        super().__init__(robot=robot, gripper=hand, **kwargs)

    @property
    def action_space(self):
        if self._model is None:
            self.reset()

        hand_range = self._hand_action_ctrlrange()
        if self.action_type == "pos":
            xyz_low = (
                self.workspace_bounds[0]
                if self.workspace_bounds is not None
                else [-np.inf, -np.inf, -np.inf]
            )
            xyz_high = (
                self.workspace_bounds[1]
                if self.workspace_bounds is not None
                else [np.inf, np.inf, np.inf]
            )
            low = np.asarray([*xyz_low, -1, -1, -1, -1, *hand_range[:, 0]])
            high = np.asarray([*xyz_high, 1, 1, 1, 1, *hand_range[:, 1]])
        else:
            low = np.asarray([-1, -1, -1, -1, -1, -1, -1, *hand_range[:, 0]])
            high = np.asarray([1, 1, 1, 1, 1, 1, 1, *hand_range[:, 1]])
        return gym.spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

    def post_compile(self) -> None:
        super().post_compile()
        if self._gripper_actuator_ids.size != self._hand_dof:
            raise ValueError(
                f"Expected {self._hand_dof} actuators for {self.hand_name}, "
                f"got {self._gripper_actuator_ids.size}."
            )
        hand_joint_ids = self.model.actuator_trnid[self._gripper_actuator_ids, 0]
        self._hand_qpos_ids = self.model.jnt_qposadr[hand_joint_ids].astype(np.int32)
        self._hand_dof_ids = self.model.jnt_dofadr[hand_joint_ids].astype(np.int32)

    def initialize_episode(self) -> None:
        super().initialize_episode()
        self.data.qpos[self._arm_qpos_ids] = UR5E_HAND_HOME_QPOS
        self.data.qvel[self._arm_dof_ids] = 0.0
        self.data.ctrl[self._arm_actuator_ids] = UR5E_HAND_HOME_QPOS
        hand_home = self._hand_neutral_qpos()
        self.data.qpos[self._hand_qpos_ids] = hand_home
        self.data.qvel[self._hand_dof_ids] = 0.0
        self.data.ctrl[self._gripper_actuator_ids] = hand_home
        mujoco.mj_forward(self.model, self.data)
        self._apply_gravity_compensation(self._arm_dof_ids, self._hand_dof_ids)

    def set_control(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        expected_shape = (7 + self._hand_dof,)
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected hand action shape {expected_shape}, got {action.shape}."
            )

        ee_pose = self.ee_pose()
        action_pos = action[:3]
        action_quat = self._normalize_quat(action[3:7])
        hand_qpos = action[7:]

        if self.action_type == "delta":
            action_pos = np.clip(action_pos, -1.0, 1.0)
            target_pos = ee_pose[:3] + action_pos * self.position_action_scale
            target_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(target_quat, ee_pose[3:], action_quat)
        else:
            target_pos = action_pos.copy()
            target_quat = action_quat.copy()

        if self.workspace_bounds is not None:
            np.clip(
                target_pos,
                self.workspace_bounds[0],
                self.workspace_bounds[1],
                out=target_pos,
            )

        target_quat /= np.linalg.norm(target_quat)
        qpos_target = self.solve_ik(target_pos, target_quat)
        arm_ctrlrange = self.model.actuator_ctrlrange[self._arm_actuator_ids]
        self.data.ctrl[self._arm_actuator_ids] = np.clip(
            qpos_target,
            arm_ctrlrange[:, 0],
            arm_ctrlrange[:, 1],
        )

        hand_ctrl = self._format_hand_action(hand_qpos)
        hand_range = self.model.actuator_ctrlrange[self._gripper_actuator_ids]
        self.data.ctrl[self._gripper_actuator_ids] = np.clip(
            hand_ctrl, hand_range[:, 0], hand_range[:, 1]
        )
        self._apply_gravity_compensation(self._arm_dof_ids, self._hand_dof_ids)

    def hand_qpos(self) -> np.ndarray:
        return self.data.qpos[self._hand_qpos_ids].copy()

    def _hand_neutral_qpos(self) -> np.ndarray:
        ctrlrange = self.model.actuator_ctrlrange[self._gripper_actuator_ids]
        return np.clip(
            np.zeros(self._hand_dof, dtype=np.float64),
            ctrlrange[:, 0],
            ctrlrange[:, 1],
        )

    def _format_hand_action(self, hand_qpos: np.ndarray) -> np.ndarray:
        return np.asarray(hand_qpos, dtype=np.float64)

    def _hand_action_ctrlrange(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self._gripper_actuator_ids]
