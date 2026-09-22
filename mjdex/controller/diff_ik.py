"""Differential inverse kinematics for MuJoCo site targets.

Adapted from the OGBench/DualDex differential IK controller pattern.
"""

from __future__ import annotations

import mujoco
import numpy as np


_PI = np.pi
_TWO_PI = 2 * np.pi


def angle_diff(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return wrapped angular difference ``q1 - q2``."""

    return np.mod(q1 - q2 + _PI, _TWO_PI) - _PI


class DiffIKController:
    """Damped least-squares differential IK controller."""

    def __init__(
        self,
        model: mujoco.MjModel,
        sites: list[str],
        qpos0: np.ndarray | None = None,
        damping_coeff: float = 1e-8,
        max_angle_change: float = np.radians(45),
    ) -> None:
        self._model = model
        self._data = mujoco.MjData(self._model)
        self._qpos0 = None if qpos0 is None else np.asarray(qpos0, dtype=np.float64)
        self._max_angle_change = float(max_angle_change)

        self._site_ids = np.asarray([self._model.site(site).id for site in sites])
        self._num_sites = len(self._site_ids)

        self._err = np.empty((self._num_sites, 6), dtype=np.float64)
        self._site_quat = np.empty((self._num_sites, 4), dtype=np.float64)
        self._site_quat_inv = np.empty((self._num_sites, 4), dtype=np.float64)
        self._err_quat = np.empty((self._num_sites, 4), dtype=np.float64)
        self._jac = np.empty((6 * self._num_sites, self._model.nv), dtype=np.float64)
        self._damping = damping_coeff * np.eye(6 * self._num_sites)
        self._eye = np.eye(self._model.nv)

    def _forward_kinematics(self) -> None:
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

    def _compute_error(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self._err[:, :3] = pos - self._data.site_xpos[self._site_ids]

        for i, site_id in enumerate(self._site_ids):
            mujoco.mju_mat2Quat(self._site_quat[i], self._data.site_xmat[site_id])
            mujoco.mju_negQuat(self._site_quat_inv[i], self._site_quat[i])
            mujoco.mju_mulQuat(self._err_quat[i], quat[i], self._site_quat_inv[i])
            mujoco.mju_quat2Vel(self._err[i, 3:], self._err_quat[i], 1.0)

    def _compute_jacobian(self) -> None:
        for i, site_id in enumerate(self._site_ids):
            jacp = self._jac[6 * i : 6 * i + 3]
            jacr = self._jac[6 * i + 3 : 6 * i + 6]
            mujoco.mj_jacSite(self._model, self._data, jacp, jacr, site_id)

    def _threshold_reached(self, pos_thresh: float, ori_thresh: float) -> bool:
        pos_achieved = np.linalg.norm(self._err[:, :3]) <= pos_thresh
        ori_achieved = np.linalg.norm(self._err[:, 3:]) <= ori_thresh
        return bool(pos_achieved and ori_achieved)

    def _solve_update(self) -> np.ndarray:
        hessian = self._jac @ self._jac.T + self._damping
        update = self._jac.T @ np.linalg.solve(hessian, self._err.ravel())

        if self._qpos0 is not None:
            jac_pinv = np.linalg.pinv(hessian)
            q_err = angle_diff(self._qpos0, self._data.qpos)
            update += (self._eye - (self._jac.T @ jac_pinv) @ self._jac) @ q_err

        update_max = np.max(np.abs(update))
        if update_max > self._max_angle_change:
            update *= self._max_angle_change / update_max
        return update

    def solve(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        curr_qpos: np.ndarray,
        max_iters: int = 20,
        pos_thresh: float = 1e-4,
        ori_thresh: float = 1e-4,
    ) -> np.ndarray:
        """Return qpos that brings configured sites toward target poses."""

        self._data.qpos[:] = np.asarray(curr_qpos, dtype=np.float64)

        target_pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
        target_quat = np.atleast_2d(np.asarray(quat, dtype=np.float64))

        for _ in range(max_iters):
            self._forward_kinematics()
            self._compute_error(target_pos, target_quat)
            if self._threshold_reached(pos_thresh, ori_thresh):
                break

            self._compute_jacobian()
            mujoco.mj_integratePos(
                self._model,
                self._data.qpos,
                self._solve_update(),
                1.0,
            )

        return self._data.qpos.copy()
