"""Geometry and pose transform helpers for MjDex."""

from __future__ import annotations

import math

import numpy as np


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Return a z-axis yaw quaternion in MuJoCo's wxyz order."""

    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def x_tilt_quat(angle: float) -> tuple[float, float, float, float]:
    """Return an x-axis tilt quaternion in MuJoCo's wxyz order."""

    return (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def y_tilt_quat(angle: float) -> tuple[float, float, float, float]:
    """Return a y-axis tilt quaternion in MuJoCo's wxyz order."""

    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def normalize_quat(quat: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    """Return a normalized quaternion in MuJoCo's wxyz order."""

    quat_arr = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat_arr)
    if norm < 1e-8:
        raise ValueError("Quaternion norm must be non-zero.")
    return quat_arr / norm


def _normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return tuple(value / norm for value in vec)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = 0.5 * angle
    return np.array(
        [
            np.cos(half_angle),
            *(np.sin(half_angle) * axis),
        ],
        dtype=np.float64,
    )


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def euler_delta_quat(droll: float, dpitch: float, dyaw: float) -> np.ndarray:
    qx = axis_angle_quat(np.array([1.0, 0.0, 0.0]), droll)
    qy = axis_angle_quat(np.array([0.0, 1.0, 0.0]), dpitch)
    qz = axis_angle_quat(np.array([0.0, 0.0, 1.0]), dyaw)
    quat = quat_mul(qz, quat_mul(qy, qx))
    return quat / np.linalg.norm(quat)


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert a 7D pose [x, y, z, qw, qx, qy, qz] to a 4x4 homogeneous matrix."""
    x, y, z = pose[:3]
    qw, qx, qy, qz = pose[3:]
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = [x, y, z]
    return mat


def is_similar_pose(
    mat1: np.ndarray,
    mat2: np.ndarray,
    pos_threshold: float | list[float] = 0.04,
    ori_bound: float = 0.90,
) -> bool:
    """Return True if two 4x4 pose matrices are within positional and rotational bounds."""
    diff = np.abs(mat1[:3, 3] - mat2[:3, 3])
    thresh = np.asarray(pos_threshold)
    if np.any(diff > thresh):
        return False
    for col in range(3):
        if float(np.dot(mat1[:3, col], mat2[:3, col])) < ori_bound:
            return False
    return True
