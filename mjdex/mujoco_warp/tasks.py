"""Batch equivalents of the four existing MjDex task hooks.

Task constants and compiled IDs come from the original task instance. Keep
predicates in parity with mjdex/tasks; no DualDex task semantics are imported.
"""

import jax.numpy as jnp

from mjdex.tasks.barcode_scan import BarcodeScanTask
from mjdex.tasks.dish_rack import DishRackTask
from mjdex.tasks.mug_rack import MugRackTask
from mjdex.tasks.relocate import YCB_HEIGHT, RelocateTask

from .math import quat_distance

SUPPORTED_TASKS = (RelocateTask, BarcodeScanTask, DishRackTask, MugRackTask)


def body_pose(data, body):
    if body < 0:
        return jnp.full((data.qpos.shape[0], 7), jnp.nan)
    return jnp.concatenate((data.xpos[:, body], data.xquat[:, body]), axis=-1)


def task_observation(task, data):
    if isinstance(task, RelocateTask):
        return {
            "object_pose": body_pose(data, task._object_body_id),
            "target_pose": body_pose(data, task._target_body_id),
        }
    if isinstance(task, BarcodeScanTask):
        return {
            "object_pose": body_pose(data, task._object_body_id),
            "scanner_pose": body_pose(data, task._scanner_body_id),
        }
    rack = body_pose(data, task._rack_body_id)
    if isinstance(task, DishRackTask):
        return {"rack_pose": rack, "dish_pose": body_pose(data, task._dish_body_id)}
    return {
        "rack_pose": jnp.concatenate((rack[:, :2], rack[:, 3:]), axis=-1),
        "mug_pose": body_pose(data, task._mug_body_id),
    }


def task_result(task, data, ee_positions, contact):
    """Return reward, termination, and the original task's info dictionary."""
    n = data.qpos.shape[0]
    if isinstance(task, RelocateTask):
        obj = body_pose(data, task._object_body_id)
        target = body_pose(data, task._target_body_id)
        distance = jnp.linalg.norm(obj[:, :3] - target[:, :3], axis=-1)
        rotation = quat_distance(obj[:, 3:], target[:, 3:])
        position_ok = distance < task.position_threshold
        success = position_ok & (
            (rotation < task.rotation_threshold) | (not task.require_orientation)
        )
        touching = contact(data, task._robot_geom_ids, task._object_geom_ids)
        table = contact(data, task._robot_geom_ids, task._table_geom_ids)
        info = {
            "object_pose": obj,
            "target_pose": target,
            "object_target_distance": distance,
            "object_target_rotation_error": rotation,
            "relocate_success": position_ok,
            "success": success,
            "robot_object_contact": touching,
            "robot_table_contact": table,
        }
        if task.sparse_reward:
            return jnp.where(success, task.success_reward, -1.0), success, info
        eo_all = jnp.linalg.norm(ee_positions - obj[:, None, :3], axis=-1)
        nearest = jnp.argmin(eo_all, axis=-1)
        ee = ee_positions[jnp.arange(n), nearest]
        eo = eo_all[jnp.arange(n), nearest]
        et = jnp.linalg.norm(ee - target[:, :3], axis=-1)
        lift = jnp.maximum(
            jnp.minimum(obj[:, 2], target[:, 2]) - YCB_HEIGHT[task.object_id] / 2, 0.0
        )
        lifted = touching & (lift > 0.015)
        placed = lifted & (distance < 0.05)
        reward = -0.1 * eo + jnp.where(touching, 0.05 + lift, 0.0)
        reward += jnp.where(
            lifted, 0.1 + 0.3 * jnp.exp(-3 * et) + 0.7 * jnp.exp(-5 * distance), 0.0
        )
        reward += jnp.where(placed, jnp.exp(-10 * distance), 0.0)
        reward += jnp.where(
            placed & task.require_orientation & (rotation < task.rotation_threshold),
            2.0,
            0.0,
        )
        reward += jnp.where(placed & success, task.success_reward, 0.0)
        return jnp.where(table, -100.0, reward), success, info

    if isinstance(task, BarcodeScanTask):
        obj = body_pose(data, task._object_body_id)
        scanner = body_pose(data, task._scanner_body_id)
        info = {
            "object_pose": obj,
            "scanner_pose": scanner,
            "object_scanner_relative_pos": obj[:, :3] - scanner[:, :3],
            "object_scanner_contact": contact(
                data, task._object_geom_ids, task._scanner_geom_ids
            ),
        }
        # CPU BarcodeScanTask deliberately has no scan-success predicate yet.
        return jnp.zeros(n), jnp.zeros(n, dtype=bool), info

    rack = body_pose(data, task._rack_body_id)
    rack_rot = data.xmat[:, task._rack_body_id]
    if isinstance(task, DishRackTask):
        dish = body_pose(data, task._dish_body_id)
        dish_rot = data.xmat[:, task._dish_body_id]
        relative_rot = jnp.swapaxes(rack_rot, -1, -2) @ dish_rot
        center_world = dish[:, :3] + jnp.einsum(
            "nij,j->ni", dish_rot, jnp.asarray(task.dish_center_offset)
        )
        center = jnp.einsum("nji,nj->ni", rack_rot, center_world - rack[:, :3])
        error = center - jnp.asarray(task.success_target_center)
        angle = jnp.rad2deg(
            jnp.arccos(jnp.clip(jnp.abs(relative_rot[:, 1, 2]), 0.0, 1.0))
        )
        velocity = data.qvel[:, task._dish_dof_addr : task._dish_dof_addr + 6]
        linear = jnp.linalg.norm(velocity[:, :3], axis=-1)
        angular = jnp.linalg.norm(velocity[:, 3:], axis=-1)
        pose_ok = jnp.all(
            jnp.abs(error) <= jnp.asarray(task.success_position_tolerance), axis=-1
        ) & (angle <= task.success_angle_tolerance)
        slow = (linear <= task.success_linear_speed) & (
            angular <= task.success_angular_speed
        )
        success = pose_ok & (slow | (not task.require_low_speed_for_success))
        info = {
            "rack_pose": rack,
            "dish_pose": dish,
            "dish_rack_relative_pos": dish[:, :3] - rack[:, :3],
            "dish_rack_contact": contact(
                data, task._dish_geom_ids, task._rack_geom_ids
            ),
            "success": success,
            "dish_rack_pose_success": pose_ok,
            "dish_low_speed": slow,
            "dish_rack_center_local": center,
            "dish_rack_position_error": error,
            "dish_rack_angle_error_deg": angle,
            "dish_linear_speed": linear,
            "dish_angular_speed": angular,
        }
        return success.astype(jnp.float32), success, info

    mug = body_pose(data, task._mug_body_id)
    relative_pos = jnp.einsum("nji,nj->ni", rack_rot, mug[:, :3] - rack[:, :3])
    relative_rot = jnp.swapaxes(rack_rot, -1, -2) @ data.xmat[:, task._mug_body_id]
    if task._assembly_candidates:
        candidates = jnp.asarray(task._assembly_candidates)
        position_ok = jnp.all(
            jnp.abs(relative_pos[:, None] - candidates[None, :, :3, 3])
            <= jnp.asarray(task.assembly_pos_threshold),
            axis=-1,
        )
        dots = jnp.einsum("cij,nij->ncj", candidates[:, :3, :3], relative_rot)
        success = jnp.any(
            position_ok & jnp.all(dots >= task.assembly_ori_bound, axis=-1), axis=-1
        )
    else:
        success = jnp.zeros(n, dtype=bool)
    info = {
        "rack_pose": rack,
        "mug_pose": mug,
        "mug_rack_relative_pos": mug[:, :3] - rack[:, :3],
        "mug_rack_contact": contact(data, task._mug_geom_ids, task._rack_geom_ids),
    }
    return success.astype(jnp.float32), success, info
