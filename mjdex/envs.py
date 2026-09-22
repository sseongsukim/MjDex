"""Gymnasium registration helpers for MjDex environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import registry

from mjdex.dual_arm import DualArmEnv, DualArmHandEnv
from mjdex.robots import DEFAULT_FLOOR_Z, DUALDEX_TABLE_SIZE
from mjdex.single_arm import SingleArmEnv, SingleArmHandEnv
from mjdex.tasks.barcode_scan import BARCODE_SCAN_OBJECTS, BarcodeScanTask
from mjdex.tasks.dish_rack import DishRackTask
from mjdex.tasks.mug_rack import MugRackTask
from mjdex.tasks.relocate import RelocateTask

ENV_SPECS: dict[str, Callable[..., gym.Env]] = {}
ENV_MAX_STEPS: dict[str, int] = {}

# YCB objects registered for relocate (banana excluded)
_RELOCATE_OBJECTS = ("cracker", "mustard", "meat", "bleach")


# How far the tabletop's near edge sits in FRONT of the robot base, for the
# dexterous hand scenes. Zero puts the edge flush with the base; this is past
# that, so the base overhangs the table and floats behind it. That is deliberate:
# the hand works out across the table, and the strip of tabletop beside the base
# is reach the arm cannot use anyway, so trading it for depth on the far side
# buys real working room.
_HAND_TABLE_SETBACK = 0.10


def _hand_table_pos() -> tuple[float, float, float]:
    """Table pose for the dexterous hand scenes, set back from the robot base.

    The base itself stays at the world origin — it is the frame the IK model, the
    workspace bounds and every task position are defined in — so the table moves,
    not the robot. Gripper and mug-rack scenes keep their own defaults.
    """

    return (
        DUALDEX_TABLE_SIZE[1] / 2.0 + _HAND_TABLE_SETBACK,
        0.0,
        DEFAULT_FLOOR_Z,
    )


def mjdex_env(
    env_id: str, max_episode_steps: int | None = None
) -> Callable[[Callable[..., gym.Env]], Callable[..., gym.Env]]:
    """Register a MjDex env factory in the local extension table."""

    def decorator(factory: Callable[..., gym.Env]) -> Callable[..., gym.Env]:
        ENV_SPECS[env_id] = factory
        if max_episode_steps is not None:
            ENV_MAX_STEPS[env_id] = max_episode_steps
        return factory

    return decorator


# ─── mug rack ────────────────────────────────────────────────────────────────
# single-mugrack-v0: SingleArmEnv, fr3, gripper


def _make_single_mugrack(
    *,
    robot: str = "fr3",
    gripper: str = "robotiq_2f85",
    action_type: str = "pos",
    position_action_scale: float = 0.05,
    control_timestep: float = 0.02,
    max_episode_steps: int = 600,
    include_table: bool = True,
    show_target: bool = False,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SingleArmEnv:
    task_options = dict(task_kwargs or {})
    task_options.setdefault("show_target", show_target)
    return SingleArmEnv(
        robot=robot,
        gripper=gripper,
        action_type=action_type,
        position_action_scale=position_action_scale,
        control_timestep=control_timestep,
        max_episode_steps=max_episode_steps,
        include_table=include_table,
        task=MugRackTask(**task_options),
        **kwargs,
    )


@mjdex_env("single-mugrack-v0", max_episode_steps=600)
def _single_mugrack_env(**kwargs: Any) -> SingleArmEnv:
    return _make_single_mugrack(**kwargs)


# ─── relocate ────────────────────────────────────────────────────────────────
# single-relocate-{object_id}-v0: SingleArmHandEnv, ur5e, sharpa_right
# dual-relocate-{object_id}-v0:   DualArmHandEnv,   ur5e, sharpa_left + sharpa_right


def _make_single_relocate(
    *,
    object_id: str,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SingleArmHandEnv:
    task_options = dict(task_kwargs or {})
    task_options.setdefault("object_id", object_id)
    return SingleArmHandEnv(
        robot="ur5e",
        hand="sharpa_right",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        table_pos=(
            _hand_table_pos() if table_pos is None else table_pos
        ),
        task=RelocateTask(**task_options),
        **kwargs,
    )


def _make_dual_relocate(
    *,
    object_id: str,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DualArmHandEnv:
    task_options = dict(task_kwargs or {})
    task_options.setdefault("object_id", object_id)
    return DualArmHandEnv(
        robot="ur5e",
        left_hand="sharpa_left",
        right_hand="sharpa_right",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        table_pos=(
            _hand_table_pos() if table_pos is None else table_pos
        ),
        task=RelocateTask(**task_options),
        **kwargs,
    )


def _make_single_relocate_factory(object_id: str) -> Callable[..., SingleArmHandEnv]:
    def factory(**kwargs: Any) -> SingleArmHandEnv:
        return _make_single_relocate(object_id=object_id, **kwargs)

    return factory


def _make_dual_relocate_factory(object_id: str) -> Callable[..., DualArmHandEnv]:
    def factory(**kwargs: Any) -> DualArmHandEnv:
        return _make_dual_relocate(object_id=object_id, **kwargs)

    return factory


for _obj in _RELOCATE_OBJECTS:
    mjdex_env(f"single-relocate-{_obj}-v0", max_episode_steps=1000)(
        _make_single_relocate_factory(_obj)
    )
    mjdex_env(f"dual-relocate-{_obj}-v0", max_episode_steps=1000)(
        _make_dual_relocate_factory(_obj)
    )


# ─── dish rack ───────────────────────────────────────────────────────────────
# single-gripper-dishrack-v0: SingleArmEnv,     ur5e,  gripper
# single-hand-dishrack-v0:   SingleArmHandEnv,  ur5e,  sharpa_right
# dual-gripper-dishrack-v0:  DualArmEnv,        fr3,   gripper
# dual-hand-dishrack-v0:     DualArmHandEnv,    ur5e,  sharpa_left + sharpa_right


def _make_single_gripper_dishrack(
    *,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SingleArmEnv:
    return SingleArmEnv(
        robot="ur5e",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        task=DishRackTask(**(task_kwargs or {})),
        **kwargs,
    )


def _make_single_hand_dishrack(
    *,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SingleArmHandEnv:
    return SingleArmHandEnv(
        robot="ur5e",
        hand="sharpa_right",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        table_pos=(
            _hand_table_pos() if table_pos is None else table_pos
        ),
        task=DishRackTask(**(task_kwargs or {})),
        **kwargs,
    )


def _make_dual_gripper_dishrack(
    *,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DualArmEnv:
    task_kwargs = {
        "rack_pos": (0.45, -0.25, 0.005),
        "dish_pos": (0.45, 0.25, 0.0025),
        "reset_xy_bounds": ((0.25, 0.65), (-0.45, 0.45)),
        **(task_kwargs or {}),
    }
    return DualArmEnv(
        robot="fr3",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        task=DishRackTask(**(task_kwargs or {})),
        **kwargs,
    )


def _make_dual_hand_dishrack(
    *,
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DualArmHandEnv:
    task_kwargs = {
        "rack_pos": (0.45, -0.25, 0.005),
        "dish_pos": (0.45, 0.25, 0.0025),
        "reset_xy_bounds": ((0.25, 0.65), (-0.45, 0.45)),
        **(task_kwargs or {}),
    }
    return DualArmHandEnv(
        robot="ur5e",
        left_hand="sharpa_left",
        right_hand="sharpa_right",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        table_pos=(
            _hand_table_pos() if table_pos is None else table_pos
        ),
        task=DishRackTask(**(task_kwargs or {})),
        **kwargs,
    )


@mjdex_env("single-gripper-dishrack-v0", max_episode_steps=1000)
def _single_gripper_dishrack_env(**kwargs: Any) -> SingleArmEnv:
    return _make_single_gripper_dishrack(**kwargs)


@mjdex_env("single-hand-dishrack-v0", max_episode_steps=1000)
def _single_hand_dishrack_env(**kwargs: Any) -> SingleArmHandEnv:
    return _make_single_hand_dishrack(**kwargs)


@mjdex_env("dual-gripper-dishrack-v0", max_episode_steps=1000)
def _dual_gripper_dishrack_env(**kwargs: Any) -> DualArmEnv:
    return _make_dual_gripper_dishrack(**kwargs)


@mjdex_env("dual-hand-dishrack-v0", max_episode_steps=1000)
def _dual_hand_dishrack_env(**kwargs: Any) -> DualArmHandEnv:
    return _make_dual_hand_dishrack(**kwargs)


# ─── barcode scan ────────────────────────────────────────────────────────────
# dual-barcode-{object_id}-v0: DualArmHandEnv, ur5e, sharpa_left + sharpa_right


def _make_dual_barcode(
    *,
    object_id: str = "meat",
    action_type: str = "pos",
    control_timestep: float = 0.02,
    include_table: bool = True,
    table_pos: tuple[float, float, float] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DualArmHandEnv:
    task_options = dict(task_kwargs or {})
    task_options.setdefault("object_id", object_id)
    return DualArmHandEnv(
        robot="ur5e",
        left_hand="sharpa_left",
        right_hand="sharpa_right",
        action_type=action_type,
        control_timestep=control_timestep,
        include_table=include_table,
        table_pos=(
            _hand_table_pos() if table_pos is None else table_pos
        ),
        task=BarcodeScanTask(**task_options),
        **kwargs,
    )


def _make_dual_barcode_factory(object_id: str) -> Callable[..., DualArmHandEnv]:
    def factory(**kwargs: Any) -> DualArmHandEnv:
        return _make_dual_barcode(object_id=object_id, **kwargs)

    return factory


# BARCODE_SCAN_OBJECTS is the task's own list (every YCB object but the banana),
# so new scannable objects register themselves without touching this file.
for _obj in BARCODE_SCAN_OBJECTS:
    mjdex_env(f"dual-barcode-{_obj}-v0", max_episode_steps=1000)(
        _make_dual_barcode_factory(_obj)
    )


# ─── registration ─────────────────────────────────────────────────────────────


def register_mjdex_envs() -> None:
    """Register all MjDex env ids with Gymnasium.

    Add future environments by defining another ``@mjdex_env("<id>")`` factory
    above and calling ``register_mjdex_envs()`` again, or simply import this
    module (repeated imports are harmless since already-registered ids are
    skipped).
    """

    for env_id, factory in ENV_SPECS.items():
        if env_id in registry:
            continue
        gym.register(
            id=env_id,
            entry_point=factory,
            max_episode_steps=ENV_MAX_STEPS.get(env_id),
        )


register_mjdex_envs()
