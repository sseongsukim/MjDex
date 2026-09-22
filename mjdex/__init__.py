"""MjDex environment entry points."""

from mjdex.core import MjDexEnv
from mjdex.contact import (
    check_contact_geom_ids,
    collect_geoms_by_name_prefix,
    collect_geoms_under_body_prefix,
    contact_pairs,
)
from mjdex.single_arm import (
    RobotGripperEnv,
    SingleArmHandEnv,
    SingleArmEnv,
)
from mjdex.dual_arm import (
    DualArmEnv,
    DualArmHandEnv,
    DualArmRobotGripperEnv,
)
from mjdex.envs import register_mjdex_envs

register_mjdex_envs()

__all__ = [
    "MjDexEnv",
    "check_contact_geom_ids",
    "collect_geoms_by_name_prefix",
    "collect_geoms_under_body_prefix",
    "contact_pairs",
    "RobotGripperEnv",
    "SingleArmHandEnv",
    "SingleArmEnv",
    "DualArmEnv",
    "DualArmHandEnv",
    "DualArmRobotGripperEnv",
    "register_mjdex_envs",
    "SingleArmAllegroTabletopPlacementEnv",
    "FixedGoalTabletopEnv",
    "TabletopPlacementTask",
]
