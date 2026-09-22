"""Small task interface for robot environments."""

from __future__ import annotations

from typing import Any
from dm_control import mjcf


class Task:
    """Optional task layer attached to an environment scene."""

    def build_mjcf(self, scene: mjcf.RootElement) -> mjcf.RootElement:
        return scene

    def post_compile(self, env: Any) -> None:
        pass

    def initialize_episode(self, env: Any) -> None:
        pass

    def compute_observation(self, env: Any) -> dict[str, Any]:
        return {}

    def compute_reward(self, env: Any) -> float:
        return 0.0

    def terminate_episode(self, env: Any) -> bool:
        return False

    def get_info(self, env: Any) -> dict[str, Any]:
        return {}

    def robot_excluded_geom_ids(self, env: Any) -> set[int]:
        return set()
