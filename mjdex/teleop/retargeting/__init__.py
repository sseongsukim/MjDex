"""MjDex hand pose retargeting utilities.

Provides hand pose retargeting from MediaPipe-format landmarks to dexterous
robot hand joint angles.

Main classes:
- Retargeter: high-level unified interface
- BaseOptimizer: low-level optimizer access

Example:
    from mjdex.teleop.retargeting import Retargeter

    retargeter = Retargeter.from_yaml("path/to/config.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (22,)
"""

from .retarget import Retargeter
from .optimizer import BaseOptimizer, LPFilter
from .mediapipe import apply_mediapipe_transformations

__all__ = [
    "Retargeter",
    "BaseOptimizer",
    "LPFilter",
    "apply_mediapipe_transformations",
]
