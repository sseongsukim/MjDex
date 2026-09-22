"""Tuned actuator gains for MjDex robot position actuators."""

from __future__ import annotations

import numpy as np


TUNED_ARM_GAINS = {
    "ur5e": {
        "kp": np.array([5000, 5000, 5000, 3500, 3500, 3500], dtype=np.float64),
        "kd": np.array([200, 200, 200, 120, 120, 120], dtype=np.float64),
        "force_limit": np.array([320, 320, 320, 160, 160, 160], dtype=np.float64),
    },
    "fr3": {
        "kp": np.array([7000, 7000, 6000, 6000, 2500, 2500, 2500], dtype=np.float64),
        "kd": np.array([180, 180, 150, 150, 70, 70, 70], dtype=np.float64),
        "force_limit": np.array([150, 150, 150, 150, 45, 45, 45], dtype=np.float64),
    },
}


def apply_tuned_arm_gains(model, robot: str, actuator_ids: np.ndarray) -> None:
    """Apply tuned Kp/Kd/force limits to arm position actuators."""

    gains = TUNED_ARM_GAINS.get(robot)
    if gains is None:
        return

    actuator_ids = np.asarray(actuator_ids, dtype=np.int32)
    expected = gains["kp"].size
    if actuator_ids.size != expected:
        raise ValueError(
            f"Expected {expected} arm actuators for '{robot}', got {actuator_ids.size}."
        )

    model.actuator_gainprm[actuator_ids, 0] = gains["kp"]
    model.actuator_biasprm[actuator_ids, 1] = -gains["kp"]
    model.actuator_biasprm[actuator_ids, 2] = -gains["kd"]
    model.actuator_forcerange[actuator_ids, 0] = -gains["force_limit"]
    model.actuator_forcerange[actuator_ids, 1] = gains["force_limit"]
    model.actuator_forcelimited[actuator_ids] = True
