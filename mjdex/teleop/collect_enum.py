"""Define keyboard/data-collection events for teleoperation."""

from __future__ import annotations

from enum import Enum


class CollectEnum(Enum):
    DONE_FALSE = 2
    SUCCESS = 3
    FAIL = 4
    REWARD = 5
    SKILL = 6
    RESET = 7
    TERMINATE = 8
    UNDO = 9
