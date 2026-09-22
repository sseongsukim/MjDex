"""Keyboard teleoperation device for Cartesian end-effector control.

The key layout follows the DualDex keyboard teleop convention, trimmed down to
the controls needed for MjDex single-arm IK environments.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from mjdex.teleop.collect_enum import CollectEnum
from mjdex.transform import euler_delta_quat


@dataclass(frozen=True)
class KeyboardAction:
    dpos: np.ndarray
    dquat_wxyz: np.ndarray
    grasp: float
    reset: bool
    collect_enum: CollectEnum
    quit: bool

    @property
    def success(self) -> bool:
        return self.collect_enum == CollectEnum.SUCCESS

    def as_delta_pose_action(self, position_scale: float) -> np.ndarray:
        scaled_dpos = self.dpos / max(position_scale, 1e-8)
        return np.concatenate(
            [
                scaled_dpos,
                self.dquat_wxyz,
                np.array([self.grasp], dtype=np.float64),
            ]
        ).astype(np.float32)


class KeyboardTeleop:
    """Keyboard interface for MjDex Cartesian IK teleoperation."""

    POS_KEYS = {"w", "s", "a", "d", "q", "e"}
    ROT_KEYS = {"i", "k", "j", "l", "u", "o"}

    def __init__(
        self,
        pos_delta: float = 0.01,
        rot_delta: float = 0.08,
        mode: str = "hold",
    ) -> None:
        if mode not in ("hold", "tap"):
            raise ValueError(f"Unsupported keyboard mode '{mode}'.")
        self.pos_delta = float(pos_delta)
        self.rot_delta = float(rot_delta)
        self.mode = mode
        self._pressed: set[str] = set()
        self._pending_dpos = np.zeros(3, dtype=np.float64)
        self._pending_drot = np.zeros(3, dtype=np.float64)
        self._grasp = 1.0
        self._reset = False
        self._collect_enum = CollectEnum.DONE_FALSE
        self._quit = False
        self._lock = threading.Lock()

        from pynput.keyboard import Listener

        self._listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def _on_press(self, key) -> None | bool:
        from pynput.keyboard import Key

        with self._lock:
            if key == Key.esc:
                self._quit = True
                self._collect_enum = CollectEnum.TERMINATE
                return False

            try:
                char = key.char.lower()
            except AttributeError:
                return None

            if char in self.POS_KEYS or char in self.ROT_KEYS:
                if self.mode == "tap" and char not in self._pressed:
                    self._add_pending_motion(char)
                self._pressed.add(char)
            elif char == "z":
                if char not in self._pressed:
                    self._grasp = -1.0 if self._grasp >= 0.0 else 1.0
                self._pressed.add(char)
            elif char == "r":
                self._reset = True
                self._collect_enum = CollectEnum.RESET
            elif char == "t":
                if char not in self._pressed:
                    self._collect_enum = CollectEnum.SUCCESS
                self._pressed.add(char)
            elif char == "n":
                if char not in self._pressed:
                    self._collect_enum = CollectEnum.FAIL
                self._pressed.add(char)
            elif char.isdigit():
                self._collect_enum = CollectEnum.REWARD
            elif char == "`":
                self._collect_enum = CollectEnum.SKILL
            elif char == "b":
                self._collect_enum = CollectEnum.UNDO
            elif char == "[":
                self.pos_delta = max(0.001, self.pos_delta - 0.001)
                self.rot_delta = max(0.01, self.rot_delta - 0.01)
            elif char == "]":
                self.pos_delta = min(0.10, self.pos_delta + 0.001)
                self.rot_delta = min(0.30, self.rot_delta + 0.01)
            return None

    def _add_pending_motion(self, char: str) -> None:
        if char == "w":
            self._pending_dpos[0] -= self.pos_delta
        elif char == "s":
            self._pending_dpos[0] += self.pos_delta
        elif char == "a":
            self._pending_dpos[1] -= self.pos_delta
        elif char == "d":
            self._pending_dpos[1] += self.pos_delta
        elif char == "q":
            self._pending_dpos[2] -= self.pos_delta
        elif char == "e":
            self._pending_dpos[2] += self.pos_delta
        elif char == "j":
            self._pending_drot[0] += self.rot_delta
        elif char == "l":
            self._pending_drot[0] -= self.rot_delta
        elif char == "k":
            self._pending_drot[1] += self.rot_delta
        elif char == "i":
            self._pending_drot[1] -= self.rot_delta
        elif char == "u":
            self._pending_drot[2] += self.rot_delta
        elif char == "o":
            self._pending_drot[2] -= self.rot_delta

    def _on_release(self, key) -> None:
        try:
            char = key.char.lower()
        except AttributeError:
            return

        with self._lock:
            self._pressed.discard(char)

    def get_action(self) -> KeyboardAction:
        with self._lock:
            pressed = set(self._pressed)
            pending_dpos = self._pending_dpos.copy()
            pending_drot = self._pending_drot.copy()
            reset = self._reset
            collect_enum = self._collect_enum
            quit_requested = self._quit
            self._pending_dpos[:] = 0.0
            self._pending_drot[:] = 0.0
            self._reset = False
            self._collect_enum = CollectEnum.DONE_FALSE

        if self.mode == "tap":
            dpos = pending_dpos
            drot = pending_drot
        else:
            dpos = np.zeros(3, dtype=np.float64)
            if "w" in pressed:
                dpos[0] -= self.pos_delta
            if "s" in pressed:
                dpos[0] += self.pos_delta
            if "a" in pressed:
                dpos[1] -= self.pos_delta
            if "d" in pressed:
                dpos[1] += self.pos_delta
            if "q" in pressed:
                dpos[2] -= self.pos_delta
            if "e" in pressed:
                dpos[2] += self.pos_delta

            drot = np.zeros(3, dtype=np.float64)
            if "j" in pressed:
                drot[0] += self.rot_delta
            if "l" in pressed:
                drot[0] -= self.rot_delta
            if "k" in pressed:
                drot[1] += self.rot_delta
            if "i" in pressed:
                drot[1] -= self.rot_delta
            if "u" in pressed:
                drot[2] += self.rot_delta
            if "o" in pressed:
                drot[2] -= self.rot_delta

        return KeyboardAction(
            dpos=dpos,
            dquat_wxyz=euler_delta_quat(*drot),
            grasp=self._grasp,
            reset=reset,
            collect_enum=collect_enum,
            quit=quit_requested,
        )

    def print_usage(self) -> None:
        print("============== MjDex Keyboard Teleop ==============")
        print("Position: q(-z) w(-x) e(+z) / a(-y) s(+x) d(+y)")
        print("Rotation: j(+roll) l(-roll) / k(+pitch) i(-pitch) / u(+yaw) o(-yaw)")
        print("Gripper: z toggle open/close")
        print("Step size: [ smaller, ] larger")
        print("Collect: t(success) n(fail) ` skill b undo")
        print("Reset: r")
        print("Quit: Esc")
        print("=================================================")

    def close(self) -> None:
        self._listener.stop()
