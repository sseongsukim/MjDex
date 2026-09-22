"""Open a registered MjDex env in the MuJoCo viewer to inspect the scene.

The viewer is driven passively: instead of feeding actions through
``env.step()``, physics is advanced with ``mj_step`` so ``data.ctrl`` keeps the
home targets written by ``initialize_episode()``. The robot therefore holds its
reset pose regardless of which action space the env uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from mjdex.envs import ENV_SPECS, register_mjdex_envs  # noqa: E402


def precise_wait(t_end: float, slack_time: float = 0.001) -> None:
    t_wait = t_end - time.monotonic()
    if t_wait <= 0.0:
        return

    t_sleep = t_wait - slack_time
    if t_sleep > 0.0:
        time.sleep(t_sleep)
    while time.monotonic() < t_end:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a registered MjDex environment in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--task",
        default="single-mugrack-v0",
        help="Env id to view (e.g. dual-hand-dishrack-v0). Use --list to see all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every registered env id and exit.",
    )
    parser.add_argument(
        "--mode",
        default="hold",
        choices=["hold", "static"],
        help=(
            "hold: advance physics so the robot holds its home pose and objects "
            "settle. static: no physics, show the reset state exactly."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the first reset; pressing R afterwards re-randomizes.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Real-time multiplier for --mode hold (default: 1.0).",
    )
    parser.add_argument(
        "--control-timestep",
        type=float,
        default=None,
        help="Override the env control timestep in seconds.",
    )
    parser.add_argument(
        "--camera",
        default="render_camera",
        help=(
            "Model camera to frame the initial view on, or 'free' for MuJoCo's "
            "default free camera. The free camera is used either way so the "
            "mouse keeps working."
        ),
    )
    parser.add_argument(
        "--lock-camera",
        action="store_true",
        help="Pin the view to --camera. Disables mouse orbit/pan/zoom.",
    )
    parser.add_argument("--no-table", action="store_true", help="Drop the table.")
    parser.add_argument(
        "--show-target",
        action="store_true",
        help="Show the task target site (mugrack and dishrack tasks only).",
    )
    parser.add_argument(
        "--hide-ui",
        action="store_true",
        help="Hide the viewer's left and right UI panels.",
    )
    return parser.parse_args()


def print_env_ids() -> None:
    print("Registered MjDex env ids:")
    for env_id in sorted(ENV_SPECS):
        print(f"  {env_id}")


def resolve_env_id(task: str) -> str:
    """Resolve a user-supplied task name to a registered env id."""

    if task in ENV_SPECS:
        return task

    versioned = f"{task}-v0"
    if versioned in ENV_SPECS:
        return versioned

    matches = sorted(env_id for env_id in ENV_SPECS if task in env_id)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise SystemExit(
            f"Ambiguous task '{task}'. Candidates: {', '.join(matches)}"
        )
    raise SystemExit(
        f"Unknown task '{task}'. Run with --list to see the {len(ENV_SPECS)} "
        "registered env ids."
    )


def build_env_kwargs(args: argparse.Namespace) -> dict:
    env_kwargs: dict = {}
    if args.control_timestep is not None:
        env_kwargs["control_timestep"] = args.control_timestep
    if args.no_table:
        env_kwargs["include_table"] = False
    if args.show_target:
        env_kwargs["task_kwargs"] = {"show_target": True}
    return env_kwargs


def make_env(env_id: str, env_kwargs: dict):
    import gymnasium as gym

    try:
        return gym.make(env_id, disable_env_checker=True, **env_kwargs).unwrapped
    except TypeError as exc:
        if "show_target" in str(exc):
            raise SystemExit(
                f"'{env_id}' does not support --show-target; only the mugrack "
                "and dishrack tasks expose a target site."
            ) from exc
        raise


def free_joint_body_names(env) -> list[str]:
    """Names of the bodies that own a free joint (the task's movable objects)."""

    import mujoco

    names = []
    for joint_id in range(env.model.njnt):
        if env.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body_id = env.model.jnt_bodyid[joint_id]
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        names.append(name if name is not None else f"body_{body_id}")
    return names


def ee_poses(env) -> list[tuple[str, "np.ndarray"]]:
    """End-effector poses as (label, pose) pairs for single- and dual-arm envs."""

    mounts = getattr(env, "mounts", None)
    if mounts is not None:
        return [(side, env.ee_pose(side)) for side in mounts]
    if hasattr(env, "ee_pose"):
        return [("ee", env.ee_pose())]
    return []


def print_ee_poses(env) -> None:
    for label, pose in ee_poses(env):
        with np.printoptions(precision=4, suppress=True):
            print(f"  {label}: {pose}")


def print_env_summary(env, env_id: str, mode: str) -> None:
    print(f"============== MjDex View Task: {env_id} ==============")
    print(f"Env class: {type(env).__name__}")
    print(f"Mode: {mode}")
    print(
        "Model: "
        f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
        f"nbody={env.model.nbody}"
    )
    print(f"Timestep: physics={env.physics_timestep}s control={env.control_timestep}s")
    action_space = env.action_space
    print(f"Action space: shape={action_space.shape} dtype={action_space.dtype}")
    objects = free_joint_body_names(env)
    if objects:
        print(f"Free bodies: {', '.join(objects)}")
    poses = ee_poses(env)
    if poses:
        print("Initial EEF pose:")
        print_ee_poses(env)
    print("--------------------------------------------------------")
    print("Mouse drag: orbit | right-drag: pan | scroll: zoom")
    print("Double-click a body to select it, then Ctrl+drag to push it around")
    print("Space: pause/resume physics")
    print("R: reset the environment")
    print("P: print the current EEF pose and observation keys")
    print("Close the viewer or press Esc to quit")
    print("========================================================")


def seed_free_camera(env, viewer, camera_id: int) -> None:
    """Point the free camera at whatever a fixed model camera frames.

    A ``mjCAMERA_FIXED`` camera reads its pose straight from the model, so
    ``mjv_moveCamera`` has no visible effect and mouse orbit/pan/zoom looks
    broken. Copying the fixed camera's pose into the free camera keeps the
    framing while leaving the mouse fully in control. ``camera_xyaxes_from_lookat``
    builds the render camera with world +z as up, so there is no roll to lose.
    """

    import mujoco

    pos = env.data.cam_xpos[camera_id].copy()
    # MuJoCo cameras look down their local -z axis.
    forward = -env.data.cam_xmat[camera_id].reshape(3, 3)[:, 2]

    # Orbit around the point on the view ray closest to the model's center.
    distance = float(np.dot(env.model.stat.center - pos, forward))
    if distance <= 1e-6:
        distance = 1.5 * float(env.model.stat.extent)

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = pos + forward * distance
    viewer.cam.distance = distance
    viewer.cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    viewer.cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))


def set_initial_camera(env, viewer, camera: str, lock: bool) -> None:
    import mujoco

    if camera == "free":
        return

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        available = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(env.model.ncam)
        ]
        print(
            f"Camera '{camera}' not found; using the free camera. "
            f"Available: {', '.join(n for n in available if n) or 'none'}"
        )
        return

    if lock:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id
        print(
            f"Locked to fixed camera '{camera}'; mouse orbit/pan/zoom is "
            "disabled until you switch back to the free camera."
        )
        return

    seed_free_camera(env, viewer, camera_id)


def main() -> None:
    args = parse_args()

    register_mjdex_envs()

    if args.list:
        print_env_ids()
        return

    env_id = resolve_env_id(args.task)

    import mujoco
    import mujoco.viewer

    env = make_env(env_id, build_env_kwargs(args))
    observation, _ = env.reset(seed=args.seed)
    print_env_summary(env, env_id, args.mode)

    state = {"paused": args.mode == "static", "reset": False, "print": False}

    def key_callback(keycode: int) -> None:
        key = chr(keycode) if 0 <= keycode < 0x110000 else ""
        if keycode == 32:  # space
            state["paused"] = not state["paused"]
            print("Physics paused" if state["paused"] else "Physics resumed")
        elif key in ("r", "R"):
            state["reset"] = True
        elif key in ("p", "P"):
            state["print"] = True

    n_substeps = max(int(round(env.control_timestep / env.physics_timestep)), 1)
    dt = env.control_timestep / max(args.speed, 1e-6)

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_callback,
        show_left_ui=not args.hide_ui,
        show_right_ui=not args.hide_ui,
    ) as viewer:
        set_initial_camera(env, viewer, args.camera, args.lock_camera)
        t_start = time.monotonic()
        step_idx = 0

        while viewer.is_running():
            t_cycle_end = t_start + (step_idx + 1) * dt

            if state["reset"]:
                state["reset"] = False
                # Reset without the seed so each reset re-randomizes the scene.
                observation, _ = env.reset()
                print("Environment reset")

            if state["print"]:
                state["print"] = False
                if ee_poses(env):
                    print("EEF pose:")
                    print_ee_poses(env)
                if isinstance(observation, dict):
                    print(f"Observation keys: {sorted(observation)}")

            if state["paused"]:
                mujoco.mj_forward(env.model, env.data)
            else:
                mujoco.mj_step(env.model, env.data, nstep=n_substeps)
                observation = env.compute_observation()

            viewer.sync()
            precise_wait(t_cycle_end)
            step_idx += 1

    env.close()


if __name__ == "__main__":
    main()
