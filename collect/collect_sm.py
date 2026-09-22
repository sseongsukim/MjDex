"""Collect demonstrations with a 3D SpaceMouse for any registered MjDex env."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import time
from multiprocessing.managers import SharedMemoryManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        default=None,
        choices=["ur5e", "fr3"],
        help="Optional legacy override; prefer choosing the robot via --env-id.",
    )
    parser.add_argument("--gripper", default="robotiq_2f85")
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Control frequency in Hz (default: 50).",
    )
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--max-pos-speed", type=float, default=0.18)
    parser.add_argument("--max-rot-speed", type=float, default=0.45)
    parser.add_argument(
        "--sm-dpos-scalar",
        type=float,
        nargs="+",
        default=[1.8],
        help="SpaceMouse translation multiplier. Pass one value or x y z values.",
    )
    parser.add_argument(
        "--sm-drot-scalar",
        type=float,
        nargs="+",
        default=[2.0],
        help="SpaceMouse rotation multiplier. Pass one value or roll pitch yaw values.",
    )
    parser.add_argument("--position-action-scale", type=float, default=0.05)
    parser.add_argument("--action-type", default="pos", choices=["pos"])
    parser.add_argument("--env-id", default="single-mugrack-v0")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument(
        "--gripper-width-record-threshold",
        type=float,
        default=0.001,
        help="Record a transition when gripper_width changes by more than this.",
    )
    parser.add_argument(
        "--store-next-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rotation", action="store_true", default=True)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--show-target", action="store_true")
    return parser.parse_args()


def next_dataset_idx(data_path: Path) -> int:
    indices = []
    for path in data_path.glob("dataset_*.pkl"):
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def default_data_path(env_id: str) -> Path:
    env_name = env_id.removesuffix("-v0").replace("-", "_")
    return Path("data") / env_name


def is_mugrack_env(env_id: str) -> bool:
    """Return whether an environment uses the MugRack collection format."""

    return "mugrack" in env_id.lower()


def parse_axis_scalar(values: list[float], name: str) -> np.ndarray:
    import numpy as np

    if len(values) == 1:
        return np.full(3, values[0], dtype=np.float64)
    if len(values) == 3:
        return np.asarray(values, dtype=np.float64)
    raise ValueError(f"{name} must receive either one value or three x y z values.")


def new_dataset() -> defaultdict:
    dataset = defaultdict(list)
    dataset["observations"] = defaultdict(list)
    dataset["next_observations"] = defaultdict(list)
    dataset["actions"] = []
    dataset["terminals"] = []
    dataset["rewards"] = []
    return dataset


def stack_dataset(value):
    import numpy as np

    if isinstance(value, defaultdict) or isinstance(value, dict):
        return {k: stack_dataset(v) for k, v in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def save_dataset(dataset: defaultdict, data_path: Path, idx: int) -> Path:
    data_path.mkdir(parents=True, exist_ok=True)
    save_path = data_path / f"dataset_{idx}.pkl"
    with save_path.open("wb") as f:
        pickle.dump(stack_dataset(dataset), f, protocol=4)
    return save_path


def observation_dict(env, observation: dict[str, np.ndarray]) -> dict:
    import numpy as np

    return {key: np.asarray(value).copy() for key, value in observation.items()}


def add_transition(
    dataset: defaultdict,
    observation: dict,
    next_observation: dict,
    action: np.ndarray,
    reward: float,
    terminal: bool,
    store_next_observations: bool,
    joint_action: np.ndarray | None = None,
) -> None:
    for key, value in observation.items():
        dataset["observations"][key].append(value)
    if store_next_observations:
        for key, value in next_observation.items():
            dataset["next_observations"][key].append(value)
    if joint_action is None:
        dataset["actions"].append(action)
    else:
        dataset["joint_actions"].append(joint_action)
        dataset["ee_actions"].append(action)
    dataset["rewards"].append(reward)
    dataset["terminals"].append(terminal)


def open_gripper(env, warmup_steps: int = 20) -> None:
    import numpy as np

    action = np.concatenate([env.ee_pose(), [1.0]]).astype(np.float32)
    for _ in range(warmup_steps):
        env.step(action)
    env._elapsed_steps = 0


def make_actions(
    target_pose: np.ndarray,
    sm_state: np.ndarray,
    gripper_closed: bool,
    dt: float,
    max_pos_speed: float,
    sm_dpos_scalar: np.ndarray,
    max_rot_speed: float,
    sm_drot_scalar: np.ndarray,
    rotation: bool,
    robot_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    from mjdex.transform import euler_delta_quat, quat_mul

    dpos = sm_state[:3] * max_pos_speed * dt * sm_dpos_scalar

    if rotation:
        drot = sm_state[3:] * max_rot_speed * dt * sm_drot_scalar
        if robot_name == "ur5e":
            # UR5e's attachment frame is turned 180 degrees around local z
            # relative to FR3's at home. Match FR3's local rotation controls.
            drot = drot * np.array([-1.0, -1.0, 1.0])
        dquat = euler_delta_quat(*drot)

    else:
        dquat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    target_pos = target_pose[:3] + dpos
    target_quat = quat_mul(target_pose[3:], dquat)
    target_quat = target_quat / np.linalg.norm(target_quat)
    next_target_pose = np.concatenate([target_pos, target_quat]).astype(np.float64)
    pos_gripper_action = -1.0 if gripper_closed else 1.0
    pos_action = np.concatenate([target_pos, target_quat, [pos_gripper_action]]).astype(
        np.float32
    )
    return pos_action, next_target_pose


def action_taken(sm_state: np.ndarray, gripper_toggled: bool, rotation: bool) -> bool:
    import numpy as np

    moved = not np.allclose(sm_state[:3], 0.0)
    rotated = rotation and not np.allclose(sm_state[3:], 0.0)
    return bool(gripper_toggled or moved or rotated)


def gripper_width(observation: dict) -> float | None:
    import numpy as np

    if "gripper_width" not in observation:
        return None
    return float(np.asarray(observation["gripper_width"]).reshape(-1)[0])


def print_usage(env_id: str, store_joint_actions: bool = False) -> None:
    mode = " (JP)" if store_joint_actions else ""
    print(f"============== MjDex SpaceMouse Collect{mode}: {env_id} ==============")
    print("Move SpaceMouse: translate EEF")
    print("Twist SpaceMouse: rotate EEF with --rotation")
    print("Left button: toggle gripper open/close")
    print("Right button: reset environment")
    print("Keyboard r: reset environment and clear current dataset")
    print("Keyboard t: save current dataset")
    print("Keyboard Esc: quit")
    print("Close viewer or press Ctrl-C to quit")
    if store_joint_actions:
        print("Stored action: IK-solved joint positions + normalized gripper [-1, 1]")
    print("=============================================================")


def main() -> None:
    args = parse_args()
    import gymnasium as gym
    import mujoco.viewer
    import numpy as np

    from mjdex import register_mjdex_envs
    from mjdex.teleop import CollectEnum, KeyboardTeleop
    from mjdex.teleop.spacemouse.spacemouse_shared_memory import Spacemouse
    from utils.logger import Logger

    register_mjdex_envs()

    store_joint_actions = is_mugrack_env(args.env_id)
    dt = 1.0 / args.frequency
    sm_dpos_scalar = parse_axis_scalar(args.sm_dpos_scalar, "--sm-dpos-scalar")
    sm_drot_scalar = parse_axis_scalar(args.sm_drot_scalar, "--sm-drot-scalar")

    data_path = (
        Path(args.data_path) if args.data_path else default_data_path(args.env_id)
    )
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_idx = next_dataset_idx(data_path) if args.idx is None else args.idx

    env = None
    keyboard = None
    env_kwargs = {}
    if args.robot is not None:
        env_kwargs["robot"] = args.robot

    gym_env = gym.make(
        args.env_id,
        gripper=args.gripper,
        action_type=args.action_type,
        position_action_scale=args.position_action_scale,
        control_timestep=dt,
        include_table=not args.no_table,
        task_kwargs={"show_target": args.show_target},
        disable_env_checker=True,
        **env_kwargs,
    )
    env = gym_env.unwrapped
    env.reset()
    open_gripper(env)
    keyboard = KeyboardTeleop()

    print_usage(args.env_id, store_joint_actions=store_joint_actions)
    print(f"Initial EEF pose: {env.ee_pose()}")
    print(f"Dataset path: {data_path}")
    print(f"Current dataset index: {dataset_idx}")
    Logger.info("sm_dpos_scalar=%s | sm_drot_scalar=%s", sm_dpos_scalar, sm_drot_scalar)

    gripper_closed = False
    prev_buttons = np.zeros(2, dtype=bool)
    target_pose = env.ee_pose()
    dataset = new_dataset()
    observation = observation_dict(env, env.compute_observation())
    t_start = time.monotonic()
    step_idx = 0
    episode_step = 0

    try:
        with SharedMemoryManager() as shm_manager:
            with Spacemouse(
                shm_manager=shm_manager,
                deadzone=args.deadzone,
                n_buttons=2,
            ) as spacemouse:
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                    while viewer.is_running():
                        t_cycle_end = t_start + (step_idx + 1) * dt

                        sm_state = spacemouse.get_motion_state_transformed()
                        buttons = spacemouse.get_button_state()
                        keyboard_action = keyboard.get_action()
                        collect_enum = keyboard_action.collect_enum

                        gripper_toggled = False
                        if buttons[0] and not prev_buttons[0]:
                            gripper_closed = not gripper_closed
                            gripper_toggled = True
                            state = "close" if gripper_closed else "open"
                            print(f"Gripper: {state}")

                        reset_requested = (
                            buttons[1] and not prev_buttons[1]
                        ) or collect_enum == CollectEnum.RESET
                        if reset_requested:
                            env.reset()
                            open_gripper(env)
                            gripper_closed = False
                            target_pose = env.ee_pose()
                            dataset = new_dataset()
                            episode_step = 0
                            observation = observation_dict(
                                env,
                                env.compute_observation(),
                            )
                            print("Environment reset")
                            print("Current dataset cleared")
                            prev_buttons = buttons.copy()
                            viewer.sync()
                            precise_wait(t_cycle_end)
                            step_idx += 1
                            continue

                        prev_buttons = buttons.copy()

                        ee_action, target_pose = make_actions(
                            target_pose=target_pose,
                            sm_state=sm_state,
                            gripper_closed=gripper_closed,
                            dt=dt,
                            max_pos_speed=args.max_pos_speed,
                            sm_dpos_scalar=sm_dpos_scalar,
                            max_rot_speed=args.max_rot_speed,
                            sm_drot_scalar=sm_drot_scalar,
                            rotation=args.rotation,
                            robot_name=env.robot_name,
                        )
                        next_raw_observation, reward, terminated, truncated, info = (
                            env.step(ee_action)
                        )
                        next_observation = observation_dict(env, next_raw_observation)
                        prev_width = gripper_width(observation)
                        next_width = gripper_width(next_observation)
                        gripper_state_changed = (
                            prev_width is not None
                            and next_width is not None
                            and abs(next_width - prev_width)
                            > args.gripper_width_record_threshold
                        )

                        if action_taken(
                            sm_state, gripper_toggled, args.rotation
                        ) or gripper_state_changed:
                            episode_step += 1
                            joint_action = None
                            if store_joint_actions:
                                # set_control() writes the IK-solved arm target to
                                # these controls during env.step().
                                joint_commanded = env.data.ctrl[
                                    env._arm_actuator_ids
                                ].copy()
                                gripper_action = np.float32(
                                    -1.0 if gripper_closed else 1.0
                                )
                                joint_action = np.concatenate(
                                    [joint_commanded, [gripper_action]]
                                ).astype(np.float32)
                            add_transition(
                                dataset=dataset,
                                observation=observation,
                                next_observation=next_observation,
                                action=ee_action,
                                reward=float(reward),
                                terminal=bool(terminated or truncated),
                                store_next_observations=args.store_next_observations,
                                joint_action=joint_action,
                            )
                            action_key = (
                                "joint_actions" if store_joint_actions else "actions"
                            )
                            Logger.info(
                                "Collect step %04d | dataset_%05d | actions=%d | success=%s",
                                episode_step,
                                dataset_idx,
                                len(dataset[action_key]),
                                info.get("success", "N/A"),
                            )
                            if store_joint_actions:
                                Logger.info("reward: %s", reward)

                        observation = next_observation

                        if collect_enum == CollectEnum.SUCCESS:
                            if dataset["terminals"]:
                                dataset["terminals"][-1] = True
                            save_path = save_dataset(dataset, data_path, dataset_idx)
                            Logger.info(
                                "Saved dataset_%05d | steps=%d | path=%s",
                                dataset_idx,
                                episode_step,
                                save_path,
                            )
                            dataset_idx += 1
                            dataset = new_dataset()
                            episode_step = 0

                        if (
                            keyboard_action.quit
                            or collect_enum == CollectEnum.TERMINATE
                        ):
                            break

                        viewer.sync()
                        precise_wait(t_cycle_end)
                        step_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        if keyboard is not None:
            keyboard.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
