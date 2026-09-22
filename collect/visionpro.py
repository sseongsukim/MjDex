"""Collect demonstrations with Vision Pro teleoperation for any registered MjDex hand env.

Keyboard controls (viewer window must be focused):
  r      Reset episode — recalibrates Vision Pro reference, clears buffer
  t / Enter  Save current episode as success
  Esc    Quit

Usage examples:
  python collect/visionpro.py --env-id dual-barcode-meat-v0 --ip 192.168.0.4
  python collect/visionpro.py --env-id single-relocate-meat-v0 --ip 192.168.0.4
  python collect/visionpro.py --env-id dual-hand-dishrack-v0 --data-path /tmp/data

  # Also render the scene in the headset, on top of the local viewer:
  python collect/visionpro.py --env-id dual-hand-dishrack-v0 --ip 192.168.0.4 --stream

With --stream the scene is exported to XML, converted to USDZ and sent to the
headset once; each step then streams body poses only, so VisionOS renders the
scene itself. The local MuJoCo viewer stays up and keeps owning the keyboard.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import time
from typing import Any

MjDex_ROOT = Path(__file__).resolve().parents[1]
if str(MjDex_ROOT) not in sys.path:
    sys.path.insert(0, str(MjDex_ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import yaml

# ─── constants ───────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = (
    MjDex_ROOT / "mjdex" / "assets" / "sharpa_wave" / "avp_sharpa_hand.yaml"
)
SHARPA_URDF_PATHS = {
    "left": MjDex_ROOT / "mjdex" / "assets" / "sharpa_wave" / "left_sharpa_wave.urdf",
    "right": MjDex_ROOT / "mjdex" / "assets" / "sharpa_wave" / "right_sharpa_wave.urdf",
}
VP_TO_MEDIAPIPE = np.asarray(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    dtype=np.int64,
)
SHARPA_URDF_TO_MUJOCO = np.asarray(
    [17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 8, 9, 10, 11, 12],
    dtype=np.int64,
)
POSE_FRAME_MAPS = {
    "identity": np.eye(3, dtype=np.float64),
    "mjdex-table": np.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
}

# Scene placement for --stream, as [x, y, z, yaw_deg] in the headset's frame
# (+x right, +y forward, +z up, origin on the floor under the viewer).
#
# MjDex puts the robot base at the world origin, on the table top, with the table
# running out to +x. The yaw sends MjDex +x to the viewer's forward axis, so the
# table lies out in front and reaching away from yourself drives the arm away
# from you. (It is the inverse of POSE_FRAME_MAPS["mjdex-table"]; that map only
# steers teleop, which --stream leaves alone, but matching them keeps what you
# see consistent with how you move. Retune this yaw if you change --pose-frame.)
#
# Placing the origin at the viewer is the obvious choice and the wrong one: the
# robot then stands inside you and its arm reaches 1.4 m, just under standing eye
# height, so it covers the very workspace you are trying to watch. Setting the
# scene down and forward puts the whole robot below your sight line — you look
# over it at the table. Since teleop is relative, this is purely about the view;
# move it freely with --attach-to, and read the placement --stream prints.
DEFAULT_ATTACH_TO = (0.0, 0.30, 0.35, 90.0)

_GLFW_KEY_R = 82
_GLFW_KEY_T = 84
_GLFW_KEY_ENTER = 257
_GLFW_KEY_ESCAPE = 256


# ─── argument parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect demonstrations with Vision Pro for a MjDex dexterous hand env.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-id",
        default="dual-hand-dishrack-v0",
        help="Registered MjDex env ID (see mjdex/envs.py). Must be a dexterous hand env.",
    )
    parser.add_argument("--ip", default="192.168.0.4", help="Vision Pro streamer IP.")
    parser.add_argument(
        "--hz", type=float, default=60.0, help="Control loop rate (Hz)."
    )
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--retry-sleep", type=float, default=0.01)
    parser.add_argument("--print-rate", type=float, default=1.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Sharpa retargeting YAML config.",
    )
    parser.add_argument(
        "--pose-frame",
        choices=tuple(POSE_FRAME_MAPS),
        default="mjdex-table",
        help="Map Vision Pro wrist deltas into the MjDex world frame.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Directory to save datasets. Defaults to data/<env_name>/.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Starting episode index. Auto-increments from last saved if not set.",
    )
    parser.add_argument(
        "--store-next-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Also render the scene in the Vision Pro (AR). The local viewer stays up.",
    )
    parser.add_argument(
        "--attach-to",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "YAW_DEG"),
        default=list(DEFAULT_ATTACH_TO),
        help="Where to place the MjDex world origin in the headset's Z-up frame "
        "(--stream only). The default stands you where the robot is, looking down "
        "the table. See DEFAULT_ATTACH_TO for how it is derived.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="gRPC port used for the one-off USDZ transfer (--stream only).",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Where to cache the exported scene (--stream only). "
        "Defaults to a per-env directory under the system temp dir.",
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the headset to accept the scene (--stream only).",
    )
    return parser.parse_args()


# ─── rotation / pose helpers ──────────────────────────────────────────────────


def project_to_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rot)
    projected = u @ vh
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vh
    return projected


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(9))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat / np.linalg.norm(quat)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def apply_relative_rotation(
    vp_rot: np.ndarray,
    vp_init_rot: np.ndarray,
    robot_init_quat: np.ndarray,
    pose_frame: np.ndarray,
) -> np.ndarray:
    vp_delta_rot = vp_rot @ vp_init_rot.T
    vp_delta_rot = pose_frame @ vp_delta_rot @ pose_frame.T
    robot_init_rot = quat_wxyz_to_matrix(robot_init_quat)
    robot_target_rot = project_to_rotation_matrix(vp_delta_rot @ robot_init_rot)
    return matrix_to_quat_wxyz(robot_target_rot)


def apply_relative_position(
    vp_pos: np.ndarray,
    vp_init_pos: np.ndarray,
    robot_init_pos: np.ndarray,
    pose_frame: np.ndarray,
    pos_scale: float,
) -> np.ndarray:
    vp_delta_pos = np.asarray(vp_pos - vp_init_pos, dtype=np.float64)
    return robot_init_pos + (pose_frame @ vp_delta_pos) * pos_scale


def convert_vp_to_mediapipe(fingers_mat: np.ndarray) -> np.ndarray:
    return np.asarray(fingers_mat, dtype=np.float64)[VP_TO_MEDIAPIPE, :3, 3].astype(
        np.float32
    )


# ─── Vision Pro streaming helpers ─────────────────────────────────────────────


def _lookup(stream_data: Any, key: str) -> Any:
    return stream_data[key][0]


def _as_single_transform(stream_data: Any, key: str) -> np.ndarray:
    value = np.asarray(stream_data[key], dtype=np.float64)
    if value.shape == (4, 4):
        return value
    value = np.asarray(_lookup(stream_data, key), dtype=np.float64)
    if value.shape == (4, 4):
        return value
    if value.ndim == 3 and value.shape[-2:] == (4, 4):
        return value[-1]
    raise ValueError(f"{key} should have shape (4,4) or (T,4,4), got {value.shape}")


def _as_single_fingers(stream_data: Any, key: str) -> np.ndarray:
    value = np.asarray(stream_data[key], dtype=np.float64)
    if value.shape == (25, 4, 4):
        return value
    value = np.asarray(_lookup(stream_data, key), dtype=np.float64)
    if value.shape == (25, 4, 4):
        return value
    if value.ndim == 4 and value.shape[-3:] == (25, 4, 4):
        return value[-1]
    raise ValueError(
        f"{key} should have shape (25,4,4) or (T,25,4,4), got {value.shape}"
    )


def _latest_valid_frame(streamer: Any, retry_sleep: float) -> dict:
    while True:
        stream_data = streamer.get_latest()
        if stream_data is None:
            time.sleep(retry_sleep)
            continue
        try:
            right_wrist = _as_single_transform(stream_data, "right_wrist")
            left_wrist = _as_single_transform(stream_data, "left_wrist")
            return {
                "left_wrist_rot": left_wrist[:3, :3].copy(),
                "left_wrist_pos": left_wrist[:3, 3].copy(),
                "left_fingers": _as_single_fingers(stream_data, "left_fingers").copy(),
                "right_wrist_rot": right_wrist[:3, :3].copy(),
                "right_wrist_pos": right_wrist[:3, 3].copy(),
                "right_fingers": _as_single_fingers(
                    stream_data, "right_fingers"
                ).copy(),
            }
        except (KeyError, IndexError, ValueError):
            time.sleep(retry_sleep)


# ─── Sharpa retargeting ───────────────────────────────────────────────────────


def _load_sharpa_config(config_path: Path, side: str) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("robot", {})
    config["robot"]["type"] = "sharpa_hand"
    config["robot"]["urdf_path"] = str(SHARPA_URDF_PATHS[side].resolve())
    return config


class SharpaRetargeter:
    def __init__(self, config_path: Path, side: str, ctrl_range: np.ndarray) -> None:
        from mjdex.teleop.retargeting import Retargeter

        self.ctrl_range = np.asarray(ctrl_range, dtype=np.float32)
        self.last_ctrl = np.mean(self.ctrl_range, axis=1).astype(np.float32)
        self.ctrl_sign = -1.0 if side == "right" else 1.0
        config = _load_sharpa_config(config_path, side)
        self.retargeter = Retargeter.from_config(config, hand_side=side)

    def retarget(self, fingers_frame: np.ndarray) -> np.ndarray:
        keypoints = convert_vp_to_mediapipe(fingers_frame)
        if np.allclose(keypoints, 0.0):
            return self.last_ctrl.copy()
        urdf_qpos = self.retargeter.retarget(keypoints)
        mujoco_ctrl = np.asarray(urdf_qpos[SHARPA_URDF_TO_MUJOCO], dtype=np.float32)
        mujoco_ctrl *= self.ctrl_sign
        self.last_ctrl = np.clip(
            mujoco_ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1]
        )
        return self.last_ctrl.copy()


# ─── action layout helpers ────────────────────────────────────────────────────


def _arm_slice(start: int, hand_dof: int) -> tuple[slice, slice, int]:
    """Returns (arm_slice, hand_slice, next_start) for [x,y,z,qw,qx,qy,qz, hand...]."""
    arm = slice(start, start + 7)
    hand = slice(start + 7, start + 7 + hand_dof)
    return arm, hand, start + 7 + hand_dof


# ─── data collection helpers ──────────────────────────────────────────────────


def _default_data_path(env_id: str) -> Path:
    env_name = env_id.removesuffix("-v0").replace("-", "_")
    return Path("data") / env_name


def _next_dataset_idx(data_path: Path) -> int:
    indices = []
    for path in data_path.glob("dataset_*.pkl"):
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=-1) + 1


def _new_dataset() -> defaultdict:
    dataset: defaultdict = defaultdict(list)
    dataset["observations"] = defaultdict(list)
    dataset["next_observations"] = defaultdict(list)
    dataset["actions"] = []
    dataset["terminals"] = []
    dataset["rewards"] = []
    return dataset


def _stack_dataset(value: Any) -> Any:
    if isinstance(value, (defaultdict, dict)):
        return {k: _stack_dataset(v) for k, v in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _save_dataset(dataset: defaultdict, data_path: Path, idx: int) -> Path:
    data_path.mkdir(parents=True, exist_ok=True)
    save_path = data_path / f"dataset_{idx}.pkl"
    with save_path.open("wb") as f:
        pickle.dump(_stack_dataset(dataset), f, protocol=4)
    return save_path


def _add_transition(
    dataset: defaultdict,
    obs: dict,
    next_obs: dict,
    action: np.ndarray,
    reward: float,
    terminal: bool,
    store_next_observations: bool,
) -> None:
    for key, value in obs.items():
        dataset["observations"][key].append(np.asarray(value).copy())
    if store_next_observations:
        for key, value in next_obs.items():
            dataset["next_observations"][key].append(np.asarray(value).copy())
    dataset["actions"].append(action.copy())
    dataset["rewards"].append(float(reward))
    dataset["terminals"].append(bool(terminal))


def _obs_dict(raw: dict) -> dict:
    return {k: np.asarray(v).copy() for k, v in raw.items()}


# ─── AR streaming ─────────────────────────────────────────────────────────────


def _export_scene(env: Any, env_id: str, scene_dir: Path | None) -> Path:
    """Write the env's MJCF (and its assets) to disk and return the XML path.

    MjDex assembles every scene programmatically with dm_control and compiles it
    from a string, but ``configure_mujoco`` converts to USDZ from a *path* so the
    meshes and textures have to resolve next to the XML. Exporting once per env
    gives the converter something to read; the result is reused on later runs.
    """
    from dm_control import mjcf

    if scene_dir is None:
        scene_dir = Path(tempfile.gettempdir()) / "mjdex_ar_scenes" / env_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    xml_path = scene_dir / "scene.xml"
    if not xml_path.exists():
        mjcf.export_with_assets(env.mjcf_model, str(scene_dir), "scene.xml")
    return xml_path


def _describe_placement(env: Any, attach_to: list[float], eye_height: float = 1.55) -> None:
    """Print where the scene lands for the viewer, so --attach-to is tunable.

    Occlusion is the thing that goes wrong here, and it is invisible from the
    numbers alone: what matters is whether the robot sits below your line of
    sight to the workspace, not where the origin is.
    """
    x, y, z, yaw = attach_to
    c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    rot = np.asarray([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    offset = np.asarray([x, y, z], dtype=np.float64)

    def to_viewer(point: np.ndarray) -> np.ndarray:
        return rot @ np.asarray(point, dtype=np.float64) + offset

    robot_top = float(env.data.xpos[1:, 2].max())
    base = to_viewer([0.0, 0.0, 0.0])
    top = to_viewer([0.0, 0.0, robot_top])
    far = to_viewer([0.8, 0.0, 0.0])

    print(
        f"[stream] robot base {base[1]:+.2f} m ahead at {base[2]:.2f} m; "
        f"table reaches {far[1]:+.2f} m; robot tops out at {top[2]:.2f} m."
    )
    clearance = eye_height - top[2]
    if clearance < 0.25:
        print(
            f"[stream] Only {clearance:+.2f} m between a standing eye and the top "
            "of the robot — it will block the table. Lower the scene (smaller z) "
            "and push it further out (larger y) with --attach-to."
        )
    else:
        print(f"[stream] {clearance:.2f} m of headroom over the robot at eye level.")


def _start_streaming(
    streamer: Any,
    env: Any,
    env_id: str,
    attach_to: list[float],
    grpc_port: int,
    scene_dir: Path | None,
    timeout: float,
) -> bool:
    """Send the scene to the headset and start pose streaming. Returns success.

    Safe to call after the first ``env.reset()``: MjDex only recompiles when the
    model is marked dirty, so ``env.model``/``env.data`` outlive later resets and
    the streamer's references stay valid for the whole session.
    """
    xml_path = _export_scene(env, env_id, scene_dir)
    print(f"[stream] Scene: {xml_path}")
    streamer.configure_mujoco(
        xml_path=str(xml_path),
        model=env.model,
        data=env.data,
        relative_to=list(attach_to),
        grpc_port=grpc_port,
    )
    # configure_mujoco() ends with set_origin("sim"), which silently re-expresses
    # every incoming wrist and finger pose in the attach_to frame. --pose-frame
    # already maps the Vision Pro's native frame into MjDex world, so leaving it
    # on "sim" applies that rotation twice: the horizontal axes swap and the
    # wrist orientation is conjugated by the wrong basis. Put it back so --stream
    # only ever adds a picture and leaves teleop byte-for-byte unchanged.
    streamer.set_origin("avp")

    streamer.start_webrtc()
    print("[stream] Waiting for the headset to accept the scene...")
    if not streamer.wait_for_sim_channel(timeout=timeout):
        print(
            "[stream] Timed out. Continuing with the local viewer only — "
            "collection is unaffected."
        )
        return False
    print(f"[stream] Live. Scene placed at {attach_to} (x, y, z, yaw deg).")
    _describe_placement(env, attach_to)
    return True


# ─── calibration ──────────────────────────────────────────────────────────────


def _calibrate(
    env: Any,
    streamer: Any,
    retry_sleep: float,
    is_dual: bool,
) -> dict:
    """Block until a valid VP frame arrives, then record it as the reference pose."""
    frame = _latest_valid_frame(streamer, retry_sleep)
    calib: dict[str, np.ndarray] = {}
    if is_dual:
        left_pose = env.ee_pose("left")
        calib["left_vp_init_pos"] = frame["left_wrist_pos"].copy()
        calib["left_vp_init_rot"] = frame["left_wrist_rot"].copy()
        calib["left_robot_init_pos"] = left_pose[:3].copy()
        calib["left_robot_init_quat"] = left_pose[3:7].copy()
        right_pose = env.ee_pose("right")
    else:
        right_pose = env.ee_pose()
    calib["right_vp_init_pos"] = frame["right_wrist_pos"].copy()
    calib["right_vp_init_rot"] = frame["right_wrist_rot"].copy()
    calib["right_robot_init_pos"] = right_pose[:3].copy()
    calib["right_robot_init_quat"] = right_pose[3:7].copy()
    return calib


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    pose_frame = POSE_FRAME_MAPS[args.pose_frame]
    dt = 1.0 / args.hz if args.hz > 0.0 else 0.0

    try:
        from avp_stream import VisionProStreamer
    except ImportError as exc:
        raise ImportError(
            "Vision Pro teleoperation requires avp_stream. "
            "Install it in the mjdex environment."
        ) from exc

    import gymnasium as gym
    from mjdex import register_mjdex_envs
    from mjdex.dual_arm import DualArmHandEnv
    from mjdex.single_arm import SingleArmHandEnv

    register_mjdex_envs()

    # ── create env ────────────────────────────────────────────────────────────
    gym_env = gym.make(
        args.env_id,
        action_type="pos",
        control_timestep=dt if dt > 0.0 else 0.02,
        include_table=not args.no_table,
        disable_env_checker=True,
    )
    env = gym_env.unwrapped

    if isinstance(env, DualArmHandEnv):
        is_dual = True
    elif isinstance(env, SingleArmHandEnv):
        is_dual = False
    else:
        raise ValueError(
            f"'{args.env_id}' is not a dexterous hand environment. "
            "Use collect/collect_kb.py for gripper environments."
        )

    # ── data path ─────────────────────────────────────────────────────────────
    data_path = (
        Path(args.data_path) if args.data_path else _default_data_path(args.env_id)
    )
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_idx = _next_dataset_idx(data_path) if args.idx is None else args.idx

    # ── env reset + arm/hand slices ───────────────────────────────────────────
    env.reset()

    if is_dual:
        left_hand_dof = env._hand_dof_by_side["left"]
        right_hand_dof = env._hand_dof_by_side["right"]
        left_arm_sl, left_hand_sl, right_start = _arm_slice(0, left_hand_dof)
        right_arm_sl, right_hand_sl, _ = _arm_slice(right_start, right_hand_dof)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[left_arm_sl] = env.ee_pose("left")
        action[right_arm_sl] = env.ee_pose("right")
        action[left_hand_sl] = env._hand_neutral_qpos(env._arms["left"])
        action[right_hand_sl] = env._hand_neutral_qpos(env._arms["right"])

        left_retargeter = SharpaRetargeter(
            args.config,
            "left",
            env.model.actuator_ctrlrange[env._arms["left"].gripper_actuator_ids],
        )
        right_retargeter = SharpaRetargeter(
            args.config,
            "right",
            env.model.actuator_ctrlrange[env._arms["right"].gripper_actuator_ids],
        )
    else:
        hand_dof = env._hand_dof
        right_arm_sl, right_hand_sl, _ = _arm_slice(0, hand_dof)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[right_arm_sl] = env.ee_pose()
        action[right_hand_sl] = env._hand_neutral_qpos()

        right_retargeter = SharpaRetargeter(
            args.config,
            "right",
            env.model.actuator_ctrlrange[env._gripper_actuator_ids],
        )

    # ── Vision Pro streamer + initial calibration ─────────────────────────────
    streamer = VisionProStreamer(ip=args.ip)
    mode = "dual" if is_dual else "single-right"
    print(
        f"Env: {args.env_id}  |  mode: {mode}  |  "
        f"action_dim={env.action_space.shape[0]}"
    )
    print(f"Data path: {data_path}  |  Starting index: {dataset_idx}")

    # Bring the AR scene up *before* calibrating. Converting the scene and
    # waiting for the headset to accept it takes seconds, and the calibration
    # frame is the origin every later wrist delta is measured from — capture it
    # first and the operator's hands have long since drifted by the time the
    # control loop starts, which offsets both position and orientation.
    streaming = False
    if args.stream:
        streaming = _start_streaming(
            streamer=streamer,
            env=env,
            env_id=args.env_id,
            attach_to=args.attach_to,
            grpc_port=args.grpc_port,
            scene_dir=args.scene_dir,
            timeout=args.stream_timeout,
        )

    print("Hold your hands in the pose you want to start from...")
    calib = _calibrate(env, streamer, args.retry_sleep, is_dual)
    print("Calibrated. Controls: r=reset  t/Enter=save  Esc=quit")

    # ── key state (thread-safe events) ────────────────────────────────────────
    reset_flag = threading.Event()
    save_flag = threading.Event()
    quit_flag = threading.Event()

    def key_callback(key: int) -> None:
        # mujoco.viewer's KeyCallbackType is Callable[[int], None]: it hands over
        # the GLFW keycode of a press and nothing else, so there is no action to
        # filter on here.
        if key == _GLFW_KEY_R:
            reset_flag.set()
        elif key in (_GLFW_KEY_T, _GLFW_KEY_ENTER):
            save_flag.set()
        elif key == _GLFW_KEY_ESCAPE:
            quit_flag.set()

    # ── dataset state ─────────────────────────────────────────────────────────
    dataset = _new_dataset()
    obs = _obs_dict(env.compute_observation())
    step_count = 0
    next_print_time = time.monotonic()

    def _reset_action_to_current() -> None:
        if is_dual:
            action[left_arm_sl] = env.ee_pose("left")
            action[right_arm_sl] = env.ee_pose("right")
            action[left_hand_sl] = env._hand_neutral_qpos(env._arms["left"])
            action[right_hand_sl] = env._hand_neutral_qpos(env._arms["right"])
        else:
            action[right_arm_sl] = env.ee_pose()
            action[right_hand_sl] = env._hand_neutral_qpos()

    try:
        with mujoco.viewer.launch_passive(
            env.model, env.data, key_callback=key_callback
        ) as viewer:
            while viewer.is_running() and not quit_flag.is_set():
                loop_start = time.monotonic()

                # ── reset ─────────────────────────────────────────────────────
                if reset_flag.is_set():
                    reset_flag.clear()
                    env.reset()
                    calib = _calibrate(env, streamer, args.retry_sleep, is_dual)
                    _reset_action_to_current()
                    dataset = _new_dataset()
                    obs = _obs_dict(env.compute_observation())
                    viewer.sync()
                    if streaming:
                        streamer.update_sim()
                    print(
                        f"[reset] Recalibrated. Buffer cleared. " f"idx={dataset_idx}"
                    )
                    continue

                # ── save ──────────────────────────────────────────────────────
                if save_flag.is_set():
                    save_flag.clear()
                    n = len(dataset["actions"])
                    if n > 0:
                        if dataset["terminals"]:
                            dataset["terminals"][-1] = True
                        save_path = _save_dataset(dataset, data_path, dataset_idx)
                        print(
                            f"[save] dataset_{dataset_idx}.pkl  "
                            f"({n} transitions) → {save_path}"
                        )
                        dataset_idx += 1
                        dataset = _new_dataset()
                        obs = _obs_dict(env.compute_observation())
                    else:
                        print("[save] Nothing to save (empty episode).")

                # ── Vision Pro frame → action ──────────────────────────────────
                frame = _latest_valid_frame(streamer, args.retry_sleep)

                if is_dual:
                    action[left_arm_sl.start : left_arm_sl.start + 3] = (
                        apply_relative_position(
                            frame["left_wrist_pos"],
                            calib["left_vp_init_pos"],
                            calib["left_robot_init_pos"],
                            pose_frame,
                            args.pos_scale,
                        )
                    )
                    action[left_arm_sl.start + 3 : left_arm_sl.stop] = (
                        apply_relative_rotation(
                            frame["left_wrist_rot"],
                            calib["left_vp_init_rot"],
                            calib["left_robot_init_quat"],
                            pose_frame,
                        )
                    )
                    action[left_hand_sl] = left_retargeter.retarget(
                        frame["left_fingers"]
                    )

                action[right_arm_sl.start : right_arm_sl.start + 3] = (
                    apply_relative_position(
                        frame["right_wrist_pos"],
                        calib["right_vp_init_pos"],
                        calib["right_robot_init_pos"],
                        pose_frame,
                        args.pos_scale,
                    )
                )
                action[right_arm_sl.start + 3 : right_arm_sl.stop] = (
                    apply_relative_rotation(
                        frame["right_wrist_rot"],
                        calib["right_vp_init_rot"],
                        calib["right_robot_init_quat"],
                        pose_frame,
                    )
                )
                action[right_hand_sl] = right_retargeter.retarget(
                    frame["right_fingers"]
                )

                # ── step + collect ─────────────────────────────────────────────
                next_raw_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = _obs_dict(next_raw_obs)

                _add_transition(
                    dataset,
                    obs,
                    next_obs,
                    action,
                    float(reward),
                    bool(terminated or truncated),
                    args.store_next_observations,
                )
                obs = next_obs
                step_count += 1

                viewer.sync()
                if streaming:
                    streamer.update_sim()

                # ── status print ──────────────────────────────────────────────
                now = time.monotonic()
                if args.print_rate > 0.0 and now >= next_print_time:
                    n_buf = len(dataset["actions"])
                    r_pos = action[right_arm_sl.start : right_arm_sl.start + 3]
                    if is_dual:
                        l_pos = action[left_arm_sl.start : left_arm_sl.start + 3]
                        print(
                            f"step={step_count} buf={n_buf} "
                            f"L={np.round(l_pos, 3)} R={np.round(r_pos, 3)}"
                        )
                    else:
                        print(f"step={step_count} buf={n_buf} pos={np.round(r_pos, 3)}")
                    next_print_time = now + 1.0 / args.print_rate

                # ── rate limit ────────────────────────────────────────────────
                if dt > 0.0:
                    sleep_time = dt - (time.monotonic() - loop_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
