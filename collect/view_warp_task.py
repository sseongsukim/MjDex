"""Inspect GPU-parallel MjDex tasks in the MuJoCo viewer.

Physics advances through WarpVectorEnv.step on the GPU. The viewer displays a
CPU copy of one selected world; it does not run a second physics simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from queue import SimpleQueue

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env", default="single-mugrack-v0")
    parser.add_argument(
        "--list", action="store_true", help="List registered tasks and exit."
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--world-index",
        type=int,
        default=0,
        help="World selected for reset, pose printing and --image; with "
        "--layout single it is also the world on screen.",
    )
    parser.add_argument("--mode", choices=["hold", "static"], default="hold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Viewer real-time multiplier."
    )
    parser.add_argument("--control-timestep", type=float)
    parser.add_argument(
        "--layout",
        choices=["grid", "single"],
        default="grid",
        help="Draw every world on a spaced grid, or mirror one world at a time.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Grid pitch in meters; measured from the scene footprint by default.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Gap in meters left between measured world footprints.",
    )
    parser.add_argument(
        "--camera",
        help="Model camera to frame. Default: the whole grid, or 'render_camera' "
        "with --layout single.",
    )
    parser.add_argument("--lock-camera", action="store_true")
    parser.add_argument("--hide-ui", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--reset-pool-size", type=int, default=0)
    parser.add_argument("--njmax", type=int, default=1024)
    parser.add_argument("--naconmax", type=int)
    parser.add_argument(
        "--headless", action="store_true", help="Run without opening a window."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Control/view iterations; default: unlimited in viewer, 100 headless.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup steps for headless timing."
    )
    parser.add_argument(
        "--image", help="Save every world's final RGB frame as one tiled image."
    )
    args = parser.parse_args()
    if args.list:
        return args
    if args.num_envs < 1 or not 0 <= args.world_index < args.num_envs:
        parser.error(
            "--num-envs must be positive and --world-index must be within the batch"
        )
    if not 0 < args.speed < float("inf"):
        parser.error("--speed must be positive and finite")
    if args.spacing is not None and not all(pitch > 0 for pitch in args.spacing):
        parser.error("--spacing must be positive")
    if args.margin < 0:
        parser.error("--margin must be nonnegative")
    if args.steps is not None and args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup nonnegative")
    if (
        args.reset_pool_size < 0
        or args.njmax < 1
        or args.naconmax is not None
        and args.naconmax < 1
    ):
        parser.error("Reset pool size must be nonnegative; capacities must be positive")
    if args.headless and args.steps is None:
        args.steps = 100
    return args


def hold_actions(env, observation):
    """Hold the reset end-effector/joint targets with hands open."""
    import jax.numpy as jnp

    parts = []
    for arm in env._arms:
        prefix = arm.prefix
        parts.append(
            observation[prefix + "arm_joint_position"]
            if env._joint_control
            else observation[prefix + "ee_pose"]
        )
        parts.append(
            observation[prefix + "hand_joint_position"]
            if arm.hand_qpos is not None
            else -jnp.ones((env.num_envs, 1))
        )
    return jnp.concatenate(parts, axis=-1)


def sync_world(env, world_index):
    """Copy only the selected world to the CPU mirror used by the viewer."""
    import jax
    import mujoco

    from mjdex.mujoco_warp.vector_env import _MODEL_FIELDS, _STATE_FIELDS

    values = jax.device_get(
        {
            **{k: getattr(env.data, k)[world_index] for k in _STATE_FIELDS},
            **{k: env._model_fields[k][world_index] for k in _MODEL_FIELDS},
        }
    )
    cpu = env._env
    for k in _MODEL_FIELDS:
        getattr(cpu.model, k)[:] = values[k]
    for k in _STATE_FIELDS:
        value = values[k]
        if value.ndim:
            getattr(cpu.data, k)[:] = value
        else:
            setattr(cpu.data, k, value.item())
    mujoco.mj_forward(cpu.model, cpu.data)
    return cpu


class SingleWorldScene:
    """CPU mirror of one selected world, switched with ``[`` and ``]``."""

    def __init__(self, env, args):
        self._env = env
        self._cpu = sync_world(env, args.world_index)
        self.model = self._cpu.model
        self.data = self._cpu.data

    def refresh(self, world):
        sync_world(self._env, world)

    def frame_camera(self, viewer, args):
        from collect.view_task import set_initial_camera

        set_initial_camera(
            self._cpu, viewer, args.camera or "render_camera", args.lock_camera
        )

    def describe(self, args):
        return f"Viewing world {args.world_index} of {self._env.num_envs}"


class GridScene:
    """Every world at once, as translated copies inside one composite model."""

    def __init__(self, env, args):
        from mjdex.mujoco_warp.composite import CompositeViewerModel, grid_shape

        pitch = tuple(args.spacing) if args.spacing else None
        self._composite = CompositeViewerModel(env, pitch=pitch, margin=args.margin)
        self._composite.sync()
        self.model = self._composite.model
        self.data = self._composite.data
        self._shape = grid_shape(env.num_envs)

    def refresh(self, world):
        self._composite.sync()

    def frame_camera(self, viewer, args):
        from collect.view_task import set_initial_camera

        self._composite.frame_camera(viewer)
        if args.camera not in (None, "free"):
            # Model cameras live inside the per-world copies, so the selected
            # world decides which of the identical copies is framed.
            set_initial_camera(
                self._composite,
                viewer,
                self._composite.prefix(args.world_index) + args.camera,
                args.lock_camera,
            )

    def describe(self, args):
        rows, columns = self._shape
        pitch = self._composite.pitch
        return (
            f"Showing all {self._composite.num_envs} worlds on a {rows}x{columns} "
            f"grid, {pitch[0]:.2f}x{pitch[1]:.2f} m apart. Selected world "
            f"{args.world_index}"
        )


def tile_frames(frames):
    """Lay a batch of equally sized RGB frames out on a near-square grid."""
    import numpy as np

    from mjdex.mujoco_warp.composite import grid_shape

    count, height, width, channels = frames.shape
    rows, columns = grid_shape(count)
    padding = np.zeros((rows * columns - count, height, width, channels), frames.dtype)
    grid = np.concatenate((frames, padding)).reshape(
        rows, columns, height, width, channels
    )
    return grid.transpose(0, 2, 1, 3, 4).reshape(
        rows * height, columns * width, channels
    )


def run_headless(env, observation, args):
    import jax
    import numpy as np

    actions = hold_actions(env, observation)
    if args.mode == "hold":
        for _ in range(args.warmup):
            env.step(actions)
        jax.block_until_ready(env.data)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions)
        jax.block_until_ready(env.data)
        duration = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "env": env.env_id,
                    "num_envs": env.num_envs,
                    "steps": args.steps,
                    "device": str(env.device),
                    "seconds": duration,
                    "environment_steps_per_second": args.steps
                    * env.num_envs
                    / duration,
                    "control_timestep": env._env.control_timestep,
                    "physics_substeps": env._env._n_steps,
                },
                indent=2,
            )
        )
    if not np.isfinite(np.asarray(env.data.qpos)).all():
        raise RuntimeError("Non-finite simulation state")
    sync_world(env, args.world_index)
    return args.world_index


def run_viewer(env, observation, args):
    import jax.numpy as jnp
    import mujoco.viewer

    from collect.view_task import print_ee_poses

    world = args.world_index
    scene = (GridScene if args.layout == "grid" else SingleWorldScene)(env, args)
    events = SimpleQueue()
    actions = hold_actions(env, observation)
    paused = args.mode == "static"
    period = env._env.control_timestep / args.speed
    print(f"MjDex Warp: {env.env_id} | {env.num_envs} worlds | {env.device}")
    print(f"{scene.describe(args)}. Mouse: orbit/pan/zoom | Space: pause/resume")
    print(
        "[/]: previous/next selected world | R: reset selected | A: reset all | "
        "P: print pose"
    )
    print(
        "Close the window or press Esc to quit. GPU worlds continue until reset manually."
    )
    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        key_callback=events.put,
        show_left_ui=not args.hide_ui,
        show_right_ui=not args.hide_ui,
    ) as viewer:
        with viewer.lock():
            scene.frame_camera(viewer, args)
        iteration = 0
        while viewer.is_running() and (args.steps is None or iteration < args.steps):
            deadline = time.monotonic() + period
            while not events.empty():
                key = events.get()
                if key == 32:
                    paused = not paused
                    print("Physics paused" if paused else "Physics resumed")
                elif key in (91, 93):  # [ and ]
                    world = (world + (-1 if key == 91 else 1)) % env.num_envs
                    print(f"Selected world {world}")
                elif key in (ord("R"), ord("r"), ord("A"), ord("a")):
                    mask = (
                        jnp.ones(env.num_envs, dtype=bool)
                        if key in (ord("A"), ord("a"))
                        else jnp.arange(env.num_envs) == world
                    )
                    # CPU reset sampling shares the viewer's model/data mirror.
                    with viewer.lock():
                        observation, _ = env.reset(mask=mask)
                    actions = jnp.where(
                        mask[:, None], hold_actions(env, observation), actions
                    )
                    print(
                        "Reset all worlds"
                        if key in (ord("A"), ord("a"))
                        else f"Reset world {world}"
                    )
                elif key in (ord("P"), ord("p")):
                    with viewer.lock():
                        # print_ee_poses reads the single-world CPU mirror, which
                        # the grid layout otherwise leaves untouched.
                        print_ee_poses(sync_world(env, world))
                    print(f"World {world}; observation keys: {sorted(observation)}")
            if not paused:
                observation, *_ = env.step(actions)
            with viewer.lock():
                scene.refresh(world)
            viewer.sync()
            time.sleep(max(0.0, deadline - time.monotonic()))
            iteration += 1
    return world


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    from collect.view_task import build_env_kwargs, print_env_ids, resolve_env_id

    if args.list:
        print_env_ids()
        return
    env_id = resolve_env_id(args.task)
    if args.show_target and not any(task in env_id for task in ("mugrack", "dishrack")):
        raise SystemExit(
            "--show-target is supported only for mugrack and dishrack tasks"
        )

    from mjdex.mujoco_warp import make_warp_env

    env = make_warp_env(
        env_id,
        args.num_envs,
        seed=args.seed,
        device=args.device,
        reset_pool_size=args.reset_pool_size,
        njmax=args.njmax,
        naconmax=args.naconmax,
        **build_env_kwargs(args),
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        world = (run_headless if args.headless else run_viewer)(env, observation, args)
        if args.image:
            import imageio.v3 as iio

            frames = env.render()
            iio.imwrite(args.image, tile_frames(frames))
            print(f"Saved all {len(frames)} worlds to {args.image}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
