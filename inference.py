import os

os.environ["MUJOCO_GL"] = "egl"

import json
import random

import jax
import numpy as np
import wandb
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from agents import agents
from utils.env_utils import create_env
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import (
    CsvLogger,
    get_exp_name,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
)
from utils.race_utils import RACEEnvWrapper, race_evaluate

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "restore_path",
    "exp/single-mugrack-v0/fbc/fbc_sd521512_20260609_193232",
    "Path to the saved experiment directory.",
)
flags.DEFINE_integer("restore_epoch", 4000000, "Checkpoint epoch to restore.")
flags.DEFINE_string(
    "env_name",
    "single-mugrack-v0",
    "Environment name (overrides flags.json if set).",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer("video_episodes", 4, "Number of video episodes.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for video rendering.")
flags.DEFINE_boolean("race", False, "Use RACE (TOPP-RA + Best-of-N) for evaluation.")
flags.DEFINE_integer(
    "num_samples", 16, "Best-of-N chunk samples per step (only used when --race)."
)
flags.DEFINE_integer(
    "open_loop_horizon", 8, "Waypoints to execute before re-querying policy (RACE only)."
)
flags.DEFINE_string(
    "dataset_dir", "data/", "Dataset root dir for obs normalization stats."
)
flags.DEFINE_integer(
    "num_runs", 1, "Number of evaluation runs (for variance estimation)."
)
flags.DEFINE_string("save_dir", "exp/", "Root directory for saving evaluation results.")
flags.DEFINE_string("wandb_mode", "offline", "Wandb mode (online/offline/disabled).")
flags.DEFINE_string("run_group", "eval", "Run group label for wandb grouping.")

config_flags.DEFINE_config_file("agent", "agents/fbc.py", lock_config=False)


def main(_):
    assert FLAGS.restore_path is not None, "Pass --restore_path"
    assert FLAGS.restore_epoch is not None, "Pass --restore_epoch"

    with open(os.path.join(FLAGS.restore_path, "agent_config.json")) as f:
        config = ConfigDict(json.load(f))

    env_name = FLAGS.env_name
    if env_name is None:
        with open(os.path.join(FLAGS.restore_path, "flags.json")) as f:
            env_name = json.load(f)["env_name"]

    # Initial seed — each run derives its own seed from this sequence.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    mode = "race" if FLAGS.race else "standard"
    exp_name = get_exp_name(f"{config['agent_name']}_{mode}", seed=FLAGS.seed)
    wandb_group = f"{FLAGS.run_group}/{env_name}/{config['agent_name']}/{mode}"

    save_dir = os.path.join(
        FLAGS.save_dir, "eval", env_name, config["agent_name"], exp_name
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    eval_logger = CsvLogger(os.path.join(save_dir, "eval.csv"))

    setup_wandb(
        project="mjdex",
        group=wandb_group,
        name=exp_name,
        mode=FLAGS.wandb_mode,
        config={
            **get_flag_dict(),
            **config.to_dict(),
        },
    )

    # Load eval env with obs/action normalization from training dataset.
    _, eval_env, _, _ = create_env(
        env_name, dataset_dir=os.path.join(FLAGS.dataset_dir, env_name)
    )

    if FLAGS.race:
        eval_env = RACEEnvWrapper(eval_env)

    obs_dim = eval_env.observation_space.shape[0]
    horizon_steps = config.get("horizon_steps", config.get("inference_steps"))
    action_dim = eval_env.action_space.shape[0]

    ex_transition = {
        "observations": np.zeros((2, obs_dim), dtype=np.float32),
        "actions": np.zeros((2, horizon_steps, action_dim), dtype=np.float32),
    }

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed=FLAGS.seed,
        ex_transition=ex_transition,
        config=config,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    mode_label = "RACE" if FLAGS.race else "Standard"
    all_stats = {}

    for run_idx in range(FLAGS.num_runs):
        # Each run gets a reproducibly different random state.
        run_seed = np.random.randint(0, 2**32)
        random.seed(run_seed)
        np.random.seed(run_seed)

        if FLAGS.race:
            stats, _, renders = race_evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                num_samples=FLAGS.num_samples,
                open_loop_horizon=FLAGS.open_loop_horizon,
            )
        else:
            stats, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )

        run_metrics = {f"evaluation/{k}": v for k, v in stats.items()}
        if FLAGS.video_episodes > 0 and renders:
            run_metrics["video"] = get_wandb_video(renders=renders)
        wandb.log(run_metrics, step=run_idx)
        eval_logger.log(run_metrics, step=run_idx)

        for k, v in stats.items():
            all_stats.setdefault(k, []).append(v)

        print(
            f"\n=== {mode_label} Evaluation Results — Run {run_idx + 1}/{FLAGS.num_runs} ({env_name}) ==="
        )
        for k, v in sorted(stats.items()):
            print(f"  {k}: {v:.4f}")

    if FLAGS.num_runs > 1:
        mean_metrics = {
            f"evaluation_mean/{k}": np.mean(v) for k, v in all_stats.items()
        }
        wandb.log(mean_metrics)
        eval_logger.log(mean_metrics, step=FLAGS.num_runs)
        print(f"\n=== {mode_label} Aggregate ({FLAGS.num_runs} runs) ===")
        for k, v in sorted(mean_metrics.items()):
            print(f"  {k}: {v:.4f}")

    eval_logger.close()
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
