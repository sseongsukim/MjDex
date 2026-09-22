import os

os.environ["MUJOCO_GL"] = "egl"
import random
import time
import json

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from utils.env_utils import create_env
from utils.evaluation import flatten, evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.datasets import (
    MultistepDataset,
    Dataset,
    ACDataset,
    ReplayBuffer,
    QCDataset,
)
from utils.log_utils import (
    get_exp_name,
    setup_wandb,
    get_flag_dict,
    get_wandb_video,
    CsvLogger,
)

from agents import agents

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "single-mugrack-v0", "Environment (dataset) name.")  #
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("dataset_dir", "data/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", None, "Restore epoch.")
flags.DEFINE_string("wandb_mode", "offline", "Wandb mode.")
flags.DEFINE_string("wandb_group", None, "Wandb group override.")
flags.DEFINE_string(
    "wandb_group_format",
    "{run_group}/{env_name}/{agent_name}",
    "Wandb group format. Available fields: run_group, env_name, agent_name.",
)

flags.DEFINE_integer("offline_steps", 2000000, "Number of online steps.")
flags.DEFINE_integer("online_steps", 0, "Number of online steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")

flags.DEFINE_integer("eval_interval", 250000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 1000000, "Save interval.")

flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer("video_episodes", 4, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer(
    "balanced_sampling", 0, "Whether to use balanced sampling for online fine-tuning."
)
config_flags.DEFINE_config_file("agent", "agents/fbc.py", lock_config=False)


def main(_):
    config = FLAGS.agent
    config["train_steps"] = FLAGS.offline_steps + FLAGS.online_steps

    exp_name = get_exp_name(config["agent_name"], seed=FLAGS.seed)

    # Save dir
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.env_name, config["agent_name"], exp_name
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Save parameters
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, "flags.json"), "w") as f:
        json.dump(flag_dict, f)

    config_dict = config.to_dict()
    with open(os.path.join(FLAGS.save_dir, "agent_config.json"), "w") as f:
        json.dump(config_dict, f)

    # Seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Wandb
    setup_wandb(
        project="MjDex",
        group=FLAGS.wandb_group_format,
        name=exp_name,
        config={
            **get_flag_dict(),
            **config.to_dict(),
        },
        mode=FLAGS.wandb_mode,
    )

    # Env
    env, eval_env, train_dataset, val_dataset = create_env(
        env_name=FLAGS.env_name,
        dataset_dir=os.path.join(FLAGS.dataset_dir, FLAGS.env_name),
    )

    dataset_class = {
        "Dataset": Dataset,
        "MultistepDataset": MultistepDataset,
        "ACDataset": ACDataset,
        "QCDataset": QCDataset,
    }
    train_dataset = dataset_class[config["dataset_class"]].create(**train_dataset)
    val_dataset = dataset_class[config["dataset_class"]].create(**val_dataset)

    if FLAGS.online_steps > 0:
        if FLAGS.balanced_sampling:
            # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(
                example_transition, size=FLAGS.buffer_size
            )
        else:
            # Use the training dataset as the replay buffer.
            train_dataset = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
            replay_buffer = train_dataset

    if hasattr(train_dataset, "pred_horizon"):
        train_dataset.pred_horizon = config["horizon_steps"]
        val_dataset.pred_horizon = config["horizon_steps"]

    ex_transition = train_dataset.sample(2)

    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        seed=FLAGS.seed, ex_transition=ex_transition, config=config
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, "train.csv"))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, "eval.csv"))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    for i in tqdm.tqdm(
        range(1, config["train_steps"] + 1), desc="training", smoothing=0.1
    ):
        if i <= FLAGS.offline_steps:
            batch = train_dataset.sample(config["batch_size"])
        else:
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {
                    f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
                }

            step += 1

            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config["batch_size"] // 2)
                replay_batch = replay_buffer.sample(config["batch_size"] // 2)
                batch = {
                    k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                    for k in dataset_batch
                }
            else:
                batch = replay_buffer.sample(config["batch_size"])

        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": float(v) for k, v in update_info.items()}
            train_metrics["time/epoch_time"] = (
                time.time() - last_time
            ) / FLAGS.log_interval
            if val_dataset is not None:
                val_batch = val_dataset.sample(config["batch_size"])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update(
                    {f"val/{k}": float(v) for k, v in val_info.items()}
                )
            train_metrics["time/total_time"] = time.time() - first_time
            train_metrics["train_step"] = i
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f"evaluation/{k}"] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics["video"] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == "__main__":
    app.run(main)
