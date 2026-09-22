from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if action_dim is None:
        action_dim = config["action_dim"]

    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []

        action_queue = []

        try:
            inference_step = agent.config["inference_steps"]
        except KeyError:
            inference_step = agent.config["horizon_steps"]

        total_rewards = 0.0
        while not done:
            if len(action_queue) == 0 or step % inference_step == 0:
                action_queue = []
                action = actor_fn(observations=observation)
                action = np.array(action).reshape(-1, action_dim)
                for a in action:
                    action_queue.append(a)

            action = action_queue.pop(0)
            action = np.clip(action, -1.0, 1.0)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = truncated
            step += 1

            total_rewards += reward

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            stats["total_rewards"].append(total_rewards)
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def _warp_rollout(
    actor_fn,
    env,
    inference_steps,
    action_dim,
    seed,
    max_steps,
    video_frame_skip=0,
    video_size=(320, 240),
    step_callback=None,
):
    """Roll one batch of episodes out to truncation and return per-world results.

    Worlds never auto-reset and all of them share the step counter, so a single
    rollout is exactly ``env.num_envs`` episodes that start and end together.
    """
    observation, _ = env.reset(seed=seed)
    num_envs = env.num_envs
    total_rewards = np.zeros(num_envs, dtype=np.float32)
    succeeded = np.zeros(num_envs, dtype=bool)
    reward = np.zeros(num_envs, dtype=np.float32)
    info = {}
    frames = []
    chunk = None
    cursor = 0

    for step in range(max_steps):
        # Chunk boundaries are the same in every world, so one shared chunk
        # covers the batch; the policy is queried on all worlds at once.
        if chunk is None or cursor >= chunk.shape[1] or step % inference_steps == 0:
            chunk = np.asarray(actor_fn(observations=observation))
            chunk = chunk.reshape(num_envs, -1, action_dim)
            cursor = 0
        action = np.clip(chunk[:, cursor], -1.0, 1.0)
        cursor += 1

        observation, reward, terminated, truncated, info = env.step(action)
        succeeded |= terminated
        total_rewards += reward
        done = bool(truncated.all())

        if step_callback is not None:
            step_callback()

        if video_frame_skip and ((step + 1) % video_frame_skip == 0 or done):
            width, height = video_size
            frames.append(env.render(width=width, height=height))
        if done:
            break

    lengths = np.asarray(info["elapsed_steps"], dtype=np.float32)
    results = {
        "total_rewards": total_rewards,
        "final_reward": np.asarray(reward, dtype=np.float32),
        "episode_length": lengths,
        "success": succeeded.astype(np.float32),
    }
    return results, frames


def evaluate_warp(
    agent,
    env,
    config,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    action_dim=None,
    video_size=(320, 240),
    seed=None,
    step_callback=None,
):
    """Evaluate the agent on a batched MuJoCo Warp environment.

    The batched counterpart of :func:`evaluate`. ``env`` is a
    ``WarpNormalizeEnv``: one rollout runs ``env.num_envs`` episodes in lockstep,
    so ``ceil(num_eval_episodes / env.num_envs)`` rollouts cover the request and
    any surplus episodes are dropped. As in :func:`evaluate`, an episode runs to
    truncation even after the task reports success, and the video episodes are
    an extra rollout that is excluded from the statistics.

    Two differences from :func:`evaluate` are worth knowing. Statistics are
    computed here rather than read out of ``ActionNormalizeEnv``'s info dict, and
    they add ``success`` — whether the task terminated successfully at any point,
    which the mug-rack tasks otherwise only signal through the reward. And no
    trajectories are collected, so the second return value is always empty.

    Returns:
        A tuple of (statistics, trajectories, rendered videos).

    Args:
        step_callback: Called with no arguments after every batched step, for
            live views such as ``BatchViewer.sync``. It paces the rollout, so
            leave it unset for throughput runs.
    """
    if action_dim is None:
        action_dim = config["action_dim"]

    rng_seed = int(np.random.randint(0, 2**30)) if seed is None else int(seed)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(rng_seed))

    try:
        inference_steps = agent.config["inference_steps"]
    except KeyError:
        inference_steps = agent.config["horizon_steps"]

    max_steps = env.env._max_episode_steps
    rounds = int(np.ceil(num_eval_episodes / env.num_envs))
    stats = defaultdict(list)

    for round_index in trange(rounds, desc="eval"):
        results, _ = _warp_rollout(
            actor_fn,
            env,
            inference_steps,
            action_dim,
            # World i resets with seed + i, so rounds have to step by num_envs
            # or consecutive rounds would replay the same scenes.
            seed=rng_seed + round_index * env.num_envs,
            max_steps=max_steps,
            step_callback=step_callback,
        )
        for key, values in results.items():
            stats[key].extend(values.tolist())

    renders = []
    if num_video_episodes > 0:
        _, frames = _warp_rollout(
            actor_fn,
            env,
            inference_steps,
            action_dim,
            seed=rng_seed + rounds * env.num_envs,
            max_steps=max_steps,
            video_frame_skip=video_frame_skip,
            video_size=video_size,
            step_callback=step_callback,
        )
        if frames:
            video = np.stack(frames)  # (time, world, height, width, channel)
            renders = list(video.transpose(1, 0, 2, 3, 4)[:num_video_episodes])

    stats = {k: float(np.mean(v[:num_eval_episodes])) for k, v in stats.items()}
    return stats, [], renders
