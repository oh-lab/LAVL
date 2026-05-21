from collections import defaultdict
import os

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
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
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
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
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

def visualize(agent, epoch, env, save_dir):
    goals = []
    start_states = []
    for i in range(5):
        observation, info = env.reset(options=dict(task_id=i+1, render_goal=False))
        goal = info.get('goal')
        goals.append(goal)
        start_states.append(observation)
    env = env.unwrapped

    def xy_to_ij_numpy(xy):
        maze_unit = env._maze_unit
        i = jnp.floor((xy[1] + env._offset_y + 0.5 * maze_unit) / maze_unit).astype(int)
        j = jnp.floor((xy[0] + env._offset_x + 0.5 * maze_unit) / maze_unit).astype(int)
        return i, j

    if 'medium' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([7, 7])
    elif 'large' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([8, 11])
    elif 'giant' in env._maze_type:
        xmin, ymin = env.ij_to_xy([0, 0])
        xmax, ymax = env.ij_to_xy([11, 15])
    else:
        raise NotImplementedError(f'Unknown maze type {env._maze_type}')

    fig, axes = plt.subplots(1, 5, figsize=(50, 10))

    for j in range(5):
        goal = goals[j]
        observations_list = []
        value_list = []
        dist_list = []

        xgrid = np.linspace(xmin, xmax, 200, endpoint=False)
        ygrid = np.linspace(ymin, ymax, 200, endpoint=False)

        obs_config = start_states[j][2:]  # keep the configuration part of the observation fixed

        X, Y = np.meshgrid(xgrid, ygrid)
        observations = np.stack([X.ravel(), Y.ravel()], axis=-1)
        observations = np.concatenate([observations, np.repeat(obs_config[None, :], observations.shape[0], axis=0)], axis=-1)
        B = 1000

        for i in range(200*200//B):
            obs = observations[i * B:(i + 1) * B]
            values = agent.compute_value(obs, np.repeat(goal[None, :], obs.shape[0], axis=0))
            dist = np.linalg.norm(obs[:, :2] - goal[None, :2], axis=-1)
            observations_list.append(obs)
            value_list.append(values)
            dist_list.append(dist)

        observations = jnp.concatenate(observations_list, axis=0)
        values = jnp.concatenate(value_list, axis=0).flatten()
        dists = jnp.concatenate(dist_list, axis=0).flatten()

        x, y = observations[:, 0], observations[:, 1]
        xi, yi = xy_to_ij_numpy([x, y])
        is_obstacle = env.maze_map[xi, yi] == 1
        values = values.at[is_obstacle].set(1000.0)

        filter_idx = values < 999.0

        observations = observations[filter_idx]
        values = values[filter_idx]
        dists = dists[filter_idx]

        vmin = values.min()
        vmax = values.max()
        print(f"goal {j+1} value range: {vmin} to {vmax}")

        # Value function scatter plot
        ax = axes[j]
        sc = ax.scatter(observations[:, 0], observations[:, 1], c=values, cmap='viridis', s=1.0)
        ax.set_title(f'Goal {j+1}')
        plt.colorbar(sc, ax=ax, label='Value')

    fig.suptitle(f'Value Function, epoch {epoch}')
    fig.savefig(os.path.join(save_dir, f"epoch{epoch}_value.png"), dpi=300)
    plt.close(fig)