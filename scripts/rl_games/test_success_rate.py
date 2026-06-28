# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate checkpoint success rate for an RL-Games policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the success rate of an RL-Games checkpoint.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=128, help="Number of episodes to evaluate.")
parser.add_argument(
    "--success_threshold",
    type=float,
    default=None,
    help="Override env task success threshold. Uses env config value when omitted.",
)
parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic actions.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_peg_insert.tasks  # noqa: F401


def _as_done_indices(dones: torch.Tensor) -> torch.Tensor:
    """Return done environment indices from an RL-Games dones tensor."""
    if dones.dtype == torch.bool:
        return dones.nonzero(as_tuple=False).squeeze(-1)
    return (dones > 0).nonzero(as_tuple=False).squeeze(-1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Run evaluation episodes and print success statistics."""
    if args_cli.num_episodes <= 0:
        raise ValueError("--num_episodes must be greater than 0.")

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        resume_path = get_checkpoint_path(
            log_root_path, run_dir, f"{agent_cfg['params']['config']['name']}.pth", other_dirs=["nn"]
        )
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    isaac_env = env.unwrapped
    num_envs = isaac_env.num_envs
    success_threshold = (
        args_cli.success_threshold if args_cli.success_threshold is not None else isaac_env.cfg_task.success_threshold
    )

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {resume_path}")
    print(f"[INFO] Num envs: {num_envs}")
    print(f"[INFO] Target episodes: {args_cli.num_episodes}")
    print(f"[INFO] Success threshold: {success_threshold}")
    print(f"[INFO] Deterministic policy: {not args_cli.stochastic}")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    dt = isaac_env.step_dt
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=isaac_env.device)
    successes = 0
    episodes = 0
    total_steps = 0

    while simulation_app.is_running() and episodes < args_cli.num_episodes:
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=not args_cli.stochastic)
            obs, _, dones, _ = env.step(actions)

            curr_successes = isaac_env._get_curr_successes(success_threshold=success_threshold)
            episode_success.logical_or_(curr_successes)

            done_ids = _as_done_indices(dones)
            if len(done_ids) > 0:
                completed = min(len(done_ids), args_cli.num_episodes - episodes)
                completed_ids = done_ids[:completed]
                successes += int(episode_success[completed_ids].sum().item())
                episodes += completed

                if agent.is_rnn and agent.states is not None:
                    for state in agent.states:
                        state[:, done_ids, :] = 0.0

                episode_success[done_ids] = False

                success_rate = successes / episodes if episodes > 0 else 0.0
                print(
                    f"[EVAL] episodes={episodes}/{args_cli.num_episodes} "
                    f"successes={successes} success_rate={success_rate:.2%}"
                )

        total_steps += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    final_rate = successes / episodes if episodes > 0 else 0.0
    print("[RESULT]")
    print(f"  Episodes: {episodes}")
    print(f"  Successes: {successes}")
    print(f"  Success rate: {final_rate:.2%}")
    print(f"  Sim steps: {total_steps}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
