# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record robomimic HDF5 demos from an RL-Games low-dimensional teacher.

This script runs the visuomotor FR3 peg-insert task so that camera observations are available, but it reconstructs the
original low-dimensional Direct-v0 observation for an already trained RL-Games teacher. The dataset stores the
visuomotor observations together with teacher actions:

    data/demo_*/obs/proprio
    data/demo_*/obs/table_cam
    data/demo_*/obs/wrist_cam
    data/demo_*/actions
    data/demo_*/rewards
    data/demo_*/dones
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record visuomotor robomimic HDF5 demos from an RL-Games teacher.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Fr3-Peg-Insert-Visuomotor-Direct-v0",
    help="Visuomotor task used for data recording.",
)
parser.add_argument(
    "--teacher_task",
    type=str,
    default="Isaac-Fr3-Peg-Insert-Direct-v0",
    help="Low-dimensional task whose RL-Games config/checkpoint is used as the teacher.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rl_games_cfg_entry_point",
    help="Name of the RL-Games teacher configuration entry point.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the RL-Games teacher checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint is provided, use the last saved teacher checkpoint instead of the best one.",
)
parser.add_argument("--output_file", type=str, default="./datasets/fr3_peg_insert_visuomotor.hdf5")
parser.add_argument("--num_demos", type=int, default=10, help="Number of successful demos to record.")
parser.add_argument("--horizon", type=int, default=400, help="Maximum environment steps per recording attempt.")
parser.add_argument("--max_attempts", type=int, default=0, help="Maximum attempts. Set 0 for no explicit limit.")
parser.add_argument("--num_success_steps", type=int, default=10, help="Consecutive successful steps required.")
parser.add_argument(
    "--success_threshold",
    type=float,
    default=None,
    help="Override the task success threshold. Uses env config value when omitted.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and teacher.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This recorder supports 1.")
parser.add_argument("--stochastic", action="store_true", help="Use stochastic teacher actions.")
parser.add_argument(
    "--record_failed",
    action="store_true",
    help="Also write failed attempts to HDF5. By default only successful demos are exported.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_rl.rl_games import RlGamesGpuEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import fr3_peg_insert.tasks  # noqa: F401
from fr3_peg_insert.tasks.direct.fr3_peg_insert import utils
from fr3_peg_insert.tasks.direct.fr3_peg_insert.fr3_peg_insert_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG


def _as_scalar_bool(value: torch.Tensor) -> bool:
    """Convert a one-env boolean tensor to a Python bool."""
    return bool(value.detach().flatten()[0].item())


def _to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a single HWC image to uint8 for robomimic datasets."""
    image = image.detach()
    if image.dtype == torch.uint8:
        return image.clone()
    return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _add_policy_obs_to_episode(episode: EpisodeData, policy_obs: dict[str, torch.Tensor], env_id: int = 0):
    """Add the current visuomotor policy observation to an episode."""
    episode.add("obs/proprio", policy_obs["proprio"][env_id].detach())
    episode.add("obs/table_cam", _to_uint8_image(policy_obs["table_cam"][env_id]))
    episode.add("obs/wrist_cam", _to_uint8_image(policy_obs["wrist_cam"][env_id]))


def _set_last_done(episode: EpisodeData, done_value: bool):
    """Overwrite the last done flag before exporting an episode."""
    if "dones" in episode.data and len(episode.data["dones"]) > 0:
        device = episode.data["dones"][-1].device
        episode.data["dones"][-1] = torch.tensor(done_value, dtype=torch.bool, device=device)


class TeacherObsEnvWrapper:
    """Minimal RL-Games vector environment that feeds teacher observations from a visuomotor DirectRLEnv."""

    def __init__(
        self,
        env: gym.Env,
        teacher_obs_order: list[str],
        teacher_state_order: list[str],
        rl_device: str,
        clip_obs: float,
        clip_actions: float,
        obs_dim: int,
        state_dim: int,
    ):
        self.env = env
        self._teacher_obs_order = teacher_obs_order
        self._teacher_state_order = teacher_state_order
        self._rl_device = rl_device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self._sim_device = env.unwrapped.device
        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self.rlg_num_states = state_dim

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.unwrapped.num_envs

    @property
    def observation_space(self):
        import gym.spaces

        return gym.spaces.Box(-self._clip_obs, self._clip_obs, shape=(self._obs_dim,))

    @property
    def action_space(self):
        import gym.spaces

        return gym.spaces.Box(-self._clip_actions, self._clip_actions, shape=self.unwrapped.single_action_space.shape)

    @property
    def state_space(self):
        import gym.spaces

        return gym.spaces.Box(-self._clip_obs, self._clip_obs, shape=(self._state_dim,))

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    def reset(self):
        self.env.reset()
        return self.get_teacher_obs()

    def step(self, actions: torch.Tensor):
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        _, reward, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        return self.get_teacher_obs(), reward.to(self._rl_device), dones.to(self._rl_device), extras

    def close(self):
        self.env.close()

    def get_teacher_obs(self):
        obs_dict, state_dict = self.unwrapped._get_obs_state_dict()
        teacher_obs = utils.collapse_obs_dict(obs_dict, self._teacher_obs_order + ["prev_actions"])
        teacher_state = utils.collapse_obs_dict(state_dict, self._teacher_state_order + ["prev_actions"])
        teacher_obs = torch.clamp(teacher_obs.to(self._rl_device), -self._clip_obs, self._clip_obs)
        teacher_state = torch.clamp(teacher_state.to(self._rl_device), -self._clip_obs, self._clip_obs)
        return {"obs": teacher_obs, "states": teacher_state}


def _resolve_checkpoint(agent_cfg: dict) -> str:
    """Resolve the RL-Games checkpoint path using the same convention as play.py."""
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        return get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    return retrieve_file_path(args_cli.checkpoint)


def _create_output_writer() -> HDF5DatasetFileHandler:
    """Create the output HDF5 writer."""
    output_dir = os.path.dirname(args_cli.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = HDF5DatasetFileHandler()
    writer.create(args_cli.output_file, env_name=args_cli.task.split(":")[-1])
    writer.add_env_args(
        {
            "teacher_env_name": args_cli.teacher_task.split(":")[-1],
            "teacher_checkpoint": args_cli.checkpoint or "",
        }
    )
    return writer


@hydra_task_config(args_cli.teacher_task, args_cli.agent)
def main(teacher_env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Record successful visuomotor demonstrations from an RL-Games teacher."""
    if args_cli.num_envs != 1:
        raise ValueError("This HDF5 recorder currently supports --num_envs 1 only.")
    if args_cli.num_demos <= 0:
        raise ValueError("--num_demos must be greater than 0.")
    if args_cli.horizon <= 0:
        raise ValueError("--horizon must be greater than 0.")
    if args_cli.num_success_steps <= 0:
        raise ValueError("--num_success_steps must be greater than 0.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = _resolve_checkpoint(agent_cfg)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    isaac_env = env.unwrapped
    teacher_obs_order = list(teacher_env_cfg.obs_order)
    teacher_state_order = list(teacher_env_cfg.state_order)
    obs_dim = sum(OBS_DIM_CFG[name] for name in teacher_obs_order) + teacher_env_cfg.action_space
    state_dim = sum(STATE_DIM_CFG[name] for name in teacher_state_order) + teacher_env_cfg.action_space

    teacher_wrapper = TeacherObsEnvWrapper(
        env=env,
        teacher_obs_order=teacher_obs_order,
        teacher_state_order=teacher_state_order,
        rl_device=rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
        obs_dim=obs_dim,
        state_dim=state_dim,
    )

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: teacher_wrapper}
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = isaac_env.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    writer = _create_output_writer()
    success_threshold = (
        args_cli.success_threshold if args_cli.success_threshold is not None else isaac_env.cfg_task.success_threshold
    )
    max_attempts = args_cli.max_attempts if args_cli.max_attempts > 0 else math.inf

    print(f"[INFO] Recording task: {args_cli.task}")
    print(f"[INFO] Teacher task: {args_cli.teacher_task}")
    print(f"[INFO] Teacher checkpoint: {resume_path}")
    print(f"[INFO] Output file: {args_cli.output_file}")
    print(f"[INFO] Target successful demos: {args_cli.num_demos}")
    print(f"[INFO] Success threshold: {success_threshold}")
    print(f"[INFO] Consecutive success steps required: {args_cli.num_success_steps}")
    print(f"[INFO] Deterministic teacher: {not args_cli.stochastic}")

    recorded = 0
    attempts = 0
    try:
        while simulation_app.is_running() and recorded < args_cli.num_demos and attempts < max_attempts:
            attempts += 1
            episode = EpisodeData()
            episode.seed = env_cfg.seed
            obs_dict, _ = isaac_env.reset()
            teacher_obs = teacher_wrapper.get_teacher_obs()["obs"]
            _ = agent.get_batch_size(teacher_obs, 1)
            if agent.is_rnn:
                agent.init_rnn()

            success_steps = 0
            attempt_success = False
            final_done = False

            for step in range(args_cli.horizon):
                policy_obs = obs_dict["policy"]
                with torch.no_grad():
                    teacher_obs_torch = agent.obs_to_torch(teacher_obs)
                    action = agent.get_action(teacher_obs_torch, is_deterministic=not args_cli.stochastic)
                sim_action = action.detach().clone().to(isaac_env.device)

                _add_policy_obs_to_episode(episode, policy_obs)
                episode.add("actions", sim_action[0])

                obs_dict, reward, terminated, truncated, _ = isaac_env.step(sim_action)
                done = terminated | truncated
                curr_success = isaac_env._get_curr_successes(success_threshold=success_threshold)
                success_steps = success_steps + 1 if _as_scalar_bool(curr_success) else 0

                final_done = _as_scalar_bool(done)
                episode.add("rewards", reward[0].detach())
                episode.add("dones", torch.tensor(final_done, dtype=torch.bool, device=isaac_env.device))

                teacher_obs = teacher_wrapper.get_teacher_obs()["obs"]

                if agent.is_rnn and agent.states is not None and final_done:
                    for state in agent.states:
                        state[:, 0:1, :] = 0.0

                if success_steps >= args_cli.num_success_steps:
                    attempt_success = True
                    _set_last_done(episode, True)
                    break
                if final_done:
                    break

            if attempt_success or args_cli.record_failed:
                episode.success = attempt_success
                episode.pre_export()
                writer.write_episode(episode)
                writer.flush()
                if attempt_success:
                    recorded += 1
                    print(
                        f"[REC] demo={recorded}/{args_cli.num_demos} "
                        f"attempt={attempts} steps={step + 1} success=True"
                    )
                else:
                    print(f"[REC] failed attempt exported attempt={attempts} steps={step + 1}")
            else:
                print(f"[SKIP] attempt={attempts} steps={step + 1} success=False done={final_done}")

        print("[RESULT]")
        print(f"  Recorded successful demos: {recorded}")
        print(f"  Attempts: {attempts}")
        print(f"  Output file: {args_cli.output_file}")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
