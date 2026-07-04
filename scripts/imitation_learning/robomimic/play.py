# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    num_envs: Number of environments to evaluate in parallel.
    seed: If provided, override the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to evaluate in parallel.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")
parser.add_argument(
    "--record_success_videos",
    action="store_true",
    default=False,
    help="Record table and wrist camera videos only for successful rollouts.",
)
parser.add_argument(
    "--success_video_dir",
    type=str,
    default="logs/robomimic/success_videos",
    help="Directory used for successful rollout videos.",
)
parser.add_argument("--success_video_fps", type=int, default=15, help="Frame rate for successful rollout videos.")
parser.add_argument(
    "--max_success_videos",
    type=int,
    default=0,
    help="Maximum successful rollouts to save. 0 means save every successful counted rollout.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

import fr3_peg_insert.tasks  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


CAMERA_NAMES = ("table_cam", "wrist_cam")


def _get_successes(env, success_term):
    """Check success for both manager-based and direct task environments."""
    if success_term is not None:
        return success_term.func(env, **success_term.params).to(device=env.device, dtype=torch.bool)
    if hasattr(env, "_get_curr_successes"):
        success_threshold = getattr(env.cfg.task, "success_threshold", 0.04)
        return env._get_curr_successes(success_threshold=success_threshold).to(dtype=torch.bool)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _get_action_dim(env) -> int:
    """Return the single-environment action dimension."""
    if hasattr(env, "single_action_space"):
        return env.single_action_space.shape[0]
    if len(env.action_space.shape) > 1:
        return env.action_space.shape[1]
    return env.action_space.shape[0]


def _prepare_obs(obs_dict, env, env_id: int) -> dict:
    """Prepare one environment's policy observation for robomimic inference."""
    obs = {}
    for ob, value in obs_dict["policy"].items():
        obs[ob] = torch.squeeze(value[env_id])

    if hasattr(env.cfg, "image_obs_list"):
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"].keys():
                image = torch.squeeze(obs_dict["policy"][image_name][env_id])
                image = image.permute(2, 0, 1).clone().float()
                if image.max() > 1.0:
                    image = image / 255.0
                image = image.clip(0.0, 1.0)
                obs[image_name] = image
    return obs


def _as_done_mask(value, num_envs: int, device) -> torch.Tensor:
    """Convert gymnasium done outputs to a vectorized boolean mask."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bool).reshape(-1)
    return torch.full((num_envs,), bool(value), dtype=torch.bool, device=device)


def _policy_from_trusted_checkpoint(ckpt_path, device):
    """Load a robomimic policy checkpoint that contains non-tensor metadata.

    Robomimic checkpoints store config and metadata alongside weights. PyTorch
    2.6 changed torch.load's default to weights_only=True, which rejects those
    objects unless the load is explicitly marked as trusted.
    """
    original_torch_load = torch.load

    def torch_load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_trusted
    try:
        return FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=False)[0]
    finally:
        torch.load = original_torch_load


def _to_uint8_frame(image: torch.Tensor) -> np.ndarray:
    """Convert one HWC camera tensor to a CPU uint8 RGB frame."""
    image = image.detach()
    if image.dtype == torch.uint8:
        return image.cpu().numpy().copy()
    return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy().copy()


def _get_camera_frame(env, camera_name: str, env_id: int) -> np.ndarray:
    """Read a raw RGB frame from the requested tiled camera."""
    camera_attr = f"_{camera_name.split('_')[0]}_camera"
    camera = getattr(env, camera_attr)
    return _to_uint8_frame(camera.data.output["rgb"][env_id])


def _write_video_or_frames(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Write frames as mp4 when possible, otherwise write a PNG frame directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames, fps=fps)
        print(f"[INFO] Wrote success video: {output_path}")
        return
    except Exception as exc:
        print(f"[WARN] Could not write mp4 with imageio: {exc}")

    frames_dir = output_path.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        for frame_idx, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"{frame_idx:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Wrote success frames: {frames_dir}")
    except Exception as exc:
        print(f"[WARN] Could not write fallback PNG frames: {exc}")


def _save_success_videos(
    camera_frames: dict[str, list[list[np.ndarray]]],
    batch_index: int,
    batch_results: list[bool],
    counted_rollouts: int,
    remaining_rollouts: int,
    saved_success_count: int,
) -> int:
    """Save camera videos for successful rollouts that count toward the requested total."""
    if not args_cli.record_success_videos:
        return saved_success_count

    for env_id, success in enumerate(batch_results[:remaining_rollouts]):
        if not success:
            continue
        if args_cli.max_success_videos > 0 and saved_success_count >= args_cli.max_success_videos:
            return saved_success_count

        rollout_index = counted_rollouts + env_id
        saved_success_count += 1
        for camera_name in CAMERA_NAMES:
            video_name = (
                f"success_{saved_success_count:04d}_rollout_{rollout_index:04d}_"
                f"batch_{batch_index:03d}_env_{env_id:02d}_{camera_name}.mp4"
            )
            output_path = (
                Path(args_cli.success_video_dir)
                / video_name
            )
            _write_video_or_frames(camera_frames[camera_name][env_id], output_path, args_cli.success_video_fps)

    return saved_success_count


def rollout_batch(policies, env, success_term, horizon, device):
    """Perform one parallel evaluation batch.

    Args:
        policies: One robomimic policy per parallel environment.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        results: Per-environment success results.
    """
    for policy in policies:
        policy.start_episode()

    obs_dict, _ = env.reset()
    num_envs = env.num_envs
    action_dim = _get_action_dim(env)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    camera_frames = None
    if args_cli.record_success_videos:
        camera_frames = {camera_name: [[] for _ in range(num_envs)] for camera_name in CAMERA_NAMES}

    for i in range(horizon):
        if camera_frames is not None:
            for env_id in range(num_envs):
                for camera_name in CAMERA_NAMES:
                    camera_frames[camera_name][env_id].append(_get_camera_frame(env, camera_name, env_id))

        actions = torch.zeros((num_envs, action_dim), device=device)
        for env_id, policy in enumerate(policies):
            obs = _prepare_obs(obs_dict, env, env_id)
            action = policy(obs)
            if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
                action = (
                    (action + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
                ) / 2 + args_cli.norm_factor_min
            actions[env_id] = torch.from_numpy(action).to(device=device).view(action_dim)

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        done_mask = _as_done_mask(terminated, num_envs, env.device) | _as_done_mask(truncated, num_envs, env.device)
        episode_success.logical_or_(_get_successes(env, success_term))

        if torch.all(done_mask):
            print(f"[INFO] Batch finished early at step {i + 1}/{horizon}")
            break

    results = [bool(value) for value in episode_success.detach().cpu().tolist()]
    return results, camera_frames


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    if args_cli.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be greater than 0.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be greater than 0.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Set observations to dictionary mode for Robomimic on manager-based environments.
    if hasattr(env_cfg, "observations"):
        env_cfg.observations.policy.concatenate_terms = False

    success_term = None
    if hasattr(env_cfg, "terminations"):
        # Set termination conditions
        env_cfg.terminations.time_out = None

        # Extract success checking function
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    # Disable recorder on manager-based environments.
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Run policy
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Num envs: {env.num_envs}")
    print(f"[INFO] Target rollouts: {args_cli.num_rollouts}")
    print(f"[INFO] Horizon: {args_cli.horizon}")
    if args_cli.record_success_videos:
        print(f"[INFO] Recording successful rollouts to: {os.path.abspath(args_cli.success_video_dir)}")

    policies = [
        _policy_from_trusted_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
        for _ in range(env.num_envs)
    ]

    results = []
    batch = 0
    saved_success_count = 0
    while len(results) < args_cli.num_rollouts:
        batch += 1
        remaining = args_cli.num_rollouts - len(results)
        print(f"[INFO] Starting batch {batch} ({min(env.num_envs, remaining)} rollouts counted)")
        batch_results, camera_frames = rollout_batch(policies, env, success_term, args_cli.horizon, device)
        saved_success_count = _save_success_videos(
            camera_frames=camera_frames,
            batch_index=batch,
            batch_results=batch_results,
            counted_rollouts=len(results),
            remaining_rollouts=remaining,
            saved_success_count=saved_success_count,
        )
        results.extend(batch_results[:remaining])
        success_rate = results.count(True) / len(results)
        print(
            f"[INFO] Batch {batch}: {batch_results[:remaining]} | "
            f"episodes={len(results)}/{args_cli.num_rollouts} success_rate={success_rate:.2%}\n"
        )

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results):.2%}")
    if args_cli.record_success_videos:
        print(f"Saved successful rollouts: {saved_success_count}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
