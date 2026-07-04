# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export every demo in a robomimic HDF5 dataset as per-demo videos."""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


CAMERA_KEYS = ("table_cam", "wrist_cam")


def _demo_sort_key(name: str) -> tuple[int, int | str]:
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return (0, int(suffix))
    return (1, name)


def sorted_demo_keys(data_group) -> list[str]:
    """Return demo keys in numeric order when names follow demo_N."""
    return sorted(data_group.keys(), key=_demo_sort_key)


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """Convert HWC RGB frames from float or uint8 into uint8 RGB."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T, H, W, 3], got {frames.shape}.")
    if frames.dtype == np.uint8:
        return frames
    return (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _read_camera_frames(demo_group, camera: str, start: int, max_frames: int, stride: int) -> np.ndarray:
    obs_group = demo_group.get("obs")
    if obs_group is None:
        raise KeyError("Demo has no /obs group.")
    if camera not in obs_group:
        raise KeyError(f"Demo has no /obs/{camera} dataset.")

    stop = None if max_frames <= 0 else start + max_frames * stride
    frames = obs_group[camera][start:stop:stride]
    frames = _as_uint8_rgb(frames)
    if len(frames) == 0:
        raise ValueError(f"No frames selected for {demo_group.name}/{camera}.")
    return frames


def build_frames(demo_group, camera: str, start: int, max_frames: int, stride: int) -> np.ndarray:
    """Build video frames for one camera or both cameras side by side."""
    if start < 0:
        raise ValueError("--start must be greater than or equal to 0.")

    if camera in CAMERA_KEYS:
        return _read_camera_frames(demo_group, camera, start, max_frames, stride)

    if camera != "both":
        raise ValueError(f"Unsupported camera: {camera}")

    table_frames = _read_camera_frames(demo_group, "table_cam", start, max_frames, stride)
    wrist_frames = _read_camera_frames(demo_group, "wrist_cam", start, max_frames, stride)
    num_frames = min(len(table_frames), len(wrist_frames))
    return np.concatenate((table_frames[:num_frames], wrist_frames[:num_frames]), axis=2)


def write_video(frames: np.ndarray, output_path: str, fps: int) -> bool:
    """Write frames to a video file. Return False when no encoder is available."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, frames, fps=fps)
        return True
    except Exception as exc:
        print(f"[WARN] Could not write video with imageio: {exc}")
        return False


def write_frames(frames: np.ndarray, output_dir: str) -> None:
    """Write video frames as PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import cv2

        for frame_idx, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{frame_idx:04d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return
    except Exception as exc:
        print(f"[WARN] Could not write PNG frames with cv2: {exc}")

    import imageio.v2 as imageio

    for frame_idx, frame in enumerate(frames):
        path = os.path.join(output_dir, f"{frame_idx:04d}.png")
        imageio.imwrite(path, frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all HDF5 camera demos to videos.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--output_dir", type=str, default="videos", help="Directory for exported videos.")
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="{demo}_{camera}.mp4",
        help="Output filename pattern. Available fields: {demo}, {demo_index}, {camera}.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="both",
        choices=("table_cam", "wrist_cam", "both"),
        help="Camera stream to export. 'both' places table and wrist views side by side.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output video frame rate.")
    parser.add_argument("--start", type=int, default=0, help="First frame index.")
    parser.add_argument("--max_frames", type=int, default=0, help="Maximum frames per demo. 0 means all.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--start_demo_index", type=int, default=0, help="First sorted demo index to export.")
    parser.add_argument("--max_demos", type=int, default=0, help="Maximum number of demos to export. 0 means all.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos that already exist.")
    return parser.parse_args()


def resolve_output_path(output_dir: str, output_pattern: str, demo_key: str, demo_index: int, camera: str) -> str:
    output_name = output_pattern.format(demo=demo_key, demo_index=demo_index, camera=camera)
    if os.path.isabs(output_name):
        return output_name
    return os.path.join(output_dir, output_name)


def selected_demo_keys(data_group, start_demo_index: int, max_demos: int) -> list[tuple[int, str]]:
    demos = sorted_demo_keys(data_group)
    if len(demos) == 0:
        return []
    if start_demo_index < 0 or start_demo_index >= len(demos):
        raise IndexError(f"--start_demo_index {start_demo_index} out of range for {len(demos)} demos.")

    end_index = len(demos) if max_demos <= 0 else min(len(demos), start_demo_index + max_demos)
    return list(enumerate(demos[start_demo_index:end_index], start=start_demo_index))


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0.")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    if args.max_frames < 0:
        raise ValueError("--max_frames must be greater than or equal to 0.")

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.dataset, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"Dataset has no /data group: {args.dataset}")

        demo_items = selected_demo_keys(h5_file["data"], args.start_demo_index, args.max_demos)
        print(f"[INFO] Dataset: {args.dataset}")
        print(f"[INFO] Exporting {len(demo_items)} demos to: {args.output_dir}")

        exported_count = 0
        skipped_count = 0
        for demo_index, demo_key in demo_items:
            output_path = resolve_output_path(args.output_dir, args.output_pattern, demo_key, demo_index, args.camera)
            if args.skip_existing and os.path.exists(output_path):
                print(f"[SKIP] {demo_key}: {output_path}")
                skipped_count += 1
                continue

            demo_group = h5_file["data"][demo_key]
            frames = build_frames(demo_group, args.camera, args.start, args.max_frames, args.stride)
            success = demo_group.attrs.get("success", None)
            num_samples = demo_group.attrs.get("num_samples", len(frames))

            print(
                f"[INFO] {demo_key}: index={demo_index} num_samples={num_samples} "
                f"success={success} frames={frames.shape}"
            )
            if write_video(frames, output_path, args.fps):
                print(f"[INFO] Wrote video: {output_path}")
            else:
                stem, _ = os.path.splitext(output_path)
                frames_dir = stem + "_frames"
                write_frames(frames, frames_dir)
                print("[WARN] No video encoder backend found. Wrote PNG frames instead.")
                print(f"[INFO] Frames directory: {frames_dir}")
            exported_count += 1

    print(f"[INFO] Done. exported={exported_count} skipped={skipped_count}")


if __name__ == "__main__":
    main()
