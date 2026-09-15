# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert labeled F1 peg-insertion MCAP recordings to an RLinf demo buffer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.envs.realworld.f1.f1_robot_env import center_crop_and_resize_rgb
from rlinf.envs.realworld.f1.tasks.peg_reward import (
    PegRewardConfig,
    peg_transition_reward,
)
from toolkits.f1.peg_demo_data import (
    AlignedEpisode,
    ConversionError,
    action_scale_statistics,
    align_episode,
    find_sustained_release,
    interpolate_poses,
    read_episode_mcap,
)

DEFAULT_POLICY_IMAGE_SHAPE = (128, 128)


@dataclass(frozen=True)
class PreparedEpisode:
    """An accepted source episode and its aligned transitions."""

    episode_id: str
    mcap_path: Path
    release_ns: int
    aligned: AlignedEpisode
    diagnostics: dict[str, Any]
    terminal_pose_m_deg: np.ndarray


def build_trajectory(
    aligned: AlignedEpisode,
    *,
    position_scale_m: float,
    orientation_scale_deg: float,
    reward_config: PegRewardConfig,
    policy_image_shape: tuple[int, int] = DEFAULT_POLICY_IMAGE_SHAPE,
) -> Trajectory:
    """Build one batch-size-one RLinf trajectory from aligned physical data."""

    if position_scale_m <= 0.0 or orientation_scale_deg <= 0.0:
        raise ValueError("action scales must be positive")
    scales = np.asarray(
        [position_scale_m] * 3 + [orientation_scale_deg] * 3,
        dtype=np.float64,
    )
    normalized = np.clip(aligned.physical_actions / scales, -1.0, 1.0)
    actions = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(1)
    rewards = torch.tensor(
        [
            peg_transition_reward(
                previous,
                current,
                reward_config,
                terminal_success=index == len(aligned.physical_actions) - 1,
            )
            for index, (previous, current) in enumerate(
                zip(
                    aligned.curr_poses_m_deg,
                    aligned.next_poses_m_deg,
                    strict=True,
                )
            )
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    trajectory_length = len(actions)
    if trajectory_length == 0:
        raise ConversionError("aligned episode contains no transitions")
    dones = torch.zeros((trajectory_length, 1), dtype=torch.bool)
    terminations = torch.zeros_like(dones)
    truncations = torch.zeros_like(dones)
    dones[-1, 0] = True
    terminations[-1, 0] = True

    def process_images(images: np.ndarray) -> torch.Tensor:
        processed = np.stack(
            [center_crop_and_resize_rgb(image, policy_image_shape) for image in images]
        )
        return torch.from_numpy(processed).unsqueeze(1)

    curr_states = torch.from_numpy(
        aligned.curr_poses_m_deg.astype(np.float32)
    ).unsqueeze(1)
    next_states = torch.from_numpy(
        aligned.next_poses_m_deg.astype(np.float32)
    ).unsqueeze(1)
    return Trajectory(
        max_episode_length=trajectory_length,
        model_weights_id="offline-f1-demo",
        actions=actions,
        intervene_flags=torch.ones_like(actions, dtype=torch.bool),
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        forward_inputs={"action": actions.clone()},
        curr_obs={
            "states": curr_states,
            "main_images": process_images(aligned.curr_images),
        },
        next_obs={
            "states": next_states,
            "main_images": process_images(aligned.next_images),
        },
    )


def _load_success_episode_ids(dataset: Path) -> list[str]:
    manifest = dataset / "episode_outcomes.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    episode_ids = [row["episode"] for row in rows if row["outcome"] == "1"]
    if not episode_ids:
        raise ConversionError("outcome manifest contains no successful episodes")
    if len(episode_ids) != len(set(episode_ids)):
        raise ConversionError("outcome manifest contains duplicate episode IDs")
    return episode_ids


def _first_release_command_ns(timestamps: np.ndarray, values: np.ndarray) -> int | None:
    closed_seen = False
    for timestamp, value in zip(timestamps, values, strict=True):
        closed_seen = closed_seen or value >= 80.0
        if closed_seen and value <= 20.0:
            return int(timestamp)
    return None


def prepare_dataset(dataset: Path) -> list[PreparedEpisode]:
    """Parse, truncate, and align every human-success episode."""

    prepared: list[PreparedEpisode] = []
    for episode_id in _load_success_episode_ids(dataset):
        paths = sorted((dataset / episode_id).glob("*.mcap"))
        if len(paths) != 1:
            raise ConversionError(
                f"{episode_id} must contain exactly one MCAP file, found {len(paths)}"
            )
        streams = read_episode_mcap(paths[0])
        release_ns = find_sustained_release(
            streams.gripper_timestamps_ns,
            streams.gripper_values,
        )
        aligned = align_episode(streams, release_ns=release_ns, include_images=False)
        terminal_pose = interpolate_poses(
            streams.tcp_timestamps_ns,
            streams.tcp_poses_m_deg,
            np.asarray([release_ns], dtype=np.int64),
        )[0]
        command_ns = _first_release_command_ns(
            streams.gripper_command_timestamps_ns,
            streams.gripper_command_values,
        )
        release_index = int(
            np.searchsorted(streams.gripper_timestamps_ns, release_ns, side="left")
        )
        later_closed = np.flatnonzero(streams.gripper_values[release_index:] > 20.0)
        release_end_index = (
            release_index + int(later_closed[0])
            if len(later_closed)
            else len(streams.gripper_values) - 1
        )
        diagnostics = {
            "episode": episode_id,
            "mcap": paths[0].name,
            "release_time_ns": release_ns,
            "num_transitions": int(len(aligned.physical_actions)),
            "retained_s": float((release_ns - streams.tcp_timestamps_ns[0]) / 1e9),
            "removed_s": float((streams.tcp_timestamps_ns[-1] - release_ns) / 1e9),
            "release_hold_s": float(
                (streams.gripper_timestamps_ns[release_end_index] - release_ns) / 1e9
            ),
            "command_to_state_delay_s": (
                None if command_ns is None else float((release_ns - command_ns) / 1e9)
            ),
            "terminal_pose_m_deg": terminal_pose.tolist(),
        }
        prepared.append(
            PreparedEpisode(
                episode_id=episode_id,
                mcap_path=paths[0],
                release_ns=release_ns,
                aligned=aligned,
                diagnostics=diagnostics,
                terminal_pose_m_deg=terminal_pose,
            )
        )
    return prepared


def _reward_config(target_pose: np.ndarray) -> PegRewardConfig:
    return PegRewardConfig(
        target_tcp_pose_m_deg=tuple(float(value) for value in target_pose),
        position_reward_scale_m=0.02,
        orientation_reward_scale_deg=10.0,
        position_weight=0.7,
        orientation_weight=0.3,
        step_penalty=0.01,
        position_tolerance_m=0.003,
        orientation_tolerance_deg=3.0,
        success_bonus=5.0,
    )


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_diagnostics(path: Path, episodes: list[PreparedEpisode]) -> None:
    rows = [episode.diagnostics for episode in episodes]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_published_buffer(path: Path, expected_size: int) -> dict[str, Any]:
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=max(expected_size, 1),
        sample_window_size=max(expected_size, 1),
        auto_save=False,
    )
    try:
        buffer.load_checkpoint(str(path))
        if buffer.size != expected_size:
            raise ConversionError(
                f"published buffer has {buffer.size} trajectories, expected {expected_size}"
            )
        sample = buffer.sample(min(64, buffer.total_samples))
        if not sample or sample["actions"].shape[-1] != 6:
            raise ConversionError("published buffer cannot sample 6D actions")
        if not torch.all(torch.isfinite(sample["rewards"])):
            raise ConversionError("published buffer sampled non-finite rewards")
        if torch.any(torch.abs(sample["actions"]) > 1.0):
            raise ConversionError("published buffer sampled out-of-range actions")
        return {
            "num_trajectories": buffer.size,
            "total_samples": buffer.total_samples,
            "sampled_transitions": int(sample["actions"].shape[0]),
            "sample_keys": sorted(sample),
        }
    finally:
        buffer.close()


def convert_dataset(
    dataset: Path,
    output: Path,
    *,
    analysis_only: bool = False,
) -> dict[str, Any]:
    """Analyze a labeled dataset and optionally publish a replay buffer."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    prepared = prepare_dataset(dataset)
    scales = action_scale_statistics(
        [episode.aligned.physical_actions for episode in prepared]
    )
    terminal_poses = np.stack([episode.terminal_pose_m_deg for episode in prepared])
    target_pose = np.median(terminal_poses, axis=0)
    summary = {
        "dataset": str(dataset.resolve()),
        "num_success_episodes": len(prepared),
        "total_transitions": int(
            sum(len(episode.aligned.physical_actions) for episode in prepared)
        ),
        "policy_hz": 10.0,
        "policy_image_shape": list(DEFAULT_POLICY_IMAGE_SHAPE),
        "target_tcp_pose_m_deg": target_pose.tolist(),
        "position_scale_m": scales["position_scale_m"],
        "orientation_scale_deg": scales["orientation_scale_deg"],
        "source_episodes": [episode.episode_id for episode in prepared],
    }

    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        _write_json(temporary / "conversion_manifest.json", summary)
        _write_json(temporary / "scale_statistics.json", scales)
        _write_diagnostics(temporary / "episode_diagnostics.csv", prepared)
        if not analysis_only:
            from rlinf.data.replay_buffer import TrajectoryReplayBuffer

            reward_config = _reward_config(target_pose)
            buffer = TrajectoryReplayBuffer(
                seed=1234,
                enable_cache=False,
                auto_save=True,
                auto_save_path=str(temporary),
                trajectory_format="pt",
            )
            try:
                for episode in prepared:
                    streams = read_episode_mcap(episode.mcap_path)
                    aligned = align_episode(
                        streams,
                        release_ns=episode.release_ns,
                        include_images=True,
                    )
                    trajectory = build_trajectory(
                        aligned,
                        position_scale_m=scales["position_scale_m"],
                        orientation_scale_deg=scales["orientation_scale_deg"],
                        reward_config=reward_config,
                    )
                    buffer.add_trajectories([trajectory])
            finally:
                buffer.close()
            validation = _validate_published_buffer(temporary, len(prepared))
            _write_json(temporary / "validation.json", validation)
        hashes = {
            item.name: _file_sha256(item)
            for item in sorted(temporary.iterdir())
            if item.is_file()
        }
        _write_json(temporary / "artifact_hashes.json", hashes)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the F1 peg demonstration converter."""

    args = _parse_args()
    summary = convert_dataset(
        args.dataset,
        args.output,
        analysis_only=args.analysis_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
