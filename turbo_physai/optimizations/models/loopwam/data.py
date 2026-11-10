# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0
# Extracted from LoopWAM's LeRobot dataset; adapts HF Column reads and avoids
# materializing columns for the upstream-disabled timestamp check.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal


def dataset_init(
    self,
    repo_id: str,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
    image_transforms: Callable | None = None,
    delta_timestamps: dict[list[float]] | None = None,
    tolerance_s: float = 1e-4,
    revision: str | None = None,
    force_cache_sync: bool = False,
    download_videos: bool = True,
    video_backend: str | None = None,
    video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
    is_compute_episode_stats_image: bool = True,
):
    import torch
    import os
    from pathlib import Path
    import numpy as np
    import packaging.version
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (
        HF_LEROBOT_HOME, CODEBASE_VERSION, LeRobotDatasetMetadata,
        get_safe_default_codec, aggregate_stats, get_episode_data_index,
        check_timestamps_sync, get_delta_indices,
    )
    torch.utils.data.Dataset.__init__(self)
    self.repo_id = repo_id
    self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    self.image_transforms = image_transforms
    self.delta_timestamps = delta_timestamps
    self.episodes = episodes
    self.tolerance_s = tolerance_s
    self.revision = revision if revision else CODEBASE_VERSION
    self.video_backend = video_backend if video_backend else get_safe_default_codec()
    self.video_codec = video_codec
    self.is_compute_episode_stats_image = is_compute_episode_stats_image
    self.delta_indices = None
    self.during_training = True

    # Unused attributes
    self.image_writer = None
    self.episode_buffer = None

    self.root.mkdir(exist_ok=True, parents=True)

    # Load metadata
    self.meta = LeRobotDatasetMetadata(
        self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
    )
    if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
        episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
        self.stats = aggregate_stats(episodes_stats)

    # Load actual data
    try:
        if force_cache_sync:
            raise FileNotFoundError
        skip_file_validation = os.environ.get(
            "FASTWAM_SKIP_EPISODE_FILE_VALIDATION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not skip_file_validation:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
        self.hf_dataset = self.load_hf_dataset()
    except (AssertionError, FileNotFoundError, NotADirectoryError):
        # self.revision = get_safe_version(self.repo_id, self.revision)
        self.revision = CODEBASE_VERSION
        self.download_episodes(download_videos)
        self.hf_dataset = self.load_hf_dataset()

    self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

    # The original timestamp validation is disabled, so do not materialize
    # the full timestamp and episode_index columns on every distributed
    # rank. For this dataset that meant converting ~11.6M rows to Python
    # objects per rank even though the resulting arrays were never used.
    validate_timestamps = os.environ.get(
        "FASTWAM_VALIDATE_TIMESTAMPS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if validate_timestamps:
        timestamp_values = list(self.hf_dataset["timestamp"])
        episode_index_values = list(self.hf_dataset["episode_index"])
        timestamps = (
            torch.stack(timestamp_values).numpy()
            if timestamp_values and torch.is_tensor(timestamp_values[0])
            else np.asarray(timestamp_values)
        )
        episode_indices = (
            torch.stack(episode_index_values).numpy()
            if episode_index_values and torch.is_tensor(episode_index_values[0])
            else np.asarray(episode_index_values)
        )
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

    # Setup delta_indices
    if self.delta_timestamps is not None:
        # check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
        self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)


def _get_query_timestamps(
    self,
    current_ts: float,
    query_indices: dict[str, list[int]] | None = None,
) -> dict[str, list[float]]:
    import torch
    query_timestamps = {}
    for key in self.meta.video_keys:
        if query_indices is not None and key in query_indices:
            timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
            query_timestamps[key] = torch.stack(list(timestamps)).tolist()
        else:
            query_timestamps[key] = [current_ts]

    return query_timestamps


def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
    import torch
    return {
        key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
        for key, q_idx in query_indices.items()
        if key not in self.meta.video_keys
    }


def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
    import torch
    result = {}
    processed_indices = set()
    index_to_selected = {}
    for key, q_idx in query_indices.items():
        if key not in self.meta.video_keys :
            if 'images' in key and not self.during_training:
                continue
            q_idx_tuple = tuple(q_idx)
            if q_idx_tuple not in processed_indices:
                selected_data = self.hf_dataset.select(q_idx)
                index_to_selected[q_idx_tuple] = selected_data
                processed_indices.add(q_idx_tuple)
            else:
                selected_data = index_to_selected[q_idx_tuple]
            result[key] = torch.stack(list(selected_data[key]))
    return result


def get_episode_data(self, episode_id: int) -> dict:
    import torch
    ep_start = self.episode_data_index["from"][episode_id].item()
    ep_end = self.episode_data_index["to"][episode_id].item()
    q_idx = list(range(ep_start, ep_end))
    # selected_data = self.hf_dataset.select(q_idx)
    selected_data = self.hf_dataset[q_idx]
    res_keys = self.meta.features.keys() - set(self.meta.video_keys)
    res = {key : torch.stack(list(selected_data[key])) for key in res_keys}
    return res


def load_hf_dataset(self) -> datasets.Dataset:
    """hf_dataset contains all the observations, states, actions, rewards, etc."""
    # Hugging Face's FileLock removes its lock file on release. On shared
    # storage, many distributed ranks opening the same cached parquet
    # builder can race between open() and fstat(), raising
    # FileNotFoundError. Let one rank build/open the cache behind a stable
    # flock, publish its Arrow files atomically, then let all other ranks
    # memory-map those files directly without entering the HF builder.
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import datasets
    from datasets import concatenate_datasets, load_dataset
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import hf_transform_to_torch
    cache_dir = Path(os.environ.get("HF_DATASETS_CACHE", "/tmp"))
    lock_path = Path(
        os.environ.get(
            "FASTWAM_HF_LOAD_LOCK_PATH",
            cache_dir / ".fastwam_load_dataset.lock",
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_parquet_dataset() -> datasets.Dataset:
        if self.episodes is None:
            path = str(self.root / "data")
            return load_dataset("parquet", data_dir=path, split="train")
        files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
        return load_dataset("parquet", data_files=files, split="train")

    cache_key = hashlib.sha1(f"{self.root.resolve()}::{self.episodes}".encode()).hexdigest()
    manifest_path = cache_dir / f".fastwam_parquet_manifest_{cache_key}.json"
    hf_dataset = None

    def _read_cache_manifest() -> list[str] | None:
        try:
            filenames = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not filenames or not all(Path(filename).is_file() for filename in filenames):
            return None
        return filenames

    cache_files = _read_cache_manifest()
    if cache_files is None:
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                cache_files = _read_cache_manifest()
                if cache_files is None:
                    hf_dataset = _load_parquet_dataset()
                    cache_files = [cache_file["filename"] for cache_file in hf_dataset.cache_files]
                    if not cache_files:
                        raise RuntimeError(f"No Arrow cache files produced for {self.root}")
                    manifest_tmp = manifest_path.with_name(
                        f"{manifest_path.name}.tmp.{os.getpid()}"
                    )
                    manifest_tmp.write_text(json.dumps(cache_files))
                    os.replace(manifest_tmp, manifest_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    if hf_dataset is None:
        cache_parts = [datasets.Dataset.from_file(filename) for filename in cache_files]
        hf_dataset = cache_parts[0] if len(cache_parts) == 1 else concatenate_datasets(cache_parts)

    # TODO(aliberts): hf_dataset.set_format("torch")
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset
