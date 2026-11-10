# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light declarations for the upstream LoopWAM training model."""

from turbo_physai import group, replace, wrap

_MODEL = "fastwam.models.wan22"
_TRAINER = "fastwam.trainer.Wan22Trainer"
_IMPL = "turbo_physai.optimizations.models.loopwam"
_DATA = "fastwam.datasets.lerobot.lerobot.lerobot_dataset.LeRobotDataset"

DATA_COLUMNS = group(
    "loopwam.dataset_columns",
    replace(f"{_DATA}.__init__", f"{_IMPL}.data.dataset_init"),
    *(replace(f"{_DATA}.{name}", f"{_IMPL}.data.{name}") for name in (
        "_get_query_timestamps", "_query_hf_dataset", "_query_hf_dataset_fast", "get_episode_data",
    )),
)
DATA_CACHE = group(
    "loopwam.dataset_cache",
    replace(f"{_DATA}.load_hf_dataset", f"{_IMPL}.data.load_hf_dataset"),
)
INITIALIZATION = group(
    "loopwam.initialization_barrier",
    wrap("fastwam.trainer.Accelerator", f"{_IMPL}.training.accelerator_barrier_wrapper"),
)

VAE_LAYOUT = group(
    "loopwam.vae_channels_last",
    replace(f"{_MODEL}.wan_video_vae.CausalConv3d.forward", f"{_IMPL}.kernels.causal_conv3d_forward"),
)
MASKED_ATTENTION = group(
    "loopwam.masked_flash_attention",
    replace(f"{_MODEL}.mot.MoT._mixed_attention", f"{_IMPL}.kernels.mixed_attention"),
)
RMSNORM = group(
    "loopwam.rmsnorm",
    replace(f"{_MODEL}.wan_video_dit.RMSNorm.forward", f"{_IMPL}.kernels.rmsnorm_forward"),
)

# Install on the prepared model at train-mode entry; every layer waits before
# its modulation/QKV reads. Evaluation and checkpointing drain pending work.
DEFERRED_GATHER = group(
    "loopwam.zero1_deferred_gather",
    wrap(f"{_TRAINER}._set_dit_only_train_mode", f"{_IMPL}.training.deferred_train_mode_wrapper"),
    wrap(f"{_MODEL}.mot.MoT._build_expert_attention_io", f"{_IMPL}.training.layer_wait_wrapper"),
    wrap(f"{_TRAINER}.evaluate", f"{_IMPL}.training.finish_gather_wrapper"),
    wrap(f"{_TRAINER}.save_checkpoint", f"{_IMPL}.training.finish_gather_wrapper"),
)
GRADIENT_OVERLAP = group(
    "loopwam.zero1_grad_overlap",
    wrap(f"{_TRAINER}.__init__", f"{_IMPL}.training.gradient_overlap_wrapper"),
)
NODE_SHARDING = group(
    "loopwam.zero1_node_sharding",
    wrap("accelerate.accelerator.Accelerator.prepare", f"{_IMPL}.training.sharded_prepare_wrapper"),
)
METRICS = group(
    "loopwam.fused_metrics",
    wrap(f"{_TRAINER}.train", f"{_IMPL}.training.fused_metrics_wrapper"),
)
